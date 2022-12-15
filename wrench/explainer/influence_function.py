from typing import Optional

import numpy as np
import torch
import torch.utils.data as tdata
from backpack import backpack
from backpack.extensions import BatchGrad
from torch import nn
from torch.autograd import grad
from tqdm import tqdm, trange


def modify_training_labels(Y_hat, L, w, if_score, sample_ratio, sample_method='weight', normal_if=False, act_func='identity', normalize=False):
    if normal_if:

        if_score_flat = if_score.flatten()
        n_sample = int(len(if_score_flat) * sample_ratio)
        thres = np.partition(if_score_flat, -n_sample)[-n_sample]
        Y_hat_ = Y_hat * (if_score >= thres)



    else:

        if sample_method == 'relabel':
            sample_method = 'term'
            normalize = True

        n_lf = L.shape[1]
        w_idx = np.arange(n_lf)
        raw_score = w[w_idx, L + 1, :]

        normalizer = np.sum(raw_score, axis=1)
        if act_func == 'exp':
            normalizer = np.exp(normalizer)
        normalizer = np.sum(normalizer, axis=1, keepdims=True)

        if sample_method == 'term':
            # if_score_flat = if_score.flatten()
            if_score_flat = if_score[if_score != 0]
            n_sample = int(len(if_score_flat) * sample_ratio)
            thres = np.partition(if_score_flat, -n_sample)[-n_sample]
            idx = if_score >= thres


        elif sample_method == 'weight':

            assert act_func == 'identity'

            weight_to_remove = sample_ratio * len(Y_hat)

            if_score_flat = if_score.flatten()
            sort_idx = np.argsort(-if_score_flat)
            nor_score = raw_score / normalizer.reshape(-1, 1, 1)
            nor_score_cumsum = np.cumsum(nor_score.flatten()[sort_idx])

            unravel_index = np.unravel_index(sort_idx[nor_score_cumsum <= weight_to_remove], shape=if_score.shape)
            idx = np.zeros_like(if_score)
            idx[unravel_index] = 1

        elif sample_method == 'LF':
            # for LF, sameple ratio should be int -> top k LF
            normalize = False
            lf_if_score = if_score.sum(axis=(0, 2))
            lf_idx_to_use = (-lf_if_score).argsort()[:sample_ratio]
            idx = np.zeros_like(if_score)
            idx[:, lf_idx_to_use] = 1

        else:
            raise NotImplementedError

        filtered_score = raw_score * idx
        Y_hat_ = np.sum(filtered_score, axis=1)
        if act_func == 'exp':
            Y_hat_ = np.exp(Y_hat_)

        if normalize:
            Y_sum = np.sum(Y_hat_, axis=1, keepdims=True)
            Y_hat_ = np.divide(Y_hat_, Y_sum, where=Y_sum != 0)
            Y_hat_[Y_sum.flatten() == 0] = 0
        else:
            Y_hat_ = Y_hat_ / normalizer

    return Y_hat_


class IF(nn.Module):
    def __init__(self, model, tr_ds, val_ds, n_lf, n_class, device: Optional[torch.device] = None,
                 damp=1.0, scale=25.0, r=2, recursion_depth=100):
        super(IF, self).__init__()
        self.model = model
        self.damp = damp
        self.scale = scale
        self.r = r
        self.recursion_depth = recursion_depth
        self.train_dataloader_ = tdata.DataLoader(tr_ds, batch_size=len(tr_ds), shuffle=True)
        self.valid_dataloader_ = tdata.DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        self.num_train_samples = len(tr_ds)
        self.n_lf = n_lf
        self.n_class = n_class

        self.device = device
        self.eye_mat = torch.eye(self.n_class).to(device)
        self.w_idx = torch.arange(self.n_lf)

    def generate_x_y(self, w, x, l, y, act_func='identity', return_raw_score=False):
        raw_score = w[self.w_idx, l + 1, :]
        if act_func == 'identity':
            ex_x = x.repeat_interleave(self.n_lf * self.n_class, dim=0)
            nor_score = raw_score / torch.sum(raw_score, dim=(1, 2), keepdim=True)
            nor_score = nor_score.view(-1, self.n_class)
            ex_label = nor_score.unsqueeze(dim=2).repeat_interleave(self.n_class, dim=-1) * self.eye_mat
            ex_label = ex_label.view(-1, self.n_class)
            return ex_x, ex_label
        elif act_func == 'exp':
            ex_x = x.repeat_interleave(self.n_class, dim=0)
            ex_label = y.unsqueeze(dim=2).repeat_interleave(self.n_class, dim=-1) * self.eye_mat
            ex_label = ex_label.view(-1, self.n_class)
            if return_raw_score:
                return ex_x, ex_label, raw_score
            else:
                return ex_x, ex_label
        else:
            raise NotImplementedError

    def generate_renormalize_x_y(self, w, x, l, y, act_func='identity'):
        ex_x = x.repeat_interleave(self.n_lf * self.n_class, dim=0)
        raw_score = w[self.w_idx, l + 1, :]
        ex_raw_score = raw_score.unsqueeze(1).expand(-1, self.n_lf * self.n_class, -1, -1).clone()

        lf_range = torch.arange(self.n_lf).repeat_interleave(self.n_class)
        cls_range = torch.arange(self.n_class).repeat(self.n_lf)
        ex_raw_score[:, torch.arange(self.n_lf * self.n_class), lf_range, cls_range] = 0

        ex_raw_score = torch.sum(ex_raw_score, dim=-2)
        if act_func == 'exp':
            ex_raw_score = torch.exp(ex_raw_score)
        sum_raw_score = torch.sum(ex_raw_score, dim=-1, keepdim=True)
        nor_score = ex_raw_score / sum_raw_score
        nor_score[sum_raw_score.squeeze() == 0] = 0
        nor_score = y.unsqueeze(1) - nor_score
        ex_label = nor_score.view(-1, self.n_class)

        return ex_x, ex_label

    def outter_prod_inverse(self, input_vec):
        outter_result = input_vec @ input_vec.t()
        return torch.inverse(outter_result.cpu() + self.damp * torch.eye(outter_result.shape[0])).to(input_vec.device)

    def batch_hvp_v3(self, x, y, params_list, batch_grad_list):
        logit = self.model(x.to(self.device))
        loss = self.model.ce_loss(logit, y)
        if len(params_list) != len(batch_grad_list):
            raise (ValueError("w and v must have the same length."))

        one_sample_grad_list = list(grad(loss, params_list, create_graph=True, retain_graph=True))

        elemwise_products = 0
        for grad_elem, v_elem in zip(one_sample_grad_list, batch_grad_list):
            elemwise_products = elemwise_products + (grad_elem.reshape(-1, ) * v_elem).sum(dim=-1)

        second_order_grads = list(grad(elemwise_products, params_list))
        return_grads = [p.clone().detach().reshape(-1) for p in second_order_grads]
        del elemwise_products, second_order_grads
        torch.cuda.empty_cache()
        return return_grads

    def batch_s_test(self, batch_v, batch_h_estimate, mode, w, act_func):
        for i in range(self.recursion_depth):
            if mode == 'normal':
                # compute normal IF
                for idx, x, _, y in self.train_dataloader_:
                    self.model.zero_grad()
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    batch_hv = self.batch_hvp_v3(x, y, params, batch_h_estimate)
                    batch_temp_list = []
                    for _v, _h_e, _hv in zip(batch_v, batch_h_estimate, batch_hv):
                        _v, _h_e, _hv = _v.detach(), _h_e.detach(), _hv.detach()
                        batch_temp_list.append(_v + (1 - self.damp) * _h_e - _hv / self.scale)
                    del batch_h_estimate, batch_hv
                    torch.cuda.empty_cache()
                    batch_h_estimate = [i.clone() for i in batch_temp_list]
                    del batch_temp_list
                    torch.cuda.empty_cache()
                    break
            else:
                # compute fine-grained IF
                for idx, x, l, y in self.train_dataloader_:
                    self.model.zero_grad()
                    params = [p for p in self.model.parameters() if p.requires_grad]

                    if mode == 'RW':
                        ex_x, ex_label = self.generate_x_y(w, x, l, y, act_func=act_func)
                    elif mode == 'WM':
                        ex_x, ex_label = self.generate_renormalize_x_y(w, x, l, y, act_func=act_func)
                    else:
                        raise NotImplementedError

                    batch_hv = self.batch_hvp_v3(ex_x, ex_label, params, batch_h_estimate)
                    batch_temp_list = []
                    for _v, _h_e, _hv in zip(batch_v, batch_h_estimate, batch_hv):
                        _v, _h_e, _hv = _v.detach(), _h_e.detach(), _hv.detach()
                        batch_temp_list.append(_v + (1 - self.damp) * _h_e - _hv / self.scale)
                    del batch_h_estimate, batch_hv
                    torch.cuda.empty_cache()
                    batch_h_estimate = [i.clone() for i in batch_temp_list]
                    del batch_temp_list
                    torch.cuda.empty_cache()
                    break

        h_estimate_vec = torch.cat(batch_h_estimate, dim=0)
        return h_estimate_vec

    def compute_hv(self, val_grad_list, mode, w, act_func):
        s_test_vec_list = []
        for i in range(self.r):
            batch_v = val_grad_list
            batch_h_estimate = [h.clone().detach() for h in batch_v]
            s_test_vec_list.append(self.batch_s_test(batch_v, batch_h_estimate, mode, w, act_func).detach().unsqueeze(0))
        s_test_vec = torch.cat(s_test_vec_list, dim=0).mean(dim=0)
        return s_test_vec

    def compute_valid_grad_and_hv(self, mode, w, act_func, batch_mode):
        print('Computing grad over Valid set')
        self.model.zero_grad()

        if batch_mode:
            for idx, x, y in tqdm(self.valid_dataloader_):
                logit = self.model(x)
                loss = self.model.ce_loss_sum(logit, y)
                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()
            assert torch.isnan(batch_train_grad_vec).sum() == 0
            val_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)

        else:
            for idx, x, y in tqdm(self.valid_dataloader_):
                logit = self.model(x)
                loss = self.model.ce_loss_sum(logit, y)
                loss.backward()
            val_grad_vec = self.model.collect_grad()
            assert torch.isnan(val_grad_vec).sum() == 0
            val_grad_list = self.model.separate_batch_grad(val_grad_vec)

        self.model.zero_grad()

        print('Computing h_v')
        if batch_mode:
            s_test_vec_list = []
            for i in trange(len(val_grad_list[0])):
                s_test_vec = self.compute_hv([v[i] for v in val_grad_list], mode, w, act_func)
                s_test_vec_list.append(s_test_vec)
            s_test_vec = torch.stack(s_test_vec_list).T

        else:
            s_test_vec = self.compute_hv(val_grad_list, mode, w, act_func)

        return s_test_vec

    def compute_IF(self, if_type, mode='normal', w=None, act_func='identity', batch_mode=False):
        assert act_func in ['identity', 'exp']
        assert mode in ['normal', 'RW', 'WM']

        # # freeze some params
        # all_params = [p for p in self.model.parameters()]
        # if self.use_bottom_top_layer > 0:
        #     assert abs(self.use_bottom_top_layer) <= len(all_params)
        #     freeze_grad_params = all_params[0:-self.use_bottom_top_layer]
        # elif self.use_bottom_top_layer < 0:
        #     assert abs(self.use_bottom_top_layer) <= len(all_params)
        #     freeze_grad_params = all_params[-self.use_bottom_top_layer:]
        # else:
        #     freeze_grad_params = []
        # need_recover_params = []
        # for p in freeze_grad_params:
        #     if p.requires_grad:
        #         need_recover_params.append(p)
        #         p.requires_grad = False

        if if_type == 'if':
            train_if = self.compute_origin_IF(mode, w, act_func, batch_mode)
        elif if_type == 'sif':
            train_if = self.compute_self_IF(mode, w, act_func, batch_mode)
        elif if_type == 'relatif':
            train_if = self.compute_relat_IF(mode, w, act_func, batch_mode, return_all=False)
        elif if_type == 'all':
            train_if = self.compute_relat_IF(mode, w, act_func, batch_mode, return_all=True)
        else:
            raise NotImplementedError

        # # recover params
        # for p in need_recover_params:
        #     p.requires_grad = True

        return train_if

    def compute_origin_IF(self, mode, w=None, act_func='identity', batch_mode=False):
        s_test_vec = self.compute_valid_grad_and_hv(mode, w, act_func, batch_mode)

        print('Computing IF over training set')
        if mode == 'normal':
            # compute normal IF
            if batch_mode:
                n_test = s_test_vec.shape[1]
                train_if = torch.zeros(self.num_train_samples, n_test).to(self.device)
            else:
                train_if = torch.zeros(self.num_train_samples, 1).to(self.device)
                s_test_vec = s_test_vec.view(-1, 1)
            for idx, x, _, y in tqdm(self.train_dataloader_):
                self.model.zero_grad()
                logit = self.model(x)
                loss = self.model.ce_loss_sum(logit, y)

                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()

                if_score = (batch_train_grad_vec @ s_test_vec)
                train_if[idx] = if_score.detach()

        else:
            # compute fine-grained IF
            if batch_mode:
                n_test = s_test_vec.shape[1]
                train_if = torch.zeros(self.num_train_samples, self.n_lf, self.n_class, n_test).to(self.device)
            else:
                train_if = torch.zeros(self.num_train_samples, self.n_lf, self.n_class).to(self.device)

            if mode == 'RW':
                if act_func == 'identity':
                    for idx, x, l, y in tqdm(self.train_dataloader_):
                        self.model.zero_grad()
                        ex_x, ex_label = self.generate_x_y(w, x, l, y, act_func=act_func)
                        logit = self.model(ex_x)
                        loss = self.model.ce_loss_sum(logit, ex_label)

                        with backpack(BatchGrad()):
                            loss.backward()
                            batch_train_grad_vec = self.model.collect_batch_grad().detach()

                        if_score = batch_train_grad_vec.view(-1, self.n_lf, self.n_class, len(s_test_vec)) @ s_test_vec
                        train_if[idx] = if_score.detach()

                elif act_func == 'exp':

                    for idx, x, l, y in tqdm(self.train_dataloader_):
                        self.model.zero_grad()
                        ex_x, ex_label, raw_score = self.generate_x_y(w, x, l, y, act_func=act_func, return_raw_score=True)
                        logit = self.model(ex_x)
                        loss = self.model.ce_loss_sum(logit, ex_label)

                        with backpack(BatchGrad()):
                            loss.backward()
                            batch_train_grad_vec = self.model.collect_batch_grad().detach()

                        if batch_mode:
                            if_score = (batch_train_grad_vec @ s_test_vec).view(-1, self.n_class, n_test).unsqueeze(1) * raw_score.unsqueeze(-1)
                        else:
                            if_score = (batch_train_grad_vec @ s_test_vec).view(-1, self.n_class).unsqueeze(1) * raw_score
                        train_if[idx] = if_score.detach()  # taylor expension: e^x = 1 + x + o(x)

                else:
                    raise NotImplementedError

            elif mode == 'WM':

                for idx, x, l, y in tqdm(self.train_dataloader_):
                    self.model.zero_grad()
                    ex_x, ex_label = self.generate_renormalize_x_y(w, x, l, y, act_func=act_func)
                    logit = self.model(ex_x)
                    loss = self.model.ce_loss_sum(logit, ex_label)

                    with backpack(BatchGrad()):
                        loss.backward()
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                    batch_train_grad_vec = batch_train_grad_vec.reshape(len(x), self.n_lf * self.n_class, -1)
                    if_score = batch_train_grad_vec.view(-1, self.n_lf, self.n_class, len(s_test_vec)) @ s_test_vec
                    train_if[idx] = if_score.detach()

            else:
                raise NotImplementedError

        train_if = train_if / self.num_train_samples

        return train_if.cpu().numpy()

    def compute_self_IF(self, mode, w=None, act_func='identity', batch_mode=False, return_all=False):
        #### self influence computation
        # compute inverse Fisher
        num_layers = self.model.num_of_layers()
        inverse_block_diag = [0 for _ in range(num_layers)]

        if mode == 'normal':
            for idx, x, _, y in self.train_dataloader_:
                logit = self.model(x)
                for i in tqdm(range(x.shape[0])):
                    self.model.zero_grad()
                    _loss = self.model.ce_loss_sum(logit[i].unsqueeze(0), y[i].unsqueeze(0))
                    _loss.backward(retain_graph=True)
                    _train_grad = self.model.collect_grad()
                    _train_grad_list = self.model.separate_batch_grad(_train_grad)
                    for j in range(num_layers):
                        inverse_block_diag[j] = inverse_block_diag[j] + self.outter_prod_inverse(_train_grad_list[j].reshape(-1, 1))
                for j in range(num_layers):
                    inverse_block_diag[j] /= x.shape[0]
                break

        else:
            for idx, x, l, y in self.train_dataloader_:
                ex_x, ex_label = self.generate_x_y(w, x, l, y, act_func=act_func)
                logit = self.model(ex_x)
                cnt = 0
                for i in tqdm(range(ex_x.shape[0])):
                    if ex_label[i].sum() != 0:
                        cnt += 1
                        self.model.zero_grad()
                        _loss = self.model.ce_loss_sum(logit[i].unsqueeze(0), ex_label[i].unsqueeze(0))
                        _loss.backward(retain_graph=True)
                        _train_grad = self.model.collect_grad()
                        _train_grad_list = self.model.separate_batch_grad(_train_grad)
                        for j in range(num_layers):
                            inverse_block_diag[j] = inverse_block_diag[j] + self.outter_prod_inverse(_train_grad_list[j].reshape(-1, 1))
                for j in range(num_layers):
                    inverse_block_diag[j] /= cnt
                break

        if mode == 'normal':

            train_sif = torch.zeros(self.num_train_samples, 1).to(self.device)
            for idx, x, _, y in tqdm(self.train_dataloader_):
                self.model.zero_grad()
                logit = self.model(x)
                loss = self.model.ce_loss_sum(logit, y)
                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()
                    # TODO: not sure whether we can do this
                    train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                    temp = 0
                    for i in range(num_layers):
                        temp += (torch.mm(train_batch_grad_list[i], inverse_block_diag[i]) * train_batch_grad_list[i]).sum(dim=-1)
                train_sif[idx] = temp.view(-1, 1)

        else:

            train_sif = torch.zeros(self.num_train_samples, self.n_lf, self.n_class).to(self.device)

            if act_func == 'identity':

                for idx, x, l, y in tqdm(self.train_dataloader_):
                    self.model.zero_grad()
                    ex_x, ex_label = self.generate_x_y(w, x, l, y, act_func=act_func)
                    logit = self.model(ex_x)
                    loss = self.model.ce_loss_sum(logit, ex_label)
                    with backpack(BatchGrad()):
                        loss.backward()
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                        # TODO: not sure whether we can do this
                        train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                        temp = 0
                        for i in range(num_layers):
                            temp += (torch.mm(train_batch_grad_list[i], inverse_block_diag[i]) * train_batch_grad_list[i]).sum(dim=-1)
                    train_sif[idx] = temp.view(-1, self.n_lf, self.n_class)

            elif act_func == 'exp':

                for idx, x, l, y in tqdm(self.train_dataloader_):
                    self.model.zero_grad()
                    ex_x, ex_label, raw_score = self.generate_x_y(w, x, l, y, act_func=act_func, return_raw_score=True)
                    logit = self.model(ex_x)
                    loss = self.model.ce_loss_sum(logit, ex_label)

                    with backpack(BatchGrad()):
                        loss.backward(retain_graph=True)
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                        # TODO: not sure whether we can do this
                        train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                        hv = []
                        for i in range(num_layers):
                            hv.append(torch.mm(train_batch_grad_list[i], inverse_block_diag[i]))
                    hv = torch.cat(hv, dim=1)

                    # get i,c info
                    raw_score = raw_score + 1 / self.n_lf

                    ex_x = x.repeat_interleave(self.n_lf, dim=0)
                    self.model.zero_grad()
                    logit = self.model(ex_x)
                    finegrained_loss = self.model.ce_loss_sum(logit, raw_score.reshape(-1, self.n_class))
                    with backpack(BatchGrad()):
                        finegrained_loss.backward(retain_graph=True)
                        batch_train_grad_vec = self.model.collect_batch_grad().detach().view(len(x), self.n_lf, -1)
                    hv = hv.view(len(x), self.n_class, -1)
                    sif = (batch_train_grad_vec @ hv.transpose(2, 1)) * raw_score
                    train_sif[idx] = sif

            else:
                raise NotImplementedError

            if batch_mode:
                train_sif = train_sif.unsqueeze(-1)

        train_sif = train_sif.cpu().numpy()
        if np.any(train_sif < 0):
            train_sif -= np.min(train_sif)
        train_sif = np.sqrt(train_sif)
        return train_sif

    def compute_relat_IF(self, mode, w=None, act_func='identity', batch_mode=False, return_all=False):
        train_if = self.compute_origin_IF(mode, w, act_func, batch_mode)
        train_sif = self.compute_self_IF(mode, w, act_func, batch_mode)

        train_ratif = np.divide(train_if, train_sif, where=train_sif != 0)
        train_ratif[train_sif == 0] = 0

        if return_all:
            return train_if, train_ratif, train_sif
        else:
            return train_ratif
