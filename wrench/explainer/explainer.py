from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from backpack import extend
from scipy.optimize import least_squares
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .influence_function import IF

ABSTAIN = -1

activation_func_dict = {
    'identity': lambda x: x,
    'exp'     : lambda x: np.exp(x),
    'tylor'   : lambda x: 1 + x + 0.5 * (x ** 2),
}


class AbstractModel(torch.nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def collect_grad(self):
        return torch.cat([p.grad.reshape(-1, ) for p in self.parameters() if p.requires_grad], dim=0)

    def collect_batch_grad(self, params=None):
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache

    def num_of_layers(self):
        n_layer = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            else:
                n_layer += 1
        return n_layer

    def separate_batch_grad(self, batch_grad_cache, params=None):
        num_param_per_layer = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    temp_num_param = np.prod(list(param.shape))
                    num_param_per_layer.append(temp_num_param)
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    temp_num_param = np.prod(list(param.shape))
                    num_param_per_layer.append(temp_num_param)

        grad_per_layer_list = []
        counter = 0
        for num_param in num_param_per_layer:
            if len(batch_grad_cache.shape) == 2:
                grad_per_layer_list.append(batch_grad_cache[:, counter:counter + num_param])
            else:
                grad_per_layer_list.append(batch_grad_cache[counter:counter + num_param])
            counter += num_param

        return grad_per_layer_list


class LinearModel(AbstractModel):
    def __init__(self, input_size, n_class):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        if n_class == 2:
            self.output_size = 1
        else:
            self.output_size = n_class
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)
        self.ce_loss = CrossEntropyLoss()
        self.ce_loss_sum = CrossEntropyLoss(reduction='sum')

        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        output = self.fc1(x.reshape(x.shape[0], -1))
        if self.output_size == 1:
            output = torch.cat([F.logsigmoid(-output), F.logsigmoid(output)], dim=-1)
        return output


def train_model(lr, weight_decay, epochs, input_size, n_class, train_dataloader, device, verbose=True):
    model = LinearModel(input_size, n_class)
    model.to(device)

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    for epoch in trange(epochs, disable=not verbose, desc='model training'):
        for _, x, y in train_dataloader:
            model.train()
            model.zero_grad()
            logit = model(x)
            loss = model.ce_loss(logit, y)
            loss.backward()
            optimizer.step()

    return model


class Explainer:
    def __init__(self,
                 n_lf: int,
                 n_class: int,
                 **kwargs: Any):
        self.n_lf = n_lf
        self.n_class = n_class
        self.w = None
        self.activation_func = None

    def augment_label_matrix(self, L):
        L = np.array(L) + 1
        L_aug = np.eye(self.n_class + 1)[L]
        return L_aug

    def approximate_label_model(self, L, y, w0=None):
        L_aug = self.augment_label_matrix(L)

        N, M, C = L_aug.shape
        C = C - 1

        L_aug_ = L_aug.reshape(N, -1)
        if w0 is not None:
            x0 = w0 - w0.min(axis=-1, keepdims=True)
        else:
            x0 = np.zeros(shape=(M, C + 1, C))

        def func(x):
            x = x.reshape(M * (C + 1), C)
            P = L_aug_ @ x
            Z = np.sum(P, axis=1, keepdims=True)
            y_hat = P / Z
            return (y_hat - y).flatten()

        res = least_squares(func, x0.flatten(), bounds=(0, np.inf))

        approx_w = res.x.reshape(x0.shape)
        self.register_label_model(approx_w, 'identity')
        return approx_w

    def register_label_model(self, w, activation_func='identity'):
        assert activation_func in ['identity', 'exp']
        self.w = w
        self.activation_func = activation_func

    def apply_label_model(self, L):
        L_aug = self.augment_label_matrix(L)
        raw_score = np.einsum('ijk,jkl->il', L_aug, self.w)
        raw_score = activation_func_dict[self.activation_func](raw_score)
        Z = np.sum(raw_score, axis=1, keepdims=True)
        y_hat = raw_score / Z
        return y_hat

    def compute_IF_score(self, L_tr, X_tr, X_te, Y_te, if_type, mode,
                   lr, weight_decay, epochs, batch_size, device: Optional[torch.device] = None,
                   damp=1.0, scale=25.0, r=2, recursion_depth=100
                   ):
        Y_tr = self.apply_label_model(L_tr)

        X_tr = np.array(X_tr)

        X_te = np.array(X_te)
        Y_te = np.eye(self.n_class)[np.array(Y_te)]

        # construct dataloaders
        tr_data = TensorDataset(torch.LongTensor(list(range(X_tr.shape[0]))).to(device),
                                torch.FloatTensor(X_tr).to(device), torch.FloatTensor(Y_tr).to(device))
        train_dataloader = DataLoader(tr_data, batch_size=len(tr_data) if batch_size == -1 else batch_size, shuffle=True)

        te_data = TensorDataset(torch.LongTensor(list(range(X_te.shape[0]))).to(device),
                                torch.FloatTensor(X_te).to(device), torch.FloatTensor(Y_te).to(device))

        model = train_model(lr, weight_decay, epochs, X_tr.shape[1], self.n_class, train_dataloader, device, verbose=True)

        model = extend(model)
        model.ce_loss_sum = extend(model.ce_loss_sum)

        tr_data_for_comp_if = TensorDataset(
            torch.LongTensor(list(range(X_tr.shape[0]))).to(device),
            torch.FloatTensor(X_tr).to(device),
            torch.LongTensor(L_tr).to(device),
            torch.FloatTensor(Y_tr).to(device)
        )

        IF_func = IF(model, tr_data_for_comp_if, te_data, self.n_lf, self.n_class, device, damp=damp, scale=scale, r=r, recursion_depth=recursion_depth)

        IF_score = IF_func.compute_IF(if_type=if_type, mode=mode, w=torch.FloatTensor(self.w).to(device), act_func=self.activation_func)

        return IF_score
