from typing import Type, Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

from ..backbone import BackBone, MLP
from ..basemodel import BaseModel, BaseTorchModel
from ..labelmodel import BaseLabelModel, MajorityVoting
from ..dataset import BaseDataset, TorchDataset
from ..dataset.utils import check_dataset, split_labeled_unlabeled


logger = logging.getLogger(__name__)


class AttentionModel(nn.Module):
    def __init__(self, input_size, n_rules, hidden_size, n_class):
        super(AttentionModel, self).__init__()
        self.n_class = n_class
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_rules)

    def forward(self, x_lf, batch):
        x_l = batch['features'].to(x_lf.device)
        x = torch.cat((x_lf, x_l), 1)
        z = self.fc2(torch.tanh(self.fc1(x)))
        score = F.softmax(z, dim=1)
        mask = (x_lf >= 0).float()
        coverage_score = score * mask

        score_matrix = torch.empty(len(x_lf), self.n_class, device=x_lf.device)
        for k in range(self.n_class):
            score_matrix[:, k] = (score * (x_lf == k).float()).sum(dim=1)

        # softmax_new_y = F.log_softmax(score_matrix, dim=1)
        # return softmax_new_y, coverage_score
        return score_matrix, coverage_score


class AssembleModel(nn.Module):
    def __init__(self, input_size, n_rules, hidden_size, n_class, backbone):
        super(AssembleModel, self).__init__()
        self.n_class = n_class
        self.backbone_model = backbone
        self.attention = AttentionModel(input_size + n_rules, n_rules, hidden_size, n_class) #TODO
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, batch_l, batch_u, x_lf_l, x_lf_u):
        predict_l = self.backbone_model(batch_l)
        predict_u = self.backbone_model(batch_u)

        lf_y_l, all_scores = self.attention(x_lf_l, batch_l)
        fix_score = F.softmax(torch.mean(all_scores, dim=0), dim=0)  # use the average as the fixed score

        lf_y_u = torch.zeros((x_lf_u.size(0), self.n_class), dtype=torch.float, device=self.dummy_param.device)
        for k in range(self.n_class):
            lf_y_u[:, k] = (fix_score.unsqueeze(0).repeat([x_lf_u.size(0), 1]) * (x_lf_u == k).float()).sum(dim=1)
        lf_y_u /= torch.sum(lf_y_u, dim=1).unsqueeze(1)
        lf_y_u[lf_y_u != lf_y_u] = 0  # handle the 'nan' (divided by 0) problem
        lf_y_u = F.log_softmax(lf_y_u, dim=1).detach()

        return predict_l, predict_u, lf_y_l, lf_y_u, fix_score.detach()


class Denoise(BaseTorchModel):
    def __init__(self,
                 lr: Optional[float] = 1e-3,
                 l2: Optional[float] = 0.0,
                 alpha: Optional[float] = 0.6,
                 c1: Optional[float] = 0.2,
                 c2: Optional[float] = 0.7,
                 batch_size: Optional[int] = 32,
                 test_batch_size: Optional[int] = 128,
                 hidden_size: Optional[int] = 100,
                 n_steps: Optional[int] = 10000):
        super().__init__()
        self.hyperparas = {
            'lr': lr,
            'l2': l2,
            'alpha': alpha,
            'c1': c1,
            'c2': c2,
            'batch_size': batch_size,
            'test_batch_size': test_batch_size,
            'hidden_size': hidden_size,
            'n_steps': n_steps,
        }
        self.model: Optional[AssembleModel] = None
        self.label_model: Optional[BaseLabelModel] = None

    def fit(self,
            dataset_train:BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            label_model_class: Type[BaseLabelModel] = MajorityVoting,
            label_model_config: Dict = None,
            backbone_class: Type[BackBone] = MLP,
            backbone_config: Dict = None,
            cut_tied: Optional[bool] = True,
            valid_mode: Optional[str] = 'feature',
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        label_model_config = label_model_config or {}
        for k, v in kwargs.items():
            if k.startswith('label_model_config_'):
                k = k.replace('label_model_config_', '')
                label_model_config[k] = v
        backbone_config = backbone_config or {}
        for k, v in kwargs.items():
            if k.startswith('backbone_config_'):
                k = k.replace('backbone_config_', '')
                backbone_config[k] = v

        if not verbose:
            logger.setLevel(logging.ERROR)

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas
        n_steps = hyperparas['n_steps']
        alpha, c1, c2 = hyperparas['alpha'], hyperparas['c1'], hyperparas['c2']

        n_rules = len(dataset_train.weak_labels[0])
        n_class = len(dataset_train.id2label)
        input_size = dataset_train.features.shape[1]

        labeled_dataset, unlabeled_dataset = split_labeled_unlabeled(dataset_train, cut_tied=cut_tied)
        labeled_dataloader = DataLoader(TorchDataset(labeled_dataset, n_data=n_steps*hyperparas['batch_size']),
                                        batch_size=hyperparas['batch_size'], shuffle=True)
        unlabeled_dataloader = DataLoader(TorchDataset(unlabeled_dataset, n_data=n_steps*hyperparas['batch_size']),
                                          batch_size=hyperparas['batch_size'], shuffle=True)
        unlabel_dataloader_iterator = iter(unlabeled_dataloader)

        backbone_config['input_size'] = input_size
        backbone_config['n_class'] = n_class
        backbone = backbone_class(**backbone_config)
        model = AssembleModel(
            input_size=input_size,
            n_rules=n_rules,
            hidden_size=hyperparas['hidden_size'],
            n_class=n_class,
            backbone=backbone
        ).to(device)
        self.model = model

        optimizer = optim.Adam(
            model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2']
        )

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        label_model = label_model_class(**label_model_config)
        label_model.fit(dataset_train=dataset_train, dataset_valid=dataset_valid, verbose=verbose)
        self.label_model = label_model
        all_y_l = torch.LongTensor(label_model.predict(labeled_dataset)).to(device)
        all_Z = torch.zeros(len(unlabeled_dataset), n_class, dtype=torch.float).to(device)
        all_z = torch.zeros(len(unlabeled_dataset), n_class, dtype=torch.float).to(device)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, ncols=200, desc="training Denoise model", unit="steps", disable=not verbose) as pbar:
                model.train()
                step = 0
                for label_batch in labeled_dataloader:
                    step += 1
                    try:
                        unlabel_batch = next(unlabel_dataloader_iterator)
                    except StopIteration:
                        unlabel_dataloader_iterator = iter(unlabeled_dataloader)
                        unlabel_batch = next(unlabel_dataloader_iterator)

                    x_lf_l = label_batch['weak_labels'].to(device)
                    x_lf_u = unlabel_batch['weak_labels'].to(device)
                    idx_l = label_batch['ids'].long().to(device)
                    idx_u = unlabel_batch['ids'].long().to(device)
                    y_l = all_y_l.index_select(0, idx_l)
                    Z = all_Z.index_select(0, idx_u)
                    z = all_z.index_select(0, idx_u)

                    optimizer.zero_grad()

                    predict_l, predict_u, lf_y_l, lf_y_u, fix_score = model(label_batch, unlabel_batch, x_lf_l, x_lf_u)

                    loss_sup = F.cross_entropy(predict_l, y_l)
                    loss_sup_weight = F.cross_entropy(lf_y_l, y_l)

                    loss_unsup = torch.FloatTensor([0.0]).to(device)
                    if step > 100:
                        loss_unsup = ((z - predict_u) ** 2).mean()
                        outputs = predict_u.data.clone()
                        Z = alpha * Z + (1. - alpha) * outputs
                        all_z[idx_u] = Z * (1. / (1. - alpha ** (step + 1)))
                        all_Z[idx_u] = Z

                    loss = c1 * loss_sup_weight + c2 * loss_sup + (1 - c2 - c1) * loss_unsup

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # --- update y_l ---
                    lf_y_l_new = torch.zeros((x_lf_l.size(0), n_class), dtype=torch.float).to(device)
                    for k in range(n_class):
                        lf_y_l_new[:, k] = (fix_score.unsqueeze(0).repeat([x_lf_l.size(0), 1]) * (x_lf_l == k).float()).sum(dim=1)
                    lf_y_l_new /= torch.sum(lf_y_l_new, dim=1).unsqueeze(1)
                    lf_y_l_new[lf_y_l_new != lf_y_l_new] = 0  # handle the 'nan' (divided by 0) problem
                    lf_y_l_new = F.log_softmax(lf_y_l_new, dim=1).detach()
                    all_y_l[idx_l] = lf_y_l_new.max(1)[1]

                    if valid_flag and step % evaluation_step == 0:
                        metric_value, early_stop_flag, info = self._valid_step(step, mode=valid_mode)
                        if early_stop_flag:
                            logger.info(info)
                            break

                        history[step] = {
                            'loss': loss.item(),
                            'loss_sup': loss_sup.item(),
                            'loss_sup_weight': loss_sup_weight.item(),
                            'loss_unsup': loss_unsup.item(),
                            f'val_{metric}': metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            f'best_step': self.best_step,
                        }
                        last_step_log.update(history[step])

                    last_step_log['loss'] = loss.item()
                    last_step_log['loss_sup'] = loss_sup.item()
                    last_step_log['loss_sup_weight'] = loss_sup_weight.item()
                    last_step_log['loss_unsup'] = loss_unsup.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)
        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history

    def predict_proba(self, dataset:Union[BaseDataset, DataLoader], mode: Optional[str] = 'feature',
                      device: Optional[torch.device] = None, **kwargs: Any):
        assert mode in ['ensemble', 'feature', 'rules'], f'mode: {mode} not support!'
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
            device = model.dummy_param.device
        model.eval()
        with torch.no_grad():
            if isinstance(dataset, BaseDataset):
                valid_dataloader = DataLoader(TorchDataset(dataset), batch_size=self.hyperparas['test_batch_size'], shuffle=False)
            else:
                valid_dataloader = dataset
            probas = []
            for batch in valid_dataloader:
                if mode == 'ensemble':
                    output1, x = model.backbone_model(batch, return_features=True)
                    prob_feature = F.softmax(output1, dim=-1)
                    x_lf = batch['weak_labels'].to(device)
                    output2, _ = model.attention(x_lf, batch)
                    prob_weak_labels = F.softmax(output2, dim=-1)
                    proba = 0.5 * (prob_feature + prob_weak_labels)
                elif mode == 'feature':
                    output1 = model.backbone_model(batch)
                    proba = F.softmax(output1, dim=-1)
                elif mode == 'rules':
                    _, x = model.backbone_model(batch, return_features=True)
                    x_lf = batch['weak_labels'].to(device)
                    output2, _ = model.attention(x_lf, x)
                    proba = F.softmax(output2, dim=-1)
                else:
                    raise NotImplementedError

                probas.append(proba.cpu().numpy())

        return np.vstack(probas)