from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import HfArgumentParser, set_seed

from .chmm_src.CHMM.Data import collate_fn
from .chmm_src.CHMM.Train import CHMMTrainer
from .chmm_src.Src.Args import CHMMArguments, CHMMConfig
from ..basemodel import BaseSeqModel
from ..dataset import BaseSeqDataset


def one_hot(x, n_class=None):
    """
    x : LongTensor of shape (batch size, sequence max_seq_length)
    n_class : integer

    Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible x).
    """

    if n_class is None:
        n_class = np.max(x) + 1
    one_hot_vec = np.zeros([int(np.prod(x.shape)), n_class])
    indices = x.reshape([-1])
    one_hot_vec[np.arange(len(indices)), indices] = 1.0
    one_hot_vec = one_hot_vec.reshape(list(x.shape) + [n_class])
    return one_hot_vec


class CHMMTorchSeqDataset(Dataset):
    def __init__(self, dataset: BaseSeqDataset, for_train=True):
        self.data = [['[CLS]'] + item['text'] for item in dataset.examples]

        id2label = dataset.id2label

        if not for_train:
            self.labels = [['O'] + [id2label[i] for i in lb] for lb in dataset.labels]

        n_class = len(id2label)
        m = len(dataset.weak_labels[0][0])
        d_emb = dataset.bert_embeddings[0].shape[1]

        weak_labels = [one_hot(np.array(weak_labels_i), n_class=n_class) for weak_labels_i in dataset.weak_labels]
        prefix = torch.zeros([1, m, n_class])  # shape: 1, n_src, d_obs
        prefix[:, :, 0] = 1
        self.weak_labels = [torch.cat([prefix, torch.FloatTensor(inst)]) for inst in weak_labels]

        pad_embed = torch.zeros((1, d_emb))
        self.embs = [torch.cat([pad_embed, torch.FloatTensor(inst)]) for inst in dataset.bert_embeddings]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.data[idx], self.embs[idx], self.weak_labels[idx], self.labels[idx]
        else:
            return self.data[idx], self.embs[idx], self.weak_labels[idx]

    @property
    def obs(self):
        return self.weak_labels


class CHMM(BaseSeqModel):
    def __init__(self,
                 nn_lr: Optional[float] = 0.001,
                 hmm_lr: Optional[float] = 0.01,
                 batch_size: Optional[float] = 16,
                 num_nn_pretrain_epochs: Optional[int] = 2,
                 num_train_epochs: Optional[int] = 50,
                 num_valid_tolerance: Optional[int] = 5,
                 trans_nn_weight: Optional[float] = 1.0,
                 emiss_nn_weight: Optional[float] = 1.0,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            "num_nn_pretrain_epochs": num_nn_pretrain_epochs,
            "num_train_epochs"      : num_train_epochs,
            "num_valid_tolerance"   : num_valid_tolerance,
            "batch_size"            : batch_size,
            "trans_nn_weight"       : trans_nn_weight,
            "emiss_nn_weight"       : emiss_nn_weight,
            "nn_lr"                 : nn_lr,
            "hmm_lr"                : hmm_lr,
        }
        self.model = None

    def fit(self,
            dataset_train: Union[BaseSeqDataset],
            dataset_valid: Optional[BaseSeqDataset] = None,
            y_valid: Optional[List[List]] = None,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            seed: int = None,
            **kwargs: Any):
        assert hasattr(dataset_train, 'bert_embeddings')

        seed = seed or np.random.randint(1e6)
        set_seed(seed)

        self._update_hyperparas(**kwargs)
        hyperparas = deepcopy(self.hyperparas)
        hyperparas['seed'] = seed

        m = dataset_train.n_lf

        parser = HfArgumentParser(CHMMArguments)
        chmm_args, = parser.parse_dict(hyperparas)
        config = CHMMConfig().from_args(chmm_args)

        config.d_emb = dataset_train.bert_embeddings[0].shape[1]
        config.bio_label_types = dataset_train.id2label
        config.entity_types = dataset_train.entity_types
        config.sources = list(range(m))
        config.src_priors = {src: {lb: (0.7, 0.7) for lb in dataset_train.entity_types} for src in range(m)}

        chmm_trainer = CHMMTrainer(
            config=config,
            collate_fn=collate_fn,
            device=device,
            training_dataset=CHMMTorchSeqDataset(dataset_train, for_train=True),
            valid_dataset=CHMMTorchSeqDataset(dataset_valid, for_train=False),
            verbose=verbose
        ).initialize_trainer()

        _, self.model = chmm_trainer.train()

        self.device = device
        self.config = config

    def predict(self, dataset: BaseSeqDataset, batch_size=128, **kwargs: Any):
        test_loader = DataLoader(
            dataset=CHMMTorchSeqDataset(dataset, for_train=False),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )

        self.model.eval()

        preds = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])

                # get prediction
                pred_lb_indices, pred_probs = self.model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    normalize_observation=self.config.obs_normalization
                )
                # drop the first dummy label
                pred_lbs = [[lb_index for lb_index in label_indices[1:]] for label_indices in pred_lb_indices]
                preds += pred_lbs

        self.model.train()
        return preds
