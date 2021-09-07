import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from seqeval import metrics
from seqeval.scheme import IOB2
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .Model import CHMM
from ..Src.Args import CHMMConfig
from ..Src.DataAssist import initialise_transmat, initialise_emissions, span_to_label


class CHMMTrainer:
    def __init__(self,
                 config: CHMMConfig,
                 collate_fn,
                 device,
                 training_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None,
                 optimizer=None,
                 verbose=True):

        self._model = None
        self._config = config
        self._device = device
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._pretrain_optimizer = pretrain_optimizer
        self._optimizer = optimizer
        self._state_prior = None
        self._trans_mat = None
        self._emiss_mat = None
        logger = logging.getLogger(__name__)
        if not verbose:
            logger.setLevel(logging.ERROR)
        self._logger = logger
        self._verbose = verbose

        self._has_appended_obs = False  # reserved for alternate training

    def initialize_trainer(self):
        """
        Initialize necessary components for training

        Returns
        -------
        the initialized trainer
        """
        self.initialize_matrices()
        self.initialize_model()
        self.initialize_optimizers()
        return self

    def initialize_matrices(self):
        """
        Initialize <HMM> transition and emission matrices

        Returns
        -------
        self
        """
        assert self._training_dataset and self._valid_dataset
        # inject prior knowledge about transition and emission
        self._state_prior = torch.zeros(self._config.d_hidden, device=self._device) + 1e-2
        self._state_prior[0] += 1 - self._state_prior.sum()

        intg_obs = list(map(np.array, self._training_dataset.obs + self._valid_dataset.obs))
        self._logger.info("Constructing transition matrix prior...")
        self._trans_mat = torch.tensor(initialise_transmat(
            observations=intg_obs, label_set=self._config.bio_label_types)[0], dtype=torch.float)
        self._logger.info("Constructing emission probabilities...")
        self._emiss_mat = torch.tensor(initialise_emissions(
            observations=intg_obs, label_set=self._config.bio_label_types,
            sources=self._config.sources, src_priors=self._config.src_priors
        )[0], dtype=torch.float)
        return self

    def initialize_model(self):
        self._logger.info('Initializing CHMM...')
        self._model = CHMM(
            config=self._config,
            device=self._device,
            state_prior=self._state_prior,
            trans_matrix=self._trans_mat,
            emiss_matrix=self._emiss_mat
        )
        self._logger.info("CHMM initialized!")
        return self

    def initialize_optimizers(self, optimizer=None, pretrain_optimizer=None):
        self._optimizer = self.get_optimizer() if optimizer is None else optimizer
        self._pretrain_optimizer = self.get_pretrain_optimizer() if pretrain_optimizer is None else pretrain_optimizer

    def pretrain_step(self, data_loader, optimizer, trans_, emiss_):
        train_loss = 0
        num_samples = 0

        self._model._nn_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self._device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self._device)

        for i, batch in enumerate(tqdm(data_loader, disable=not self._verbose)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss = self._model._nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()
            n_obs = nn_emiss.size(-1)

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self._device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            emiss_mask = loss_mask.view(batch_size, max_seq_len, 1, 1, 1)
            emiss_pred = emiss_mask * nn_emiss
            emiss_true = emiss_mask * emiss_.view(
                1, 1, self._config.n_src, n_hidden, n_obs
            ).repeat(batch_size, max_seq_len, 1, 1, 1)
            if trans_ is not None:
                l1 = F.mse_loss(trans_pred, trans_true)
            else:
                l1 = 0
            if emiss_ is not None:
                l2 = F.mse_loss(emiss_pred, emiss_true)
            else:
                l2 = 0
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        return train_loss

    def training_step(self, data_loader, optimizer):
        train_loss = 0
        num_samples = 0

        self._model.train()

        for i, batch in enumerate(tqdm(data_loader, disable=not self._verbose)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, _ = self._model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens,
                normalize_observation=self._config.obs_normalization
            )

            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples

        return train_loss

    def train(self):
        training_dataloader = self.get_training_dataloader()
        eval_dataloader = self.get_valid_dataloader()

        # ----- pre-train neural module -----
        if self._config.num_nn_pretrain_epochs > 0:
            self._logger.info(" ----- \nPre-training neural module...")
            for epoch_i in range(self._config.num_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._trans_mat, self._emiss_mat
                )
                self._logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            self._logger.info("Neural module pretrained!")

        valid_results = list()
        best_f1 = 0
        tolerance_epoch = 0
        best_checkpoint = self.return_model_checkpoint()
        # ----- start training process -----
        self._logger.info(" ----- \nStart training CHMM...")
        for epoch_i in range(self._config.num_train_epochs):
            self._logger.info("------")
            self._logger.info(f"Epoch {epoch_i + 1} of {self._config.num_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer)
            results = self.evaluate(eval_dataloader)

            self._logger.info("Training loss: %.4f" % train_loss)
            self._logger.info("Validation results:")
            self._logger.info(f"\tPrecision: {results[0]:.4f}")
            self._logger.info(f"\tRecall: {results[1]:.4f}")
            self._logger.info(f"\tF1: {results[2]:.4f}")

            # ----- save model -----
            if results[2] > best_f1:
                # self.save_model()
                # self._logger.info("Checkpoint Saved!\n")
                best_checkpoint = self.return_model_checkpoint()
                best_f1 = results[2]
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            valid_results.append(results)
            if tolerance_epoch >= self._config.num_valid_tolerance:
                self._logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        # self.load_model()
        self.load_from_checkpoint(best_checkpoint)

        return valid_results, self._model

    def test(self):
        test_dataloader = self.get_test_dataloader()
        test_results = self.evaluate(test_dataloader)
        return test_results

    def evaluate(self, data_loader):

        self._model.eval()

        metric_values = np.zeros(3)  # precision, recall, f1
        preds = []
        lbs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, disable=not self._verbose)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._device), batch[:3])
                lbs_batch = batch[-1]

                # get prediction
                pred_lb_indices, pred_probs = self._model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    normalize_observation=self._config.obs_normalization
                )
                pred_lbs = [[self._config.bio_label_types[lb_index] for lb_index in label_indices]
                            for label_indices in pred_lb_indices]
                preds += pred_lbs
                lbs += lbs_batch
            metric_values[0] = metrics.precision_score(lbs, preds, mode='strict', scheme=IOB2)  # * \
            # len(lbs_batch)
            metric_values[1] = metrics.recall_score(lbs, preds, mode='strict', scheme=IOB2)  # * \
            # len(lbs_batch)
            metric_values[2] = metrics.f1_score(lbs, preds, mode='strict', scheme=IOB2)  # * \

        return metric_values

    def annotate_data(self, partition, save_dir=''):
        if partition == 'train':
            data_loader = self.get_training_dataloader(shuffle=False)
        elif partition == 'eval':
            data_loader = self.get_valid_dataloader()
        elif partition == 'test':
            data_loader = self.get_test_dataloader()
        else:
            raise ValueError("[CHMM] invalid data partition")

        score_list = list()
        span_list = list()  # prediction on span-level, format: {(start_id, end_id): [(token_type, prob)] }
        pred_list = list()
        txt_list = list()
        label_list = list()  # prediction on token-level
        save_dict = {}
        self._model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, disable=not self._verbose)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._device), batch[:3])
                # get prediction
                # the scores are shifted back, i.e., len = len(emb)-1 = len(sentence)
                _, (scored_spans, scores) = self._model.annotate(
                    emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_types=self._config.bio_label_types,
                    normalize_observation=self._config.obs_normalization
                )
                score_list += scores
                span_list += scored_spans  # span lists are shifted here
                txt_list += batch[-2]
                for token, labeled_spans in zip(batch[-1], scored_spans):
                    batch_pred = span_to_label(token[1:], labeled_spans)
                    pred_list.append(batch_pred)
                    label_list.append(token[1:])
                # print(len(span_list), span_list[-1], seq_lens, batch[-1], span_to_label(tokens = batch[-1][-1], labeled_spans = span_list[-1]))

        with open(os.path.join(save_dir, partition + '_preds.json'), 'w') as f:
            for i in range(len(label_list)):
                save_dict[i] = {"data": {"text": txt_list[i][1:]}, "pred": pred_list[i], "label": label_list[i]}
            json.dump(save_dict, f, indent=2)

        return span_list, score_list

    def get_training_dataloader(self, shuffle=True):
        if self._training_dataset:
            training_loader = DataLoader(
                dataset=self._training_dataset,
                batch_size=self._config.batch_size,
                collate_fn=self._collate_fn,
                shuffle=shuffle,
                drop_last=False
            )
            return training_loader
        else:
            raise ValueError("Training dataset is not defined!")

    def get_valid_dataloader(self):
        if self._valid_dataset:
            eval_loader = DataLoader(
                dataset=self._valid_dataset,
                batch_size=self._config.batch_size,
                collate_fn=self._collate_fn,
                shuffle=False,
                drop_last=False
            )
            return eval_loader
        else:
            raise ValueError("Validation dataset is not defined!")

    def get_test_dataloader(self):
        if self._test_dataset:
            test_loader = DataLoader(
                dataset=self._test_dataset,
                batch_size=self._config.batch_size,
                collate_fn=self._collate_fn,
                shuffle=False,
                drop_last=False
            )
            return test_loader
        else:
            raise ValueError("Test dataset is not defined!")

    def get_pretrain_optimizer(self):
        pretrain_optimizer = torch.optim.Adam(
            self._model._nn_module.parameters(),
            lr=5e-4,
            weight_decay=1e-5
        )
        return pretrain_optimizer

    def get_optimizer(self):
        # ----- initialize optimizer -----
        hmm_params = [
            self._model._unnormalized_emiss,
            self._model._unnormalized_trans,
            self._model._state_priors
        ]
        optimizer = torch.optim.Adam(
            [{'params': self._model._nn_module.parameters(), 'lr': self._config.nn_lr},
             {'params': hmm_params}],
            lr=self._config.hmm_lr,
            weight_decay=1e-5
        )
        return optimizer

    def return_model_checkpoint(self):
        model_state_dict = self._model.state_dict()
        optimizer_state_dict = self._optimizer.state_dict()
        pretrain_optimizer_state_dict = self._pretrain_optimizer.state_dict()
        checkpoint = {
            'model'             : model_state_dict,
            'optimizer'         : optimizer_state_dict,
            'pretrain_optimizer': pretrain_optimizer_state_dict,
            'state_prior'       : self._state_prior,
            'transitions'       : self._trans_mat,
            'emissions'         : self._emiss_mat,
            'config'            : self._config
        }
        return checkpoint

    def load_from_checkpoint(self, checkpoint):
        self._model.load_state_dict(checkpoint['model'])

    def save_model(self, model_dir: Optional[str] = None):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        model_dir: model directory

        Returns
        -------
        None
        """
        model_state_dict = self._model.state_dict()
        optimizer_state_dict = self._optimizer.state_dict()
        pretrain_optimizer_state_dict = self._pretrain_optimizer.state_dict()
        checkpoint = {
            'model'             : model_state_dict,
            'optimizer'         : optimizer_state_dict,
            'pretrain_optimizer': pretrain_optimizer_state_dict,
            'state_prior'       : self._state_prior,
            'transitions'       : self._trans_mat,
            'emissions'         : self._emiss_mat,
            'config'            : self._config
        }
        if model_dir:
            torch.save(checkpoint, model_dir)
        else:
            torch.save(checkpoint, os.path.join(self._config.output_dir, 'chmm.bin'))

    def load_model(self, model_dir: Optional[str] = None, load_trainer_params: Optional[bool] = False):
        """
        Load model parameters.

        Parameters
        ----------
        model_dir: model directory
        load_trainer_params: whether load other trainer parameters

        Returns
        -------
        self
        """
        if model_dir:
            checkpoint = torch.load(model_dir)
        else:
            checkpoint = torch.load(os.path.join(self._config.output_dir, 'chmm.bin'))
        self._model.load_state_dict(checkpoint['model'])
        self._config = checkpoint['config']
        if load_trainer_params:
            self._optimizer.load_state_dict([checkpoint['optimizer']])
            self._pretrain_optimizer.load_state_dict([checkpoint['pretrain_optimizer']])
            self._state_prior = checkpoint['state_prior']
            self._trans_mat = checkpoint['transitions']
            self._emiss_mat = checkpoint['emissions']
        return self
