import logging
from typing import Optional

import torch
import torch.nn as nn

from ..Src.Args import CHMMConfig
from ..Src.Utils import log_matmul, log_maxmul, validate_prob, logsumexp

logger = logging.getLogger(__name__)


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 d_hidden,
                 n_src,
                 d_obs):
        super(NeuralModule, self).__init__()

        self._d_hidden = d_hidden
        self._n_src = n_src
        self._d_obs = d_obs
        self._neural_transition = nn.Linear(d_emb, self._d_hidden * self._d_hidden)
        self._neural_emissions = nn.ModuleList([
            nn.Linear(d_emb, self._d_hidden * self._d_obs) for _ in range(self._n_src)
        ])

        self._init_parameters()

    def forward(self,
                embs: torch.Tensor,
                temperature: Optional[int] = 1.0):
        batch_size, max_seq_length, _ = embs.size()
        trans_temp = self._neural_transition(embs).view(
            batch_size, max_seq_length, self._d_hidden, self._d_hidden
        )
        nn_trans = torch.softmax(trans_temp / temperature, dim=-1)

        nn_emiss = torch.stack([torch.softmax(emiss(embs).view(
            batch_size, max_seq_length, self._d_hidden, self._d_obs
        ) / temperature, dim=-1) for emiss in self._neural_emissions]).permute(1, 2, 0, 3, 4)
        return nn_trans, nn_emiss

    def _init_parameters(self):
        nn.init.xavier_uniform_(self._neural_transition.weight.data, gain=nn.init.calculate_gain('relu'))
        for emiss in self._neural_emissions:
            nn.init.xavier_uniform_(emiss.weight.data, gain=nn.init.calculate_gain('relu'))


class CHMM(nn.Module):

    def __init__(self,
                 config: CHMMConfig,
                 device,
                 state_prior=None,
                 trans_matrix=None,
                 emiss_matrix=None):
        super(CHMM, self).__init__()

        self._d_emb = config.d_emb  # embedding dimension
        self._n_src = config.n_src
        self._d_obs = config.d_obs  # number of possible obs_set
        self._n_hidden = config.d_hidden  # number of states

        self._trans_weight = config.trans_nn_weight
        self._emiss_weight = config.emiss_nn_weight

        self._device = device

        self._nn_module = NeuralModule(
            d_emb=self._d_emb, d_hidden=self._n_hidden, n_src=self._n_src, d_obs=self._d_obs
        )

        # initialize unnormalized state-prior, transition and emission matrices
        self._initialize_model(
            state_prior=state_prior, trans_matrix=trans_matrix, emiss_matrix=emiss_matrix
        )
        self.to(self._device)

    @property
    def log_trans(self):
        try:
            return self._log_trans
        except NameError:
            raise NameError('CHMM.log_trans is not defined!')

    @property
    def log_emiss(self):
        try:
            return self._log_emiss
        except NameError:
            raise NameError('CHMM.log_emiss is not defined!')

    def _initialize_model(self,
                          state_prior: torch.Tensor,
                          trans_matrix: torch.Tensor,
                          emiss_matrix: torch.Tensor):
        """
        Initialize model parameters

        Parameters
        ----------
        state_prior: state prior (pi)
        trans_matrix: transition matrices
        emiss_matrix: emission matrices

        Returns
        -------
        self
        """

        if state_prior is None:
            priors = torch.zeros(self._n_hidden, device=self._device) + 1E-3
            priors[0] = 1
            self._state_priors = nn.Parameter(torch.log(priors))
        else:
            state_prior.to(self._device)
            priors = validate_prob(state_prior, dim=0)
            self._state_priors = nn.Parameter(torch.log(priors))

        if trans_matrix is None:
            self._unnormalized_trans = nn.Parameter(torch.randn(self._n_hidden, self._n_hidden, device=self._device))
        else:
            trans_matrix.to(self._device)
            trans_matrix = validate_prob(trans_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self._unnormalized_trans = nn.Parameter(torch.log(trans_matrix))

        if emiss_matrix is None:
            self._unnormalized_emiss = nn.Parameter(
                torch.zeros(self._n_src, self._n_hidden, self._d_obs, device=self._device)
            )
        else:
            emiss_matrix.to(self._device)
            emiss_matrix = validate_prob(emiss_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self._unnormalized_emiss = nn.Parameter(torch.log(emiss_matrix))

        return self

    def _initialize_states(self,
                           embs: torch.Tensor,
                           obs: torch.Tensor,
                           temperature: Optional[int] = 1.0,
                           normalize_observation: Optional[bool] = True):
        """
        Initialize inference states. Should be called before forward inference.

        Parameters
        ----------
        embs: token embeddings
        obs: observations
        temperature: softmax temperature
        normalize_observation: whether to normalize observations

        Returns
        -------
        self
        """
        # normalize and put the probabilities into the log domain
        batch_size, max_seq_length, n_src, _ = obs.size()
        self._log_state_priors = torch.log_softmax(self._state_priors / temperature, dim=-1)
        trans = torch.softmax(self._unnormalized_trans / temperature, dim=-1)
        emiss = torch.softmax(self._unnormalized_emiss / temperature, dim=-1)

        # get neural transition and emission matrices
        # TODO: we can add layer-norm later to see what happens
        nn_trans, nn_emiss = self._nn_module(embs)

        self._log_trans = torch.log((1 - self._trans_weight) * trans + self._trans_weight * nn_trans)
        self._log_emiss = torch.log((1 - self._emiss_weight) * emiss + self._emiss_weight * nn_emiss)

        # if at least one source observes an entity at a position, set the probabilities of other sources to
        # the mean value (so that they will not affect the prediction)
        # maybe we can also set them all to 0?
        # [10/20/2020] The current version works fine. No need to change for now.
        # [10/20/2020] Pack this process into an if branch
        if normalize_observation:
            lbs = obs.argmax(dim=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(dim=-1) > 1E-6
            # the sources that do not observe any entity
            no_entity_idx = lbs <= 1E-6
            no_obs_src_idx = entity_idx.unsqueeze(-1) * no_entity_idx
            subsitute_prob = torch.zeros_like(obs[0, 0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / self._d_obs
            obs[no_obs_src_idx] = subsitute_prob

        # Calculate the emission probabilities in one time, so that we don't have to compute this repeatedly
        # log-domain subtract is regular-domain divide
        self._log_emiss_evidence = log_matmul(
            self._log_emiss, torch.log(obs).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-2)

        self._log_alpha = torch.zeros([batch_size, max_seq_length, self._n_hidden], device=self._device)
        self._log_beta = torch.zeros([batch_size, max_seq_length, self._n_hidden], device=self._device)
        # Gamma can be readily computed and need no initialization
        self._log_gamma = None
        # only values in 1:max_seq_length are valid. The first state is a dummy
        self._log_xi = torch.zeros([batch_size, max_seq_length, self._n_hidden, self._n_hidden], device=self._device)
        return self

    def _forward_step(self, t):
        # initial alpha state
        if t == 0:
            log_alpha_t = self._log_state_priors + self._log_emiss_evidence[:, t, :]
        # do the forward step
        else:
            log_alpha_t = self._log_emiss_evidence[:, t, :] + \
                          log_matmul(self._log_alpha[:, t - 1, :].unsqueeze(1), self._log_trans[:, t, :, :]).squeeze(1)

        # normalize the result
        normalized_log_alpha_t = log_alpha_t - log_alpha_t.logsumexp(dim=-1, keepdim=True)
        return normalized_log_alpha_t

    def _backward_step(self, t):
        # do the backward step
        # beta is not a distribution, so we do not need to normalize it
        log_beta_t = log_matmul(
            self._log_trans[:, t, :, :],
            (self._log_emiss_evidence[:, t, :] + self._log_beta[:, t + 1, :]).unsqueeze(-1)
        ).squeeze(-1)
        return log_beta_t

    def _forward_backward(self, seq_lengths):
        max_seq_length = seq_lengths.max().item()
        # calculate log alpha
        for t in range(0, max_seq_length):
            self._log_alpha[:, t, :] = self._forward_step(t)

        # calculate log beta
        # The last beta state beta[:, -1, :] = log1 = 0, so no need to re-assign the value
        for t in range(max_seq_length - 2, -1, -1):
            self._log_beta[:, t, :] = self._backward_step(t)
        # shift the output (since beta is calculated in backward direction,
        # we need to shift each instance in the batch according to its length)
        shift_distances = seq_lengths - max_seq_length
        self._log_beta = torch.stack(
            [torch.roll(beta, s.item(), 0) for beta, s in zip(self._log_beta, shift_distances)]
        )
        return None

    def _compute_xi(self, t):
        temp_1 = self._log_emiss_evidence[:, t, :] + self._log_beta[:, t, :]
        temp_2 = log_matmul(self._log_alpha[:, t - 1, :].unsqueeze(-1), temp_1.unsqueeze(1))
        log_xi_t = self._log_trans[:, t, :, :] + temp_2
        return log_xi_t

    def _expected_complete_log_likelihood(self, seq_lengths):
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # calculate expected sufficient statistics: gamma_t(j) = P(z_t = j|x_{1:T})
        self._log_gamma = self._log_alpha + self._log_beta
        # normalize as gamma is a distribution
        log_gamma = self._log_gamma - self._log_gamma.logsumexp(dim=-1, keepdim=True)

        # calculate expected sufficient statistics: psi_t(i, j) = P(z_{t-1}=i, z_t=j|x_{1:T})
        for t in range(1, max_seq_length):
            self._log_xi[:, t, :, :] = self._compute_xi(t)
        stabled_norm_term = logsumexp(self._log_xi[:, 1:, :, :].view(batch_size, max_seq_length - 1, -1), dim=-1) \
            .view(batch_size, max_seq_length - 1, 1, 1)
        log_xi = self._log_xi[:, 1:, :, :] - stabled_norm_term

        # calculate the expected complete data log likelihood
        log_prior = torch.sum(torch.exp(log_gamma[:, 0, :]) * self._log_state_priors, dim=-1)
        log_prior = log_prior.mean()
        # sum over j, k
        log_tran = torch.sum(torch.exp(log_xi) * self._log_trans[:, 1:, :, :], dim=[-2, -1])
        # sum over valid time steps, and then average over batch. Note this starts from t=2
        log_tran = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_tran, seq_lengths - 1)]))
        # same as above
        log_emis = torch.sum(torch.exp(log_gamma) * self._log_emiss_evidence, dim=-1)
        log_emis = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_emis, seq_lengths)]))
        log_likelihood = log_prior + log_tran + log_emis

        return log_likelihood

    def forward(self, emb, obs, seq_lengths, normalize_observation=True):
        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, n_src, n_obs = obs.size()
        assert n_obs == self._d_obs
        assert n_src == self._n_src

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        self._forward_backward(seq_lengths=seq_lengths)
        log_likelihood = self._expected_complete_log_likelihood(seq_lengths=seq_lengths)
        return log_likelihood, (self.log_trans, self.log_emiss)

    def viterbi(self, emb, obs, seq_lengths, normalize_observation=True):
        """
        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # initialize states
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        # maximum probabilities
        log_delta = torch.zeros([batch_size, max_seq_length, self._n_hidden], device=self._device)
        # most likely previous state on the most probable path to z_t = j. a[0] is undefined.
        pre_states = torch.zeros([batch_size, max_seq_length, self._n_hidden], dtype=torch.long, device=self._device)

        # the initial delta state
        log_delta[:, 0, :] = self._log_state_priors + self._log_emiss_evidence[:, 0, :]
        for t in range(1, max_seq_length):
            # udpate delta and a. The location of the emission probabilities does not matter
            max_log_prob, argmax_val = log_maxmul(
                log_delta[:, t - 1, :].unsqueeze(1),
                self._log_trans[:, t, :, :] + self._log_emiss_evidence[:, t, :].unsqueeze(1)
            )
            log_delta[:, t, :] = max_log_prob.squeeze(1)
            pre_states[:, t, :] = argmax_val.squeeze(1)

        # The terminal state
        batch_max_log_prob = list()
        batch_z_t_star = list()

        for l_delta, length in zip(log_delta, seq_lengths):
            max_log_prob, z_t_star = l_delta[length - 1, :].max(dim=-1)
            batch_max_log_prob.append(max_log_prob)
            batch_z_t_star.append(z_t_star)

        # Trace back
        batch_z_star = [[z_t_star.item()] for z_t_star in batch_z_t_star]
        for p_states, z_star, length in zip(pre_states, batch_z_star, seq_lengths):
            for t in range(length - 2, -1, -1):
                z_t = p_states[t + 1, z_star[0]].item()
                z_star.insert(0, z_t)

        # compute the smoothed marginal p(z_t = j | obs_{1:T})
        self._forward_backward(seq_lengths)
        log_marginals = self._log_alpha + self._log_beta
        norm_marginals = torch.exp(log_marginals - logsumexp(log_marginals, dim=-1, keepdim=True))
        batch_marginals = list()
        for marginal, length in zip(norm_marginals, seq_lengths):
            mgn_list = marginal[1:length].detach().cpu().numpy()
            batch_marginals.append(mgn_list)

        return batch_z_star, batch_marginals

    # def annotate(self, emb, obs, seq_lengths, label_types, normalize_observation=True):
    #     batch_label_indices, batch_probs = self.viterbi(
    #         emb, obs, seq_lengths, normalize_observation=normalize_observation
    #     )
    #     batch_labels = [[label_types[lb_index] for lb_index in label_indices]
    #                     for label_indices in batch_label_indices]
    #
    #     # For batch_spans, we are going to compare them with the true spans,
    #     # and the true spans is already shifted, so we do not need to shift predicted spans back
    #     batch_spans = list()
    #     batch_scored_spans = list()
    #     for labels, probs, indices in zip(batch_labels, batch_probs, batch_label_indices):
    #         spans = label_to_span(labels)
    #         batch_spans.append(spans)
    #
    #         ps = [p[s] for p, s in zip(probs, indices[1:])]
    #         scored_spans = dict()
    #         for k, v in spans.items():
    #             if k == (0, 1):
    #                 continue
    #             start = k[0] - 1 if k[0] > 0 else 0
    #             end = k[1] - 1
    #             score = np.mean(ps[start:end])
    #             scored_spans[(start, end)] = [(v, score)]
    #         batch_scored_spans.append(scored_spans)
    #
    #     return batch_spans, (batch_scored_spans, batch_probs)
