""""
https://github.com/BatsResearch/labelmodels
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


class LabelModel(nn.Module):
    """Parent class for all generative label models.
    Concrete subclasses should implement at least forward(),
    estimate_label_model(), and get_label_distribution().
    """

    def forward(self, *args):
        """Computes the marginal log-likelihood of a batch of observed
        function outputs provided as input.
        :param args: batch of observed function outputs and related metadata
        :return: 1-d tensor of log-likelihoods, one for each input example
        """
        raise NotImplementedError

    def estimate_label_model(self, *args, config=None):
        """Learns the parameters of the label model from observed
        function outputs.
        Subclasses that implement this method should call _do_estimate_label_model()
        if possible, to provide consistent behavior.
        :param args: observed function outputs and related metadata
        :param config: an instance of LearningConfig. If none, will initialize
                       with default LearningConfig constructor
        """
        raise NotImplementedError

    def get_label_distribution(self, *args):
        """Returns the estimated posterior distribution over true labels given
        observed function outputs.
        :param args: observed function outputs and related metadata
        :return: distribution over true labels. Structure depends on model type
        """
        raise NotImplementedError

    def get_most_probable_labels(self, *args):
        """Returns the most probable true labels given observed function outputs.
        :param args: observed function outputs and related metadata
        :return: 1-d Numpy array of most probable labels
        """
        raise NotImplementedError

    def _do_estimate_label_model(self, batches, config):
        """Internal method for optimizing model parameters.
        :param batches: sequence of inputs to forward(). The sequence must
                        contain tuples, even if forward() takes one
                        argument (besides self)
        :param config: an instance of LearningConfig
        """

        # Sets up optimization hyperparameters
        optimizer = torch.optim.SGD(
            self.parameters(), lr=config.step_size, momentum=config.momentum,
            weight_decay=0)
        if config.step_schedule is not None and config.step_size_mult is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.step_schedule, gamma=config.step_size_mult)
        else:
            scheduler = None

        # Iterates over epochs
        for epoch in range(config.epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, config.epochs))
            if scheduler is not None:
                scheduler.step()

            # Sets model to training mode
            self.train()
            running_loss = 0.0

            # Iterates over training data
            for i_batch, inputs in enumerate(batches):
                optimizer.zero_grad()
                log_likelihood = self(*inputs)
                loss = -1 * torch.mean(log_likelihood)
                loss += self._get_regularization_loss() * config.reg_weight
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(batches)
            logging.info('Train Loss: %.6f', epoch_loss)

    def _get_regularization_loss(self):
        """Gets the value of the regularization loss for the current values of
        the model's parameters
        :return: regularization loss
        """
        return 0.0


class ClassConditionalLabelModel(LabelModel):
    """
    Abstract parent class for generative label models that assume labeling
    functions are conditionally independent given the true label, and that each
    labeling function is characterized by the following parameters:
        * a propensity, which is the probability that it does not abstain
        * class-conditional accuracies, each of which is the probability that
          the labeling function's output is correct given that the true label
          has a certain value. It is assumed that when a labeling function makes
          a mistake, the label it outputs is chosen uniformly at random
    """

    def __init__(self, num_classes, num_lfs, init_acc, acc_prior):
        """Constructor.
        Initializes label source accuracies argument and propensities uniformly.
        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
                          function accuracies toward their initial values
        """
        super().__init__()

        # Converts init_acc to log scale
        init_acc = -1 * np.log(1.0 / init_acc - 1) / 2

        init_param = torch.tensor(
            [[init_acc] * num_classes for _ in range(num_lfs)])
        self.accuracy = nn.Parameter(init_param)
        self.propensity = nn.Parameter(torch.zeros([num_lfs]))

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_acc = init_acc
        self.acc_prior = acc_prior

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        acc = self.accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        prop = self.propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

    def _get_labeling_function_likelihoods(self, votes):
        """
        Computes conditional log-likelihood of labeling function votes given
        class as an m x k matrix.
        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.
        :param votes: m x n matrix in {0, ..., k}, where m is the sum of the
                      lengths of the sequences in the batch, n is the number of
                      labeling functions and k is the number of classes
        :return: matrix of dimension m x k, where element is the conditional
                 log-likelihood of votes given class
        """
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        # Initializes conditional log-likelihood of votes as an m x k matrix
        cll = torch.zeros(votes.shape[0], self.num_classes)

        # Initializes normalizing constants
        z_prop = self.propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for propensities from cll
        # (since it applies to all outcomes)
        cll -= torch.sum(z_prop)

        # Loops over votes and classes to compute conditional log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            for k in range(self.num_classes):
                if v == (k + 1):
                    logp = self.propensity[j] + self.accuracy[j, k] - z_acc[j, k]
                    cll[i, k] += logp
                elif v != 0:
                    logp = self.propensity[j] - self.accuracy[j, k] - z_acc[j, k]
                    logp -= torch.log(torch.tensor(self.num_classes - 1.0))
                    cll[i, k] += logp

        return cll

    def _get_regularization_loss(self):
        """Computes the regularization loss of the model:
        acc_prior * \|accuracy - init_acc\|_2
        :return: value of regularization loss
        """
        return self.acc_prior * torch.norm(self.accuracy - self.init_acc)


class LearningConfig(object):
    """Container for hyperparameters used by label models during learning"""

    def __init__(self):
        """Initializes all hyperparameters to default values"""
        self.epochs = 5
        self.batch_size = 64
        self.step_size = 0.01
        self.reg_weight = 1.0
        self.step_schedule = None
        self.step_size_mult = None
        self.momentum = 0.9
        self.random_seed = 0


def init_random(seed):
    """Initializes PyTorch and NumPy random seeds.
    Also sets the CuDNN back end to deterministic.
    :param seed: integer to use as random seed
    """
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info("Random seed: %d", seed)


class NaiveBayes(ClassConditionalLabelModel):
    """A generative label model that assumes that all labeling functions are
    conditionally independent given the true class label, i.e., the naive Bayes
    assumption.
    Proposed in: A. P. Dawid and A. M. Skene. Maximum likelihood
    estimation of observer error-rates using the EM algorithm.
    Journal of the Royal Statistical Society C, 28(1):20–28, 1979.
    Proposed for labeling functions in: A. Ratner, C. De Sa, S. Wu, D. Selsam,
    and C. Ré. Data programming: Creating large training sets, quickly. In
    Neural Information Processing Systems, 2016.
    """

    def __init__(self, num_classes, num_lfs, init_acc=.9, acc_prior=0.025,
                 balance_prior=0.025, learn_class_balance=True):
        """Constructor.
        Initializes labeling function accuracies using optional argument and all
        other model parameters uniformly.
        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
                          function accuracies toward their initial values
        :param learn_class_balance: whether to estimate the distribution over
                                    target classes (True) or assume to be
                                    uniform (False)
        """
        super().__init__(num_classes, num_lfs, init_acc, acc_prior)
        self.class_balance = nn.Parameter(
            torch.zeros([num_classes]), requires_grad=learn_class_balance)

        self.balance_prior = balance_prior

    def forward(self, votes):
        """Computes log likelihood of labeling function outputs for each
        example in the batch.
        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.
        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        class_ll = self._get_norm_class_balance()
        conditional_ll = self._get_labeling_function_likelihoods(votes)
        joint_ll = conditional_ll + class_ll
        return torch.logsumexp(joint_ll, dim=1)

    def estimate_label_model(self, votes, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.
        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        batches = self._create_minibatches(
            votes, config.batch_size, shuffle_rows=True)
        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes):
        """Returns the posterior distribution over true labels given labeling
        function outputs according to the model
        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: m x k matrix, where each row is the posterior distribution over
                 the true class label for the corresponding example
        """
        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        labels = np.ndarray((votes.shape[0], self.num_classes))
        batches = self._create_minibatches(votes, 4096, shuffle_rows=False)

        offset = 0
        for votes, in batches:
            class_balance = self._get_norm_class_balance()
            lf_likelihood = self._get_labeling_function_likelihoods(votes)
            jll = class_balance + lf_likelihood
            for i in range(votes.shape[0]):
                p = torch.exp(jll[i, :] - torch.max(jll[i, :]))
                p = p / p.sum()
                for j in range(self.num_classes):
                    labels[offset + i, j] = p[j]
            offset += votes.shape[0]

        return labels

    def get_most_probable_labels(self, votes):
        """Returns the most probable true labels given observed function outputs.
        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d Numpy array of most probable labels
        """
        return np.argmax(self.get_label_distribution(votes), axis=1) + 1

    def get_class_balance(self):
        """Returns the model's estimated class balance
        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp(self._get_norm_class_balance().detach().numpy())

    def _create_minibatches(self, votes, batch_size, shuffle_rows=False):
        if shuffle_rows:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]

        # Creates minibatches
        batches = [(sparse.coo_matrix(
            votes[i * batch_size: (i + 1) * batch_size, :],
            copy=True),)
            for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        return batches

    def _get_regularization_loss(self):
        neg_entropy = 0.0
        norm_class_balance = self._get_norm_class_balance()
        exp_class_balance = torch.exp(norm_class_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_class_balance[k] * exp_class_balance[k]
        entropy_prior = self.balance_prior * neg_entropy

        return super()._get_regularization_loss() + entropy_prior

    def _get_norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)
