import logging
from typing import Optional, List

from dataclasses import dataclass, field
from transformers.file_utils import cached_property

logger = logging.getLogger(__name__)


@dataclass
class CHMMArguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """
    train_file: Optional[str] = field(
        default='', metadata={'help': 'training data name'}
    )
    valid_file: Optional[str] = field(
        default='', metadata={'help': 'development data name'}
    )
    test_file: Optional[str] = field(
        default='', metadata={'help': 'test data name'}
    )
    output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "The output folder where the model predictions and checkpoints will be written."},
    )
    trans_nn_weight: Optional[float] = field(
        default=1, metadata={'help': 'the weight of neural part in the transition matrix'}
    )
    emiss_nn_weight: Optional[float] = field(
        default=1, metadata={'help': 'the weight of neural part in the emission matrix'}
    )
    num_train_epochs: Optional[int] = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    num_nn_pretrain_epochs: Optional[int] = field(
        default=5, metadata={'help': 'number of denoising model pre-training epochs'}
    )
    num_valid_tolerance: Optional[int] = field(
        default=5, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    hmm_lr: Optional[float] = field(
        default=0.01, metadata={'help': 'learning rate of the original hidden markov model transition and emission'}
    )
    nn_lr: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate of the neural networks in CHMM'}
    )
    batch_size: Optional[int] = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    obs_normalization: Optional[bool] = field(
        default=True, metadata={'help': 'whether normalize observations'}
    )
    log_dir: Optional[str] = field(
        default='',
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )


@dataclass
class CHMMConfig(CHMMArguments):
    """
    Conditional HMM training configuration
    """
    sources: Optional[str] = None
    entity_types: Optional[List[str]] = None
    bio_label_types: Optional[List[str]] = None
    src_priors: Optional[dict] = None
    d_emb: Optional[int] = None

    @cached_property
    def _get_hidden_dim(self) -> "int":
        """
        Returns the HMM hidden dimension, AKA, the number of bio labels
        """
        return len(self.bio_label_types)

    @property
    def d_hidden(self) -> "int":
        return self._get_hidden_dim

    @property
    def d_obs(self) -> "int":
        return self._get_hidden_dim

    @property
    def n_src(self) -> "int":
        """
        Returns the number of sources
        """
        return len(self.sources)

    def from_args(self, args: CHMMArguments) -> "CHMMConfig":
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: CHMMConfig)
        """
        arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                        and not attr.startswith("__") and not attr.startswith("_")}
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self
