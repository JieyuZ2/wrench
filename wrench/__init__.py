import torch

from .version import VERSION as __version__

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
