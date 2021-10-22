from contextlib import ContextDecorator
from typing import Any

import torch

from .version import VERSION as __version__

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

_TORCH_AMP = False
_TORCH_NUMBER_WORKER = 0
_TORCH_PIN_MEMORY = False


def set_amp_flag(value: bool):
    global _TORCH_AMP
    _TORCH_AMP = value


def get_amp_flag():
    global _TORCH_AMP
    return _TORCH_AMP


def set_num_workers(value: int):
    global _TORCH_NUMBER_WORKER
    _TORCH_NUMBER_WORKER = value


def get_num_workers():
    global _TORCH_NUMBER_WORKER
    return _TORCH_NUMBER_WORKER


def set_pin_memory(value: bool):
    global _TORCH_PIN_MEMORY
    _TORCH_PIN_MEMORY = value


def get_pin_memory():
    global _TORCH_PIN_MEMORY
    return _TORCH_PIN_MEMORY


class efficient_training(ContextDecorator):
    def __init__(self, amp: bool = False, num_workers: int = 0, pin_memory: bool = False):
        self.prev_amp = get_amp_flag()
        self.prev_num_workers = get_num_workers()
        self.prev_pin_memory = get_pin_memory()
        self.amp = amp
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __enter__(self):
        set_amp_flag(torch.cuda.is_available() & self.amp)
        set_num_workers(self.num_workers)
        set_pin_memory(self.pin_memory)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_amp_flag(self.prev_amp)
        set_num_workers(self.prev_num_workers)
        set_pin_memory(self.prev_pin_memory)
