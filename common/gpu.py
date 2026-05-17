"""GPU availability helpers shared across training scripts."""
from __future__ import annotations

from common.logging_config import get_logger

log = get_logger(__name__)


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
            log.info("GPU detected: %s (%d MB VRAM) – using CUDA", name, vram)
            return "cuda"
    except ImportError:
        pass
    log.info("No CUDA GPU detected – using CPU")
    return "cpu"


def gpu_batch_size(cpu_size: int = 32, gpu_size: int = 256) -> int:
    """Return a batch size appropriate for the available hardware."""
    try:
        import torch
        return gpu_size if torch.cuda.is_available() else cpu_size
    except ImportError:
        return cpu_size
