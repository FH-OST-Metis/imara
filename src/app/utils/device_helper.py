import torch
import logging

_log = logging.getLogger(__name__)


def get_device(device_preference: str = "cpu") -> str:
    """
    Finds and returns the available GPU device or CPU.
    In case of multiple GPUs, returns the second one (cuda:1).

    Returns:
        str: Device string like "cuda:0", "cuda:1", "mps", or "cpu"
    """

    device = "cpu"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        _log.info(f"Number of available GPUs: {num_gpus}")
        if num_gpus > 1:
            device = "cuda:1"
            _log.info(f"Multiple GPUs detected. Using: {device}")
        else:
            device = "cuda:0"
            _log.info(f"Single GPU detected. Using: {device}")
        return device

    if torch.mps.is_available():
        device = "mps"
        _log.info(f"Apple Silicon detected. Using: {device}")
        return device

    _log.info("No GPU detected. Using CPU.")
    return device
