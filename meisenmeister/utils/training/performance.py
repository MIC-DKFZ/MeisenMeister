from __future__ import annotations

from contextlib import nullcontext

import torch


def configure_training_performance(device: torch.device) -> None:
    if device.type == "cuda" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def is_amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def create_grad_scaler(device: torch.device):
    if not is_amp_enabled(device):
        return None
    return torch.amp.GradScaler("cuda")


def autocast_context(device: torch.device):
    if not is_amp_enabled(device):
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)
