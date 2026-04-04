from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn


class BaseArchitecture(nn.Module, ABC):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Run a forward pass."""
