from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    def __init__(
        self,
        dataset_id: str,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_dir = dataset_dir
        self.preprocessed_dataset_dir = preprocessed_dataset_dir

    @abstractmethod
    def fit(self) -> None:
        """Run the trainer's fit loop."""

    @abstractmethod
    def get_architecture(self):
        """Return the model architecture used by this trainer."""

    @abstractmethod
    def get_loss(self):
        """Return the loss function used by this trainer."""

    @abstractmethod
    def get_dataset(self):
        """Return the dataset instance used by this trainer."""

    @abstractmethod
    def get_dataloader(self):
        """Return the dataloader used by this trainer."""

    @abstractmethod
    def get_optimizer(self):
        """Return the optimizer used by this trainer."""

    @abstractmethod
    def get_scheduler(self):
        """Return the scheduler used by this trainer."""

    @abstractmethod
    def train_step(self, batch, batch_idx: int):
        """Run one training step."""

    @abstractmethod
    def validate_step(self, batch, batch_idx: int):
        """Run one validation step."""
