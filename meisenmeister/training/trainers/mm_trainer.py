from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.base_trainer import BaseTrainer


class mmTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_id: str,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        num_epochs: int = 1,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> None:
        super().__init__(dataset_id, dataset_dir, preprocessed_dataset_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dataset = None
        self._architecture = None
        self._loss = None
        self._optimizer = None
        self._scheduler = None

    def fit(self) -> None:
        dataset = self.get_dataset()
        dataloader = self.get_dataloader()
        architecture = self.get_architecture()
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler()

        print(f"Trainer: {self.__class__.__name__}")
        print(f"Dataset id: {self.dataset_id}")
        print(f"Dataset: {self.dataset_dir.name}")
        print(f"Samples: {len(dataset)}")
        print(f"Architecture: {architecture.__class__.__name__}")
        print(f"Loss: {loss_fn.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Scheduler: {scheduler.__class__.__name__}")
        print(f"Epochs: {self.num_epochs}")

        for epoch_idx in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch_idx}/{self.num_epochs}")
            for batch_idx, batch in enumerate(dataloader, start=1):
                self.train_step(batch, batch_idx)
                self.validate_step(batch, batch_idx)

        print("DONE")

    def get_architecture(self):
        if self._architecture is None:
            self._architecture = nn.Identity().to(self.device)
        return self._architecture

    def get_loss(self):
        if self._loss is None:
            self._loss = nn.CrossEntropyLoss()
        return self._loss

    def get_dataset(self):
        if self._dataset is None:
            self._dataset = MeisenmeisterROIDataset(self.preprocessed_dataset_dir)
        return self._dataset

    def get_dataloader(self):
        return DataLoader(
            self.get_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_optimizer(self):
        if self._optimizer is None:
            architecture = self.get_architecture()
            parameter = next(architecture.parameters(), None)
            if parameter is None:
                parameter = nn.Parameter(
                    torch.zeros(1, requires_grad=True, device=self.device)
                )
            self._optimizer = torch.optim.SGD([parameter], lr=0.01)
        return self._optimizer

    def get_scheduler(self):
        if self._scheduler is None:
            optimizer = self.get_optimizer()
            self._scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=1,
            )
        return self._scheduler

    def train_step(self, batch, batch_idx: int):
        print(
            f"Train batch {batch_idx}: image_shape={tuple(batch['image'].shape)}, "
            f"sample_ids={list(batch['sample_id'])}"
        )
        return {
            "batch_idx": batch_idx,
            "num_samples": len(batch["sample_id"]),
        }

    def validate_step(self, batch, batch_idx: int):
        print(
            f"Validate batch {batch_idx}: labels_shape={tuple(batch['label'].shape)}, "
            f"sample_ids={list(batch['sample_id'])}"
        )
        return {
            "batch_idx": batch_idx,
            "num_samples": len(batch["sample_id"]),
        }
