from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from meisenmeister.architectures import get_architecture_class
from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.base_trainer import BaseTrainer
from meisenmeister.training.splits import get_fold_sample_ids


class mmTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_id: str,
        fold: int,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        architecture_name: str = "ResNet3D18",
        num_epochs: int = 100,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        initial_lr: float = 3e-4,
        weight_decay: float = 3e-5,
    ) -> None:
        super().__init__(dataset_id, fold, dataset_dir, preprocessed_dataset_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.architecture_name = architecture_name
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_sample_ids = get_fold_sample_ids(preprocessed_dataset_dir, fold)
        self._train_dataset = None
        self._val_dataset = None
        self._architecture = None
        self._loss = None
        self._optimizer = None
        self._scheduler = None

    def fit(self) -> None:
        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        architecture = self.get_architecture()
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler()

        print(f"Trainer: {self.__class__.__name__}")
        print(f"Dataset id: {self.dataset_id}")
        print(f"Fold: {self.fold}")
        print(f"Dataset: {self.dataset_dir.name}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Architecture: {architecture.__class__.__name__}")
        print(f"Loss: {loss_fn.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Scheduler: {scheduler.__class__.__name__}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Initial LR: {self.initial_lr}")
        print(f"Weight decay: {self.weight_decay}")

        for epoch_idx in range(1, self.num_epochs + 1):
            train_metrics = []
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                train_metrics.append(self.train_step(batch, batch_idx))

            val_metrics = []
            for batch_idx, batch in enumerate(val_dataloader, start=1):
                val_metrics.append(self.validate_step(batch, batch_idx))

            train_loss, train_accuracy = self._aggregate_epoch_metrics(train_metrics)
            val_loss, val_accuracy = self._aggregate_epoch_metrics(val_metrics)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch_idx}/{self.num_epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- train_acc: {train_accuracy:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_acc: {val_accuracy:.4f} "
                f"- lr: {current_lr:.6f}"
            )
            scheduler.step()

        print("DONE")

    def get_architecture(self):
        if self._architecture is None:
            architecture_class = get_architecture_class(self.architecture_name)
            in_channels = int(self.get_train_dataset()[0]["image"].shape[0])
            labels = [
                sample["label"]
                for sample in (
                    self.get_train_dataset().samples + self.get_val_dataset().samples
                )
            ]
            num_classes = max(2, max(labels) + 1)
            self._architecture = architecture_class(
                in_channels=in_channels,
                num_classes=num_classes,
            ).to(self.device)
        return self._architecture

    def get_loss(self):
        if self._loss is None:
            self._loss = nn.CrossEntropyLoss()
        return self._loss

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = MeisenmeisterROIDataset(
                self.preprocessed_dataset_dir,
                allowed_sample_ids=set(self.split_sample_ids["train"]),
            )
        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = MeisenmeisterROIDataset(
                self.preprocessed_dataset_dir,
                allowed_sample_ids=set(self.split_sample_ids["val"]),
            )
        return self._val_dataset

    def get_train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_val_dataloader(self):
        return DataLoader(
            self.get_val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_optimizer(self):
        if self._optimizer is None:
            architecture = self.get_architecture()
            parameters = list(architecture.parameters())
            if not parameters:
                raise ValueError(
                    f"Architecture '{architecture.__class__.__name__}' has no trainable parameters"
                )
            self._optimizer = torch.optim.SGD(
                parameters,
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        return self._optimizer

    def get_scheduler(self):
        if self._scheduler is None:
            optimizer = self.get_optimizer()
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
            )
        return self._scheduler

    def train_step(self, batch, batch_idx: int):
        del batch_idx
        architecture = self.get_architecture()
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()
        images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(self.device, dtype=torch.long, non_blocking=True)

        architecture.train()
        optimizer.zero_grad()

        logits = architecture(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        return {
            "loss": float(loss.detach().item()),
            "num_samples": int(labels.shape[0]),
            "num_correct": int((predictions == labels).sum().detach().item()),
        }

    def validate_step(self, batch, batch_idx: int):
        del batch_idx
        architecture = self.get_architecture()
        loss_fn = self.get_loss()
        images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(self.device, dtype=torch.long, non_blocking=True)

        architecture.eval()
        with torch.no_grad():
            logits = architecture(images)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

        return {
            "loss": float(loss.detach().item()),
            "num_samples": int(labels.shape[0]),
            "num_correct": int((predictions == labels).sum().item()),
        }

    def _aggregate_epoch_metrics(self, metrics):
        total_samples = sum(metric["num_samples"] for metric in metrics)
        if total_samples == 0:
            raise ValueError("Cannot aggregate metrics for an empty epoch")

        total_loss = sum(metric["loss"] * metric["num_samples"] for metric in metrics)
        total_correct = sum(metric["num_correct"] for metric in metrics)
        return total_loss / total_samples, total_correct / total_samples
