from __future__ import annotations

import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from meisenmeister.architectures import get_architecture_class
from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.base_trainer import BaseTrainer
from meisenmeister.training.splits import get_fold_sample_ids

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class mmTrainer(BaseTrainer):
    EMA_ALPHA = 0.1
    BEST_SCORE_TOLERANCE = 1e-12

    def __init__(
        self,
        dataset_id: str,
        fold: int,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str = "ResNet3D18",
        num_epochs: int = 100,
        batch_size: int = 2,
        num_workers: int = 0,
        shuffle: bool = True,
        initial_lr: float = 3e-4,
        weight_decay: float = 3e-5,
        continue_training: bool = False,
    ) -> None:
        super().__init__(dataset_id, fold, dataset_dir, preprocessed_dataset_dir)
        self.results_dir = results_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.architecture_name = architecture_name
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.continue_training = continue_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_sample_ids = get_fold_sample_ids(preprocessed_dataset_dir, fold)
        self.experiment_dir = (
            self.results_dir
            / self.dataset_dir.name
            / f"{self.__class__.__name__}_{self.architecture_name}"
        )
        self.fold_dir = self.experiment_dir / f"fold_{self.fold}"
        self.log_path = self.fold_dir / "train.log"
        self.last_checkpoint_path = self.fold_dir / "model_last.pt"
        self.best_checkpoint_path = self.fold_dir / "model_best.pt"
        self.plot_path = self.fold_dir / "training_curves.png"
        self._train_dataset = None
        self._val_dataset = None
        self._architecture = None
        self._loss = None
        self._optimizer = None
        self._scheduler = None
        self._history = self._create_empty_history()
        self._best_state = {
            "epoch": None,
            "ema_val_balanced_accuracy": None,
            "val_loss": None,
        }
        self._resume_state = None

    def fit(self) -> None:
        self._prepare_output_dir()
        self._maybe_restore_training_state()

        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        architecture = self.get_architecture()
        loss_fn = self.get_loss()
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler()

        start_epoch = 1
        if self._resume_state is not None:
            self._restore_from_checkpoint_payload(self._resume_state)
            optimizer = self.get_optimizer()
            scheduler = self.get_scheduler()
            start_epoch = int(self._resume_state["last_completed_epoch"]) + 1
            self._log(f"Resuming training from epoch {start_epoch} in {self.fold_dir}")

        self._log(f"Trainer: {self.__class__.__name__}")
        self._log(f"Dataset id: {self.dataset_id}")
        self._log(f"Fold: {self.fold}")
        self._log(f"Dataset: {self.dataset_dir.name}")
        self._log(f"Results dir: {self.fold_dir}")
        self._log(f"Train samples: {len(train_dataset)}")
        self._log(f"Val samples: {len(val_dataset)}")
        self._log(f"Architecture: {architecture.__class__.__name__}")
        self._log(f"Loss: {loss_fn.__class__.__name__}")
        self._log(f"Optimizer: {optimizer.__class__.__name__}")
        self._log(f"Scheduler: {scheduler.__class__.__name__}")
        self._log(f"Epochs: {self.num_epochs}")
        self._log(f"Initial LR: {self.initial_lr}")
        self._log(f"Weight decay: {self.weight_decay}")

        if start_epoch > self.num_epochs:
            self._log(
                f"Training already completed through epoch {start_epoch - 1}; nothing to do."
            )
            self._log("DONE")
            return

        for epoch_idx in range(start_epoch, self.num_epochs + 1):
            epoch_start_time = time.perf_counter()

            train_metrics = []
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                train_metrics.append(self.train_step(batch, batch_idx))

            val_metrics = []
            for batch_idx, batch in enumerate(val_dataloader, start=1):
                val_metrics.append(self.validate_step(batch, batch_idx))

            train_loss, train_accuracy = self._aggregate_epoch_metrics(train_metrics)
            val_loss, val_accuracy = self._aggregate_epoch_metrics(val_metrics)
            val_balanced_accuracy, val_macro_auc, macro_auc_defined = (
                self._aggregate_validation_classification_metrics(val_metrics)
            )
            ema_val_balanced_accuracy = self._compute_ema(
                self._history["ema_val_balanced_accuracy"],
                val_balanced_accuracy,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_time_sec = float(time.perf_counter() - epoch_start_time)

            self._append_history(
                epoch=epoch_idx,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_balanced_accuracy=val_balanced_accuracy,
                val_macro_auc=val_macro_auc,
                ema_val_balanced_accuracy=ema_val_balanced_accuracy,
                lr=current_lr,
                epoch_time_sec=epoch_time_sec,
            )

            if not macro_auc_defined:
                self._log(
                    f"Epoch {epoch_idx}/{self.num_epochs} - val_macro_auc undefined for this epoch because validation targets do not span enough classes."
                )

            self._log(
                f"Epoch {epoch_idx}/{self.num_epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- train_acc: {train_accuracy:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_acc: {val_accuracy:.4f} "
                f"- val_bal_acc: {val_balanced_accuracy:.4f} "
                f"- val_macro_auc: {self._format_metric(val_macro_auc)} "
                f"- ema_val_bal_acc: {ema_val_balanced_accuracy:.4f} "
                f"- epoch_time_sec: {epoch_time_sec:.4f} "
                f"- lr: {current_lr:.6f}"
            )

            if self._should_update_best(ema_val_balanced_accuracy, val_loss):
                self._best_state = {
                    "epoch": epoch_idx,
                    "ema_val_balanced_accuracy": ema_val_balanced_accuracy,
                    "val_loss": val_loss,
                }
                self._save_checkpoint(self.best_checkpoint_path, epoch_idx)
                self._log(
                    f"Saved new best model at epoch {epoch_idx} with ema_val_bal_acc={ema_val_balanced_accuracy:.4f} and val_loss={val_loss:.4f}"
                )

            scheduler.step()
            self._save_checkpoint(self.last_checkpoint_path, epoch_idx)
            self._save_training_curves()

        self._log("DONE")

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
            probabilities = torch.softmax(logits, dim=1)

        return {
            "loss": float(loss.detach().item()),
            "num_samples": int(labels.shape[0]),
            "num_correct": int((predictions == labels).sum().item()),
            "labels": labels.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
        }

    def _prepare_output_dir(self) -> None:
        fold_dir_exists = self.fold_dir.exists()
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        if not self.continue_training:
            self.log_path.write_text("", encoding="utf-8")
        if fold_dir_exists and not self.continue_training:
            self._log(
                f"WARNING: YOU ARE OVERWRITING EXISTING TRAINING OUTPUT IN {self.fold_dir}"
            )

    def _maybe_restore_training_state(self) -> None:
        if not self.continue_training:
            return
        if (
            not self.last_checkpoint_path.is_file()
            and not self.best_checkpoint_path.is_file()
        ):
            raise FileNotFoundError(
                "Cannot continue training because neither model_last.pt nor model_best.pt exists "
                f"in {self.fold_dir}"
            )
        self._resume_state = self._load_resume_checkpoint()
        self._validate_resume_state(self._resume_state)

    def _load_resume_checkpoint(self) -> dict:
        candidate_paths = [self.last_checkpoint_path, self.best_checkpoint_path]
        errors: list[str] = []
        for index, candidate_path in enumerate(candidate_paths):
            if not candidate_path.is_file():
                continue
            try:
                checkpoint = torch.load(
                    candidate_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as error:
                errors.append(f"{candidate_path}: {error}")
                continue

            if index == 1:
                self._log(
                    f"WARNING: model_last.pt is unavailable or unreadable; resuming from fallback checkpoint {candidate_path}"
                )
            return checkpoint

        error_message = "\n".join(errors) if errors else "<no readable checkpoints>"
        raise RuntimeError(
            "Cannot continue training because no readable checkpoint was found.\n"
            f"{error_message}"
        )

    def _validate_resume_state(self, checkpoint: dict) -> None:
        config = checkpoint["trainer_config"]
        expected = {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_dir.name,
            "fold": self.fold,
            "trainer_name": self.__class__.__name__,
            "architecture_name": self.architecture_name,
        }
        for key, expected_value in expected.items():
            actual_value = config.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Resume checkpoint mismatch for '{key}': expected {expected_value!r}, got {actual_value!r}"
                )

    def _restore_from_checkpoint_payload(self, checkpoint: dict) -> None:
        self.get_architecture().load_state_dict(checkpoint["model_state_dict"])
        self.get_optimizer().load_state_dict(checkpoint["optimizer_state_dict"])
        self.get_scheduler().load_state_dict(checkpoint["scheduler_state_dict"])
        self._history = checkpoint["history"]
        self._best_state = checkpoint["best_state"]
        self._restore_rng_state(checkpoint.get("rng_state"))

    def _restore_rng_state(self, rng_state: dict | None) -> None:
        if rng_state is None:
            return
        random_state = rng_state.get("python")
        if random_state is not None:
            random.setstate(random_state)
        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        torch_state = rng_state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _aggregate_epoch_metrics(self, metrics):
        total_samples = sum(metric["num_samples"] for metric in metrics)
        if total_samples == 0:
            raise ValueError("Cannot aggregate metrics for an empty epoch")

        total_loss = sum(metric["loss"] * metric["num_samples"] for metric in metrics)
        total_correct = sum(metric["num_correct"] for metric in metrics)
        return total_loss / total_samples, total_correct / total_samples

    def _aggregate_validation_classification_metrics(
        self,
        metrics,
    ) -> tuple[float, float, bool]:
        labels = torch.cat([metric["labels"] for metric in metrics], dim=0).numpy()
        predictions = torch.cat(
            [metric["predictions"] for metric in metrics],
            dim=0,
        ).numpy()
        probabilities = torch.cat(
            [metric["probabilities"] for metric in metrics],
            dim=0,
        ).numpy()
        balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
        macro_auc = math.nan
        macro_auc_defined = True
        try:
            macro_auc = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                    labels=list(range(probabilities.shape[1])),
                )
            )
        except ValueError:
            macro_auc_defined = False
        if math.isnan(macro_auc):
            macro_auc_defined = False
        return balanced_accuracy, macro_auc, macro_auc_defined

    def _compute_ema(self, history: list[float], current_value: float) -> float:
        if not history:
            return current_value
        previous_value = history[-1]
        return self.EMA_ALPHA * current_value + (1 - self.EMA_ALPHA) * previous_value

    def _append_history(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        val_balanced_accuracy: float,
        val_macro_auc: float,
        ema_val_balanced_accuracy: float,
        lr: float,
        epoch_time_sec: float,
    ) -> None:
        self._history["epoch"].append(int(epoch))
        self._history["train_loss"].append(float(train_loss))
        self._history["train_accuracy"].append(float(train_accuracy))
        self._history["val_loss"].append(float(val_loss))
        self._history["val_accuracy"].append(float(val_accuracy))
        self._history["val_balanced_accuracy"].append(float(val_balanced_accuracy))
        self._history["val_macro_auc"].append(float(val_macro_auc))
        self._history["ema_val_balanced_accuracy"].append(
            float(ema_val_balanced_accuracy)
        )
        self._history["lr"].append(float(lr))
        self._history["epoch_time_sec"].append(float(epoch_time_sec))

    def _should_update_best(
        self, ema_val_balanced_accuracy: float, val_loss: float
    ) -> bool:
        best_ema = self._best_state["ema_val_balanced_accuracy"]
        best_val_loss = self._best_state["val_loss"]
        if best_ema is None:
            return True
        if ema_val_balanced_accuracy > best_ema + self.BEST_SCORE_TOLERANCE:
            return True
        if abs(ema_val_balanced_accuracy - best_ema) <= self.BEST_SCORE_TOLERANCE:
            return best_val_loss is None or val_loss < best_val_loss
        return False

    def _save_checkpoint(self, path: Path, epoch_idx: int) -> None:
        checkpoint = {
            "trainer_config": self._get_trainer_config(),
            "last_completed_epoch": int(epoch_idx),
            "history": self._history,
            "best_state": self._best_state,
            "model_state_dict": self.get_architecture().state_dict(),
            "optimizer_state_dict": self.get_optimizer().state_dict(),
            "scheduler_state_dict": self.get_scheduler().state_dict(),
            "rng_state": self._capture_rng_state(),
        }
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

    def _capture_rng_state(self) -> dict:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        return rng_state

    def _get_trainer_config(self) -> dict:
        architecture = self.get_architecture()
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_dir.name,
            "fold": self.fold,
            "trainer_name": self.__class__.__name__,
            "architecture_name": self.architecture_name,
            "results_dir": str(self.results_dir),
            "experiment_dir": str(self.experiment_dir),
            "fold_dir": str(self.fold_dir),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay,
            "device": str(self.device),
            "num_classes": getattr(architecture, "num_classes", None),
            "in_channels": getattr(architecture, "in_channels", None),
        }

    def _save_training_curves(self) -> None:
        epochs = self._history["epoch"]
        figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axes[0].plot(epochs, self._history["train_loss"], label="train_loss")
        axes[0].plot(epochs, self._history["val_loss"], label="val_loss")
        axes[0].plot(
            epochs,
            self._history["val_balanced_accuracy"],
            label="val_balanced_accuracy",
        )
        axes[0].plot(epochs, self._history["val_macro_auc"], label="val_macro_auc")
        axes[0].plot(
            epochs,
            self._history["ema_val_balanced_accuracy"],
            label="ema_val_balanced_accuracy",
        )
        axes[0].set_ylabel("metrics")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self._history["epoch_time_sec"], label="epoch_time_sec")
        axes[1].set_ylabel("seconds")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self._history["lr"], label="lr")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("lr")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        figure.tight_layout()
        figure.savefig(self.plot_path)
        plt.close(figure)

    def _log(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(f"{message}\n")

    def _format_metric(self, value: float) -> str:
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"

    def _create_empty_history(self) -> dict[str, list[float]]:
        return {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_balanced_accuracy": [],
            "val_macro_auc": [],
            "ema_val_balanced_accuracy": [],
            "lr": [],
            "epoch_time_sec": [],
        }
