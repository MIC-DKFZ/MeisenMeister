from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from meisenmeister.training.trainers.mm_trainer import mmTrainer


class _TinyROIDataset(Dataset):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.samples = [
            {"sample_id": f"case_{index:03d}_left", "label": label}
            for index, label in enumerate(labels)
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        label = self.labels[index]
        return {
            "image": torch.full((1, 2, 2, 2), float(index + 1), dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": f"case_{index:03d}_left",
            "case_id": f"case_{index:03d}",
            "roi_name": "left",
        }


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _CountingScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class MMTrainerTests(unittest.TestCase):
    def _make_trainer(self) -> mmTrainer:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
                return_value={
                    "train": ["case_000_left", "case_001_left"],
                    "val": ["case_002_left", "case_003_left"],
                },
            ):
                trainer = mmTrainer(
                    dataset_id="001",
                    fold=0,
                    dataset_dir=root,
                    preprocessed_dataset_dir=root,
                    num_epochs=2,
                    batch_size=2,
                    num_workers=0,
                    shuffle=False,
                )

        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset([0, 1])
        trainer._val_dataset = _TinyROIDataset([1, 0])
        trainer._architecture = _TinyNet().to(trainer.device)
        trainer._loss = nn.CrossEntropyLoss()
        trainer._optimizer = None
        trainer._scheduler = None
        return trainer

    def test_get_optimizer_uses_model_parameters(self) -> None:
        trainer = self._make_trainer()

        optimizer = trainer.get_optimizer()

        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            len(optimizer_parameters),
            len(list(trainer.get_architecture().parameters())),
        )
        self.assertSetEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in trainer.get_architecture().parameters()},
        )

    def test_train_step_returns_loss_and_accuracy_metrics(self) -> None:
        trainer = self._make_trainer()
        batch = next(iter(DataLoader(trainer.get_train_dataset(), batch_size=2)))
        before = trainer.get_architecture().fc.weight.detach().clone()

        metrics = trainer.train_step(batch, batch_idx=1)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics["num_samples"], 2)
        self.assertGreaterEqual(metrics["num_correct"], 0)
        self.assertLessEqual(metrics["num_correct"], 2)
        self.assertFalse(
            torch.equal(before, trainer.get_architecture().fc.weight.detach())
        )

    def test_validate_step_runs_without_recording_gradients(self) -> None:
        trainer = self._make_trainer()
        batch = next(iter(DataLoader(trainer.get_val_dataset(), batch_size=2)))

        metrics = trainer.validate_step(batch, batch_idx=1)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics["num_samples"], 2)
        self.assertGreaterEqual(metrics["num_correct"], 0)
        self.assertLessEqual(metrics["num_correct"], 2)
        for parameter in trainer.get_architecture().parameters():
            self.assertIsNone(parameter.grad)

    def test_fit_runs_end_to_end_and_steps_scheduler_each_epoch(self) -> None:
        trainer = self._make_trainer()
        trainer._optimizer = trainer.get_optimizer()
        scheduler = _CountingScheduler(trainer._optimizer)
        trainer._scheduler = scheduler

        train_loader = DataLoader(
            trainer.get_train_dataset(), batch_size=2, shuffle=False
        )
        val_loader = DataLoader(trainer.get_val_dataset(), batch_size=2, shuffle=False)

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            trainer.fit()

        self.assertEqual(scheduler.step_calls, trainer.num_epochs)
        self.assertIn("Epoch 1/2 - train_loss:", stdout.getvalue())
        self.assertIn("DONE", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
