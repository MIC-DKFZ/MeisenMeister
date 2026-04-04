from __future__ import annotations

import io
import math
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
        self.in_channels = 1
        self.num_classes = 2
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MMTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _make_trainer(
        self,
        *,
        continue_training: bool = False,
        num_epochs: int = 2,
    ) -> mmTrainer:
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
                dataset_dir=self.root / "Dataset_001_Test",
                preprocessed_dataset_dir=self.root / "Dataset_001_Test",
                results_dir=self.root / "results",
                architecture_name="ResNet3D18",
                num_epochs=num_epochs,
                batch_size=2,
                num_workers=0,
                shuffle=False,
                continue_training=continue_training,
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
        self.assertEqual(tuple(metrics["labels"].shape), (2,))
        self.assertEqual(tuple(metrics["probabilities"].shape), (2, 2))
        for parameter in trainer.get_architecture().parameters():
            self.assertIsNone(parameter.grad)

    def test_fit_creates_artifacts_and_logs_overwrite_warning(self) -> None:
        trainer = self._make_trainer()
        trainer.fold_dir.mkdir(parents=True, exist_ok=True)
        (trainer.fold_dir / "stale.txt").write_text("stale", encoding="utf-8")

        train_loader = [object()]
        val_loader = [object()]
        train_step_metrics = [
            {"loss": 0.8, "num_samples": 2, "num_correct": 1},
            {"loss": 0.7, "num_samples": 2, "num_correct": 2},
        ]
        validate_step_metrics = [
            {
                "loss": 0.6,
                "num_samples": 2,
                "num_correct": 2,
                "labels": torch.tensor([0, 1]),
                "predictions": torch.tensor([0, 1]),
                "probabilities": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            },
            {
                "loss": 0.5,
                "num_samples": 2,
                "num_correct": 2,
                "labels": torch.tensor([0, 1]),
                "predictions": torch.tensor([0, 1]),
                "probabilities": torch.tensor([[0.95, 0.05], [0.05, 0.95]]),
            },
        ]

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch.object(trainer, "train_step", side_effect=train_step_metrics),
            patch.object(trainer, "validate_step", side_effect=validate_step_metrics),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            trainer.fit()

        output = stdout.getvalue()
        self.assertIn("WARNING: YOU ARE OVERWRITING", output)
        self.assertTrue(trainer.log_path.is_file())
        self.assertTrue(trainer.last_checkpoint_path.is_file())
        self.assertTrue(trainer.best_checkpoint_path.is_file())
        self.assertTrue(trainer.plot_path.is_file())

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["last_completed_epoch"], 2)
        self.assertEqual(
            checkpoint["trainer_config"]["fold_dir"], str(trainer.fold_dir)
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])
        self.assertEqual(checkpoint["best_state"]["epoch"], 2)
        self.assertIn("Saved new best model at epoch 1", trainer.log_path.read_text())
        self.assertIn("Saved new best model at epoch 2", trainer.log_path.read_text())

    def test_fit_resume_continues_same_experiment_directory(self) -> None:
        initial_trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        initial_train_step_metrics = [{"loss": 0.8, "num_samples": 2, "num_correct": 1}]
        initial_validate_step_metrics = [
            {
                "loss": 0.6,
                "num_samples": 2,
                "num_correct": 2,
                "labels": torch.tensor([0, 1]),
                "predictions": torch.tensor([0, 1]),
                "probabilities": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            }
        ]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer, "train_step", side_effect=initial_train_step_metrics
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                side_effect=initial_validate_step_metrics,
            ),
        ):
            initial_trainer.fit()

        resumed_trainer = self._make_trainer(continue_training=True, num_epochs=2)
        resumed_train_step_metrics = [{"loss": 0.4, "num_samples": 2, "num_correct": 2}]
        resumed_validate_step_metrics = [
            {
                "loss": 0.4,
                "num_samples": 2,
                "num_correct": 2,
                "labels": torch.tensor([0, 1]),
                "predictions": torch.tensor([0, 1]),
                "probabilities": torch.tensor([[0.98, 0.02], [0.02, 0.98]]),
            }
        ]

        with (
            patch.object(
                resumed_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer, "train_step", side_effect=resumed_train_step_metrics
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                side_effect=resumed_validate_step_metrics,
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            resumed_trainer.fit()

        output = stdout.getvalue()
        self.assertIn("Resuming training from epoch 2", output)
        checkpoint = torch.load(
            resumed_trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])
        self.assertEqual(checkpoint["last_completed_epoch"], 2)
        self.assertEqual(checkpoint["best_state"]["epoch"], 2)

    def test_resume_falls_back_to_best_checkpoint_if_last_is_corrupted(self) -> None:
        initial_trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer,
                "train_step",
                return_value={"loss": 0.8, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                return_value={
                    "loss": 0.6,
                    "num_samples": 2,
                    "num_correct": 2,
                    "labels": torch.tensor([0, 1]),
                    "predictions": torch.tensor([0, 1]),
                    "probabilities": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                },
            ),
        ):
            initial_trainer.fit()

        initial_trainer.last_checkpoint_path.write_bytes(b"corrupted")

        resumed_trainer = self._make_trainer(continue_training=True, num_epochs=2)
        with (
            patch.object(
                resumed_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer,
                "train_step",
                return_value={"loss": 0.4, "num_samples": 2, "num_correct": 2},
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                return_value={
                    "loss": 0.4,
                    "num_samples": 2,
                    "num_correct": 2,
                    "labels": torch.tensor([0, 1]),
                    "predictions": torch.tensor([0, 1]),
                    "probabilities": torch.tensor([[0.98, 0.02], [0.02, 0.98]]),
                },
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            resumed_trainer.fit()

        output = stdout.getvalue()
        self.assertIn("resuming from fallback checkpoint", output.lower())
        checkpoint = torch.load(
            resumed_trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])

    def test_fit_handles_undefined_macro_auc_without_crashing(self) -> None:
        trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                trainer,
                "get_train_dataloader",
                return_value=train_loader,
            ),
            patch.object(
                trainer,
                "get_val_dataloader",
                return_value=val_loader,
            ),
            patch.object(
                trainer,
                "train_step",
                return_value={"loss": 0.9, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                trainer,
                "validate_step",
                return_value={
                    "loss": 0.7,
                    "num_samples": 2,
                    "num_correct": 2,
                    "labels": torch.tensor([0, 0]),
                    "predictions": torch.tensor([0, 0]),
                    "probabilities": torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
                },
            ),
        ):
            trainer.fit()

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertTrue(math.isnan(checkpoint["history"]["val_macro_auc"][0]))
        self.assertIn("val_macro_auc undefined", trainer.log_path.read_text())


if __name__ == "__main__":
    unittest.main()
