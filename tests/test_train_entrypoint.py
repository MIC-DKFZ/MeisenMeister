from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

train_module = importlib.import_module("meisenmeister.training.train")


class _MockTrainer:
    last_instance = None

    def __init__(
        self,
        dataset_id: str,
        fold: int,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str,
        continue_training: bool = False,
        weights_path: Path | None = None,
        experiment_postfix: str | None = None,
    ):
        self.dataset_id = dataset_id
        self.fold = fold
        self.dataset_dir = dataset_dir
        self.preprocessed_dataset_dir = preprocessed_dataset_dir
        self.results_dir = results_dir
        self.architecture_name = architecture_name
        self.continue_training = continue_training
        self.weights_path = weights_path
        self.experiment_postfix = experiment_postfix
        self.fit_called = False
        type(self).last_instance = self

    def fit(self) -> None:
        self.fit_called = True


class TrainEntrypointTests(unittest.TestCase):
    def test_train_resolves_and_runs_selected_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ) as mock_get_fold_sample_ids,
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ) as mock_get_trainer_class,
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    architecture_name="ResNet3D18",
                )

        mock_get_fold_sample_ids.assert_called_once_with(
            preprocessed_root / dataset_dir.name,
            0,
        )
        mock_get_trainer_class.assert_called_once_with("mmTrainer_Debug")
        self.assertIsNotNone(_MockTrainer.last_instance)
        self.assertEqual(_MockTrainer.last_instance.dataset_id, "001")
        self.assertEqual(_MockTrainer.last_instance.fold, 0)
        self.assertEqual(_MockTrainer.last_instance.architecture_name, "ResNet3D18")
        self.assertEqual(_MockTrainer.last_instance.dataset_dir, dataset_dir)
        self.assertEqual(
            _MockTrainer.last_instance.preprocessed_dataset_dir,
            preprocessed_root / dataset_dir.name,
        )
        self.assertEqual(_MockTrainer.last_instance.results_dir, root / "results")
        self.assertFalse(_MockTrainer.last_instance.continue_training)
        self.assertIsNone(_MockTrainer.last_instance.weights_path)
        self.assertIsNone(_MockTrainer.last_instance.experiment_postfix)
        self.assertTrue(_MockTrainer.last_instance.fit_called)

    def test_train_passes_continue_training_to_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    architecture_name="ResNet3D18",
                    continue_training=True,
                )

        self.assertTrue(_MockTrainer.last_instance.continue_training)

    def test_train_passes_weights_and_postfix_to_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    architecture_name="ResNet3D18",
                    weights_path="/tmp/pretrained.pt",
                    experiment_postfix="finetuningNNSSL",
                )

        self.assertEqual(
            _MockTrainer.last_instance.weights_path, Path("/tmp/pretrained.pt")
        )
        self.assertEqual(
            _MockTrainer.last_instance.experiment_postfix, "finetuningNNSSL"
        )

    def test_train_rejects_continue_training_with_weights(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Cannot use --continue-training and --weights together",
        ):
            train_module.train.__wrapped__(
                1,
                fold=0,
                continue_training=True,
                weights_path="/tmp/pretrained.pt",
            )


if __name__ == "__main__":
    unittest.main()
