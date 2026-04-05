from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister import cli


class CliTests(unittest.TestCase):
    def test_mm_train_uses_default_trainer(self) -> None:
        with (
            patch("sys.argv", ["mm_train", "-d", "1", "-f", "0"]),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer",
            architecture_name="ResNet3D18",
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val=None,
        )

    def test_mm_train_passes_explicit_trainer(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_train",
                    "-d",
                    "1",
                    "-f",
                    "3",
                    "--trainer",
                    "mmTrainer_Debug",
                    "-a",
                    "ResNet3D18",
                ],
            ),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=3,
            trainer_name="mmTrainer_Debug",
            architecture_name="ResNet3D18",
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val=None,
        )

    def test_mm_train_passes_continue_training_flag(self) -> None:
        with (
            patch("sys.argv", ["mm_train", "-d", "1", "-f", "0", "-c"]),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer",
            architecture_name="ResNet3D18",
            continue_training=True,
            weights_path=None,
            experiment_postfix=None,
            val=None,
        )

    def test_mm_train_passes_weights_and_postfix(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_train",
                    "-d",
                    "1",
                    "-f",
                    "0",
                    "-w",
                    "/tmp/pretrained.pt",
                    "--postfix",
                    "finetuningNNSSL",
                ],
            ),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer",
            architecture_name="ResNet3D18",
            continue_training=False,
            weights_path="/tmp/pretrained.pt",
            experiment_postfix="finetuningNNSSL",
            val=None,
        )

    def test_mm_predict_uses_defaults(self) -> None:
        with (
            patch(
                "sys.argv",
                ["mm_predict", "-d", "1", "-i", "/tmp/in", "-o", "/tmp/out", "-f", "0"],
            ),
            patch("meisenmeister.cli.predict") as mock_predict,
        ):
            cli.mm_predict()

        mock_predict.assert_called_once_with(
            1,
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=[0],
            trainer_name="mmTrainer",
            architecture_name="ResNet3D18",
            experiment_postfix=None,
            checkpoint="best",
            use_tta=True,
        )

    def test_mm_predict_passes_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_predict",
                    "-d",
                    "1",
                    "-i",
                    "/tmp/in",
                    "-o",
                    "/tmp/out",
                    "-f",
                    "0",
                    "2",
                    "--trainer",
                    "mmTrainer_Debug",
                    "-a",
                    "ResNet3D18",
                    "--postfix",
                    "finetuningNNSSL",
                    "--checkpoint",
                    "last",
                    "--no-tta",
                ],
            ),
            patch("meisenmeister.cli.predict") as mock_predict,
        ):
            cli.mm_predict()

        mock_predict.assert_called_once_with(
            1,
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=[0, 2],
            trainer_name="mmTrainer_Debug",
            architecture_name="ResNet3D18",
            experiment_postfix="finetuningNNSSL",
            checkpoint="last",
            use_tta=False,
        )

    def test_mm_create_5fold_writes_to_preprocessed_dataset(self) -> None:
        with (
            patch("sys.argv", ["mm_create_5fold", "-d", "1"]),
            patch("builtins.print"),
            patch(
                "meisenmeister.cli.verify_required_global_paths_set",
                return_value={
                    "mm_raw": Path("/tmp/raw"),
                    "mm_preprocessed": Path("/tmp/pre"),
                },
            ),
            patch(
                "meisenmeister.cli.find_dataset_dir",
                return_value=Path("/tmp/raw/Dataset_001_Test"),
            ),
            patch("meisenmeister.cli.create_five_fold_splits") as mock_create_splits,
        ):
            cli.mm_create_5fold.__wrapped__()

        mock_create_splits.assert_called_once_with(Path("/tmp/pre/Dataset_001_Test"))


if __name__ == "__main__":
    unittest.main()
