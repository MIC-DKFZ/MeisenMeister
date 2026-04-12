from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister import cli


class CliTests(unittest.TestCase):
    def test_mm_plan_experiment_passes_dataset_id(self) -> None:
        with (
            patch("sys.argv", ["mm_plan_experiment", "-d", "1"]),
            patch("meisenmeister.cli.plan_experiment") as mock_plan_experiment,
        ):
            cli.mm_plan_experiment()

        mock_plan_experiment.assert_called_once_with(1)

    def test_mm_train_accepts_all_fold(self) -> None:
        with (
            patch("sys.argv", ["mm_train", "-d", "1", "-f", "all"]),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold="all",
            trainer_name="mmTrainer",
            num_workers=None,
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val=None,
            compile_enabled=True,
            grad_cam_enabled=False,
        )

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
            num_workers=None,
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val=None,
            compile_enabled=True,
            grad_cam_enabled=False,
        )

    def test_mm_preview_da_passes_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_preview_da",
                    "-d",
                    "1",
                    "-f",
                    "0",
                    "--trainer",
                    "mmTrainer_Debug",
                    "--num-workers",
                    "4",
                    "--postfix",
                    "debug",
                    "--num-samples",
                    "5",
                    "--output",
                    "/tmp/da_preview.png",
                ],
            ),
            patch("meisenmeister.cli.preview_da") as mock_preview_da,
        ):
            cli.mm_preview_da()

        mock_preview_da.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer_Debug",
            num_workers=4,
            experiment_postfix="debug",
            num_samples=5,
            output_path="/tmp/da_preview.png",
        )

    def test_mm_preview_da_prints_output_path(self) -> None:
        with (
            patch("sys.argv", ["mm_preview_da", "-d", "1", "-f", "0"]),
            patch(
                "meisenmeister.cli.preview_da",
                return_value=Path("/tmp/da_preview.png"),
            ),
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            cli.mm_preview_da()

        self.assertEqual(mock_stdout.getvalue().strip(), "/tmp/da_preview.png")

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
                ],
            ),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=3,
            trainer_name="mmTrainer_Debug",
            num_workers=None,
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val=None,
            compile_enabled=True,
            grad_cam_enabled=False,
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
            num_workers=None,
            continue_training=True,
            weights_path=None,
            experiment_postfix=None,
            val=None,
            compile_enabled=True,
            grad_cam_enabled=False,
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
            num_workers=None,
            continue_training=False,
            weights_path="/tmp/pretrained.pt",
            experiment_postfix="finetuningNNSSL",
            val=None,
            compile_enabled=True,
            grad_cam_enabled=False,
        )

    def test_mm_train_passes_grad_cam_flag(self) -> None:
        with (
            patch(
                "sys.argv",
                ["mm_train", "-d", "1", "-f", "0", "--val", "best", "--grad-cam"],
            ),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer",
            num_workers=None,
            continue_training=False,
            weights_path=None,
            experiment_postfix=None,
            val="best",
            compile_enabled=True,
            grad_cam_enabled=True,
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
            experiment_postfix=None,
            checkpoint="best",
            use_tta=True,
            compile_model=True,
            num_workers=8,
            concise_output_path=None,
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
                    "--postfix",
                    "finetuningNNSSL",
                    "--checkpoint",
                    "last",
                    "--no-tta",
                    "--num-workers",
                    "4",
                    "--concise-output",
                    "/tmp/concise.json",
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
            experiment_postfix="finetuningNNSSL",
            checkpoint="last",
            use_tta=False,
            compile_model=True,
            num_workers=4,
            concise_output_path="/tmp/concise.json",
        )

    def test_mm_predict_accepts_all_fold_keyword(self) -> None:
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
                    "all",
                ],
            ),
            patch("meisenmeister.cli.predict") as mock_predict,
        ):
            cli.mm_predict()

        mock_predict.assert_called_once_with(
            1,
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=["all"],
            trainer_name="mmTrainer",
            experiment_postfix=None,
            checkpoint="best",
            use_tta=True,
            compile_model=True,
            num_workers=8,
            concise_output_path=None,
        )

    def test_mm_predict_from_modelfolder_passes_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_predict_from_modelfolder",
                    "/tmp/model",
                    "-i",
                    "/tmp/in",
                    "-o",
                    "/tmp/out",
                    "-f",
                    "0",
                    "2",
                    "--checkpoint",
                    "last",
                    "--no-tta",
                    "--num-workers",
                    "3",
                    "--concise-output",
                    "/tmp/concise.json",
                ],
            ),
            patch("meisenmeister.cli.predict_from_modelfolder") as mock_predict,
        ):
            cli.mm_predict_from_modelfolder()

        mock_predict.assert_called_once_with(
            "/tmp/model",
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=[0, 2],
            checkpoint="last",
            use_tta=False,
            compile_model=True,
            num_workers=3,
            concise_output_path="/tmp/concise.json",
        )

    def test_mm_predict_from_modelfolder_accepts_all_fold_keyword(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_predict_from_modelfolder",
                    "/tmp/model",
                    "-i",
                    "/tmp/in",
                    "-o",
                    "/tmp/out",
                    "-f",
                    "all",
                ],
            ),
            patch("meisenmeister.cli.predict_from_modelfolder") as mock_predict,
        ):
            cli.mm_predict_from_modelfolder()

        mock_predict.assert_called_once_with(
            "/tmp/model",
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=["all"],
            checkpoint="best",
            use_tta=True,
            compile_model=True,
            num_workers=8,
            concise_output_path=None,
        )

    def test_mm_evaluate_predictions_passes_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_evaluate_predictions",
                    "-t",
                    "/tmp/targets.json",
                    "-p",
                    "/tmp/predictions.json",
                    "-o",
                    "/tmp/evaluation.json",
                ],
            ),
            patch(
                "meisenmeister.cli.evaluate_predictions"
            ) as mock_evaluate_predictions,
        ):
            cli.mm_evaluate_predictions()

        mock_evaluate_predictions.assert_called_once_with(
            targets_path="/tmp/targets.json",
            predictions_path="/tmp/predictions.json",
            output_path="/tmp/evaluation.json",
        )

    def test_mm_benchmark_train_passes_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_benchmark_train",
                    "-d",
                    "1",
                    "-f",
                    "0",
                    "--trainer",
                    "mmTrainer_Debug",
                    "--num-workers",
                    "6",
                    "--postfix",
                    "bench",
                    "--train-warmup-steps",
                    "4",
                    "--train-steps",
                    "12",
                    "--val-warmup-steps",
                    "3",
                    "--val-steps",
                    "7",
                    "--disable-compile",
                ],
            ),
            patch("meisenmeister.cli.benchmark_train") as mock_benchmark,
        ):
            cli.mm_benchmark_train()

        mock_benchmark.assert_called_once_with(
            1,
            fold=0,
            trainer_name="mmTrainer_Debug",
            num_workers=6,
            experiment_postfix="bench",
            compile_enabled=False,
            train_warmup_steps=4,
            train_steps=12,
            val_warmup_steps=3,
            val_steps=7,
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
