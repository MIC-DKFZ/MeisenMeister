from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from meisenmeister.utils.file_utils import discover_case_files

predict_module = importlib.import_module("meisenmeister.training.predict")


class _ConstantModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        del x
        return self._logits.unsqueeze(0)


class _PortableModel(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.received = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            **kwargs,
        }
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        del x
        return torch.zeros((1, 2), dtype=torch.float32)


class PredictEntrypointTests(unittest.TestCase):
    def test_iter_prepared_case_prediction_inputs_yields_sorted_cases(self) -> None:
        case_files = {
            "case_002": [Path("/tmp/case_002_0000.nii.gz")],
            "case_001": [Path("/tmp/case_001_0000.nii.gz")],
        }
        breast_mask_paths = {
            "case_001": Path("/tmp/case_001_breast_mask.nii.gz"),
            "case_002": Path("/tmp/case_002_breast_mask.nii.gz"),
        }

        def _fake_prepare_case_prediction_inputs(**kwargs):
            case_id = kwargs["case_id"]
            return (
                {"left": f"tensor-{case_id}"},
                {"breast_mask": str(kwargs["breast_mask_path"])},
            )

        with patch(
            "meisenmeister.training.predict._prepare_case_prediction_inputs",
            side_effect=_fake_prepare_case_prediction_inputs,
        ):
            prepared = list(
                predict_module._iter_prepared_case_prediction_inputs(
                    case_files_by_case_id=case_files,
                    breast_mask_paths=breast_mask_paths,
                    dataset_json={"file_ending": ".nii.gz"},
                    plans={"roi_labels": {"left": 1}},
                    output_dir=Path("/tmp/output"),
                    num_workers=1,
                )
            )

        self.assertEqual([item[0] for item in prepared], ["case_001", "case_002"])
        self.assertEqual(prepared[0][1]["left"], "tensor-case_001")
        self.assertEqual(
            prepared[1][2]["breast_mask"],
            "/tmp/case_002_breast_mask.nii.gz",
        )

    def test_generate_breast_masks_for_cases_runs_single_directory_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.nii.gz",
                    input_dir / "case_001_0001.nii.gz",
                ],
                "case_002": [
                    input_dir / "case_002_0000.nii.gz",
                    input_dir / "case_002_0001.nii.gz",
                ],
            }
            captured_calls: list[tuple[Path, Path]] = []

            class _Predictor:
                def predict(self, *, input_path, output_path):
                    input_path = Path(input_path)
                    output_path = Path(output_path)
                    captured_calls.append((input_path, output_path))
                    staged_files = sorted(path.name for path in input_path.iterdir())
                    if staged_files != ["case_001_0000.nii.gz", "case_002_0000.nii.gz"]:
                        raise AssertionError(f"Unexpected staged files: {staged_files}")
                    (output_path / "case_001.nii.gz").write_text(
                        "mask-1", encoding="utf-8"
                    )
                    (output_path / "case_002.nii.gz").write_text(
                        "mask-2", encoding="utf-8"
                    )

            mask_paths = predict_module._generate_breast_masks_for_cases(
                case_files_by_case_id=case_files,
                dataset_json={"file_ending": ".nii.gz"},
                predictor=_Predictor(),
                output_dir=output_dir,
            )
            self.assertEqual(len(captured_calls), 1)
            self.assertEqual(
                sorted(path.name for path in mask_paths.values()),
                ["case_001_breast_mask.nii.gz", "case_002_breast_mask.nii.gz"],
            )
            self.assertEqual(
                (output_dir / "case_001_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "mask-1",
            )
            self.assertEqual(
                (output_dir / "case_002_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "mask-2",
            )

    def test_generate_breast_masks_for_cases_skips_existing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")
            (output_dir / "case_001_breast_mask.nii.gz").write_text(
                "existing-mask", encoding="utf-8"
            )

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.nii.gz",
                    input_dir / "case_001_0001.nii.gz",
                ],
                "case_002": [
                    input_dir / "case_002_0000.nii.gz",
                    input_dir / "case_002_0001.nii.gz",
                ],
            }
            captured_calls: list[tuple[Path, Path]] = []

            class _Predictor:
                def predict(self, *, input_path, output_path):
                    input_path = Path(input_path)
                    output_path = Path(output_path)
                    captured_calls.append((input_path, output_path))
                    staged_files = sorted(path.name for path in input_path.iterdir())
                    if staged_files != ["case_002_0000.nii.gz"]:
                        raise AssertionError(f"Unexpected staged files: {staged_files}")
                    (output_path / "case_002.nii.gz").write_text(
                        "new-mask", encoding="utf-8"
                    )

            mask_paths = predict_module._generate_breast_masks_for_cases(
                case_files_by_case_id=case_files,
                dataset_json={"file_ending": ".nii.gz"},
                predictor=_Predictor(),
                output_dir=output_dir,
            )

            self.assertEqual(len(captured_calls), 1)
            self.assertEqual(
                (output_dir / "case_001_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "existing-mask",
            )
            self.assertEqual(
                (output_dir / "case_002_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "new-mask",
            )
            self.assertEqual(
                mask_paths["case_001"],
                output_dir / "case_001_breast_mask.nii.gz",
            )
            self.assertEqual(
                mask_paths["case_002"],
                output_dir / "case_002_breast_mask.nii.gz",
            )

    def test_predict_roi_with_tta_averages_all_flip_variants(self) -> None:
        model = _ConstantModel([1.0, 3.0])
        roi_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)

        probabilities = predict_module._predict_roi_with_tta(
            model,
            roi_tensor,
            device=torch.device("cpu"),
            use_tta=True,
        )

        expected = torch.softmax(torch.tensor([1.0, 3.0]), dim=0).numpy()
        np.testing.assert_allclose(probabilities, expected)
        self.assertEqual(len(predict_module._get_flip_axes(True)), 8)

    def test_discover_case_files_groups_multiple_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).touch()

            dataset_json = {
                "channel_names": {"0": "a", "1": "b"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "neg", "1": "pos"},
                "numTraining": 2,
            }

            discovered = discover_case_files(input_dir, dataset_json)

        self.assertEqual(sorted(discovered), ["case_001", "case_002"])
        self.assertEqual(len(discovered["case_001"]), 2)
        self.assertEqual(len(discovered["case_002"]), 2)

    def test_discover_case_files_rejects_missing_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "case_001_0000.nii.gz").touch()
            dataset_json = {
                "channel_names": {"0": "a", "1": "b"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "neg", "1": "pos"},
                "numTraining": 1,
            }

            with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                discover_case_files(input_dir, dataset_json)

    def test_predict_writes_combined_json_and_flat_mask_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "predictions"
            input_dir = root / "input"
            input_dir.mkdir()
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            preprocessed_dir = preprocessed_root / dataset_dir.name
            dataset_dir.mkdir()
            preprocessed_dir.mkdir(parents=True)

            case_files = {
                "case_001": [input_dir / "case_001_0000.nii.gz"],
                "case_002": [input_dir / "case_002_0000.nii.gz"],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.nii.gz"
                left_mask = case_output_dir / f"{case_id}_left_mask.nii.gz"
                right_mask = case_output_dir / f"{case_id}_right_mask.nii.gz"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.predict._resolve_trainer_architecture_name",
                    return_value="ResNet3D18",
                ),
                patch(
                    "meisenmeister.training.predict.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "channel_names": {"0": "image"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {"0": "neg", "1": "pos"},
                        "numTraining": 2,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_mm_plans",
                    return_value={
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    },
                ),
                patch(
                    "meisenmeister.training.predict.discover_case_files",
                    return_value=case_files,
                ),
                patch(
                    "meisenmeister.training.predict.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.predict._generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.nii.gz",
                        "case_002": output_dir / "case_002_breast_mask.nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict._load_fold_predictors",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": "/tmp/fold_0.pt",
                            "model": object(),
                        },
                        {
                            "fold": 1,
                            "device": torch.device("cpu"),
                            "checkpoint_path": "/tmp/fold_1.pt",
                            "model": object(),
                        },
                    ],
                ),
                patch(
                    "meisenmeister.training.predict._prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.predict._predict_roi_with_tta",
                    side_effect=[
                        np.array([0.2, 0.8], dtype=np.float32),
                        np.array([0.4, 0.6], dtype=np.float32),
                        np.array([0.7, 0.3], dtype=np.float32),
                        np.array([0.6, 0.4], dtype=np.float32),
                        np.array([0.1, 0.9], dtype=np.float32),
                        np.array([0.3, 0.7], dtype=np.float32),
                        np.array([0.8, 0.2], dtype=np.float32),
                        np.array([0.5, 0.5], dtype=np.float32),
                    ],
                ),
            ):
                predictions_path = predict_module.predict.__wrapped__(
                    1,
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0, 1],
                    trainer_name="mmTrainer",
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                )

            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(predictions_path, output_dir / "predictions.json")
            self.assertTrue((output_dir / "case_001_breast_mask.nii.gz").is_file())
            self.assertTrue((output_dir / "case_001_left_mask.nii.gz").is_file())
            self.assertTrue((output_dir / "case_001_right_mask.nii.gz").is_file())
            self.assertIn("case_001", payload["cases"])
            self.assertIn("case_002", payload["cases"])
            self.assertTrue(payload["config"]["tta_enabled"])
            self.assertEqual(payload["config"]["checkpoint"], "best")
            left_case_001 = payload["cases"]["case_001"]["rois"]["left"]
            self.assertEqual(left_case_001["prediction"], 1)
            self.assertEqual(sorted(left_case_001["per_fold"]), ["0", "1"])

    def test_predict_uses_literal_fold_all_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            preprocessed_dir = preprocessed_root / dataset_dir.name
            results_dir = root / "results"
            experiment_dir = results_dir / dataset_dir.name / "mmTrainer_ResNet3D18"
            dataset_dir.mkdir()
            preprocessed_dir.mkdir(parents=True)
            (experiment_dir / "fold_all").mkdir(parents=True)

            with (
                patch(
                    "meisenmeister.training.predict._resolve_trainer_architecture_name",
                    return_value="ResNet3D18",
                ),
                patch(
                    "meisenmeister.training.predict.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": results_dir,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "file_ending": ".nii.gz",
                        "channel_names": {"0": "image"},
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_mm_plans",
                    return_value={"roi_labels": {"left": 1}},
                ),
                patch(
                    "meisenmeister.training.predict._load_fold_predictors",
                    return_value=[],
                ) as mock_load_fold_predictors,
                patch(
                    "meisenmeister.training.predict._run_prediction",
                    return_value=Path("/tmp/predictions.json"),
                ) as mock_run_prediction,
            ):
                predict_module.predict.__wrapped__(
                    1,
                    input_dir=str(root / "input"),
                    output_dir=str(root / "output"),
                    folds=["all"],
                    num_workers=8,
                )

        self.assertEqual(mock_load_fold_predictors.call_args.kwargs["folds"], ["all"])
        self.assertEqual(mock_run_prediction.call_args.kwargs["folds"], ["all"])

    def test_predict_from_modelfolder_writes_outputs_without_global_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = (
                root / "results" / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            output_dir = root / "predictions"
            input_dir = root / "input"
            input_dir.mkdir()
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "dataset.json").write_text(
                json.dumps(
                    {
                        "channel_names": {"0": "image"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {"0": "neg", "1": "pos"},
                        "numTraining": 2,
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "mmPlans.json").write_text(
                json.dumps(
                    {
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    }
                ),
                encoding="utf-8",
            )

            case_files = {
                "case_001": [input_dir / "case_001_0000.nii.gz"],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.nii.gz"
                left_mask = case_output_dir / f"{case_id}_left_mask.nii.gz"
                right_mask = case_output_dir / f"{case_id}_right_mask.nii.gz"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.predict.discover_case_files",
                    return_value=case_files,
                ),
                patch(
                    "meisenmeister.training.predict.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.predict._generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict._get_experiment_metadata",
                    return_value={
                        "dataset_id": "001",
                        "dataset_name": "Dataset_001_Test",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                        "experiment_postfix": None,
                    },
                ),
                patch(
                    "meisenmeister.training.predict._load_fold_predictors_from_experiment_dir",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": str(
                                experiment_dir / "fold_0" / "model_best.pt"
                            ),
                            "model": object(),
                        }
                    ],
                ),
                patch(
                    "meisenmeister.training.predict._prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.predict._predict_roi_with_tta",
                    side_effect=[
                        np.array([0.2, 0.8], dtype=np.float32),
                        np.array([0.7, 0.3], dtype=np.float32),
                    ],
                ),
            ):
                predictions_path = predict_module.predict_from_modelfolder(
                    str(experiment_dir),
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0],
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                )

            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(predictions_path, output_dir / "predictions.json")
            self.assertEqual(payload["config"]["dataset_id"], "001")
            self.assertEqual(payload["config"]["trainer_name"], "mmTrainer")
            self.assertEqual(sorted(payload["cases"]), ["case_001"])

    def test_get_experiment_metadata_reads_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = (
                Path(temp_dir) / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            fold_dir = experiment_dir / "fold_0"
            fold_dir.mkdir(parents=True)
            torch.save(
                {
                    "trainer_config": {
                        "dataset_id": "001",
                        "dataset_name": "Dataset_001_Test",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                        "experiment_postfix": "portable",
                        "in_channels": 1,
                        "num_classes": 2,
                    },
                    "model_state_dict": {},
                },
                fold_dir / "model_best.pt",
            )

            metadata = predict_module._get_experiment_metadata(
                experiment_dir,
                [0],
                "best",
            )

        self.assertEqual(metadata["dataset_id"], "001")
        self.assertEqual(metadata["dataset_name"], "Dataset_001_Test")
        self.assertEqual(metadata["trainer_name"], "mmTrainer")
        self.assertEqual(metadata["architecture_name"], "ResNet3D18")
        self.assertEqual(metadata["experiment_postfix"], "portable")

    def test_load_model_from_checkpoint_payload_passes_architecture_kwargs(
        self,
    ) -> None:
        checkpoint_payload = {
            "trainer_config": {
                "in_channels": 1,
                "num_classes": 2,
                "architecture_kwargs": {
                    "input_shape": (16, 16, 16),
                    "patch_embed_size": (8, 8, 8),
                },
            },
            "model_state_dict": _PortableModel(1, 2).state_dict(),
        }

        with (
            patch(
                "meisenmeister.training.predict.get_architecture_class",
                return_value=_PortableModel,
            ),
            patch(
                "meisenmeister.training.predict.maybe_compile_model",
                side_effect=lambda model, device, enabled: (
                    model,
                    False,
                    "compile disabled",
                ),
            ),
        ):
            model, compile_applied, compile_status = (
                predict_module._load_model_from_checkpoint_payload(
                    checkpoint_payload=checkpoint_payload,
                    architecture_name="PrimusMClsNetwork",
                    device=torch.device("cpu"),
                    compile_model=False,
                )
            )

        self.assertFalse(compile_applied)
        self.assertEqual(compile_status, "compile disabled")
        self.assertEqual(
            model.received,
            {
                "in_channels": 1,
                "num_classes": 2,
                "input_shape": (16, 16, 16),
                "patch_embed_size": (8, 8, 8),
            },
        )

    def test_predict_uses_single_identity_variant_when_tta_disabled(self) -> None:
        with patch("meisenmeister.training.predict._get_flip_axes", return_value=[()]):
            model = _ConstantModel([2.0, 1.0])
            roi_tensor = torch.ones((1, 2, 2, 2), dtype=torch.float32)

            probabilities = predict_module._predict_roi_with_tta(
                model,
                roi_tensor,
                device=torch.device("cpu"),
                use_tta=False,
            )

        expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0).numpy()
        np.testing.assert_allclose(probabilities, expected)
