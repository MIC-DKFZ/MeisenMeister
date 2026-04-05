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


class PredictEntrypointTests(unittest.TestCase):
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
                    checkpoint="best",
                    use_tta=True,
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
