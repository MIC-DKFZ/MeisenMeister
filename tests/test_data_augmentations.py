from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import blosc2
import numpy as np
import torch

from meisenmeister.data_augmentations import Compose3D, FlipAxes3D, apply_augmentations
from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.trainers.mm_trainer import mmTrainer


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_b2nd(path: Path, array: np.ndarray) -> None:
    blosc2.asarray(array, urlpath=str(path), mode="w")


class _AppendOrderMarker:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __call__(self, sample: dict) -> dict:
        order = list(sample.get("order", []))
        order.append(self.marker)
        return {**sample, "order": order}


class _ChangeShape:
    def __call__(self, sample: dict) -> dict:
        return {**sample, "image": sample["image"][:, :-1, :, :]}


class DataAugmentationTests(unittest.TestCase):
    def _make_sample(self) -> dict:
        return {
            "image": np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
            "label": torch.tensor(1, dtype=torch.long),
            "sample_id": "case_000_left",
            "case_id": "case_000",
            "roi_name": "left",
        }

    def test_compose3d_runs_augmentations_in_declared_order(self) -> None:
        pipeline = Compose3D(
            [_AppendOrderMarker("first"), _AppendOrderMarker("second")]
        )

        output = pipeline(
            {"image": np.zeros((1, 2, 2, 2), dtype=np.float32), "order": []}
        )

        self.assertEqual(output["order"], ["first", "second"])

    def test_apply_augmentations_with_empty_compose_returns_sample(self) -> None:
        sample = self._make_sample()

        output = apply_augmentations(sample, Compose3D([]), patch_size=(2, 2, 2))

        self.assertTrue(np.array_equal(output["image"], sample["image"]))
        self.assertEqual(output["sample_id"], sample["sample_id"])

    def test_flip_axes3d_probability_zero_never_changes_image(self) -> None:
        sample = self._make_sample()

        output = FlipAxes3D(probability=0.0, axes=(0, 1, 2))(sample)

        self.assertTrue(np.array_equal(output["image"], sample["image"]))

    def test_flip_axes3d_single_axis_uses_spatial_axes_only(self) -> None:
        sample = self._make_sample()

        output = FlipAxes3D(probability=1.0, axes=(0,))(sample)

        expected = np.flip(sample["image"], axis=(1,)).copy()
        self.assertTrue(np.array_equal(output["image"], expected))

    def test_flip_axes3d_multiple_axes(self) -> None:
        sample = self._make_sample()

        output = FlipAxes3D(probability=1.0, axes=(0, 2))(sample)

        expected = np.flip(sample["image"], axis=(1, 3)).copy()
        self.assertTrue(np.array_equal(output["image"], expected))

    def test_flip_axes3d_rejects_invalid_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "axes must be chosen from"):
            FlipAxes3D(probability=0.5, axes=(3,))

    def test_flip_axes3d_rejects_invalid_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability must be between 0 and 1"):
            FlipAxes3D(probability=1.5, axes=(0,))

    def test_apply_augmentations_rejects_shape_mismatch(self) -> None:
        sample = self._make_sample()

        with self.assertRaisesRegex(
            ValueError, "must preserve the designated patch size"
        ):
            apply_augmentations(
                sample,
                Compose3D([_ChangeShape()]),
                patch_size=(2, 2, 2),
            )


class DatasetAugmentationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        self.preprocessed_dataset_dir.mkdir()
        self.data_dir = self.preprocessed_dataset_dir / "mm_b2nd"
        self.data_dir.mkdir()

        _write_json(
            self.preprocessed_dataset_dir / "mmPlans.json",
            {
                "target_shape": [2, 2, 2],
                "output_folder_name": "mm_b2nd",
            },
        )
        _write_json(
            self.preprocessed_dataset_dir / "labelsTr.json",
            {
                "case_000_left": 1,
                "case_001_left": 0,
            },
        )
        self.base_array = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        _write_b2nd(self.data_dir / "case_000_left.b2nd", self.base_array)
        _write_b2nd(self.data_dir / "case_001_left.b2nd", self.base_array + 10.0)

    def test_dataset_applies_augmentations_and_preserves_metadata(self) -> None:
        dataset = MeisenmeisterROIDataset(
            self.preprocessed_dataset_dir,
            allowed_sample_ids={"case_000_left"},
            augmentation_pipeline=Compose3D([FlipAxes3D(probability=1.0, axes=(1,))]),
        )

        sample = dataset[0]

        expected = torch.flip(torch.from_numpy(self.base_array), dims=(2,))
        self.assertTrue(torch.equal(sample["image"], expected))
        self.assertEqual(sample["label"].item(), 1)
        self.assertEqual(sample["sample_id"], "case_000_left")
        self.assertEqual(sample["case_id"], "case_000")
        self.assertEqual(sample["roi_name"], "left")

    def test_mm_trainer_uses_augmentation_pipeline_for_train_only(self) -> None:
        with (
            patch(
                "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
                return_value={
                    "train": ["case_000_left"],
                    "val": ["case_001_left"],
                },
            ),
            patch.object(
                mmTrainer,
                "get_train_augmentation_pipeline",
                return_value=Compose3D([FlipAxes3D(probability=1.0, axes=(0, 1, 2))]),
            ),
        ):
            trainer = mmTrainer(
                dataset_id="001",
                fold=0,
                dataset_dir=self.root / "Dataset_001_Test",
                preprocessed_dataset_dir=self.preprocessed_dataset_dir,
                results_dir=self.root / "results",
                shuffle=False,
            )

        train_sample = trainer.get_train_dataset()[0]
        val_sample = trainer.get_val_dataset()[0]

        expected_train = torch.flip(torch.from_numpy(self.base_array), dims=(1, 2, 3))
        expected_val = torch.from_numpy(self.base_array + 10.0)
        self.assertTrue(torch.equal(train_sample["image"], expected_train))
        self.assertTrue(torch.equal(val_sample["image"], expected_val))
