from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meisenmeister.dataset_conversion.ambl_to_odelia_ambl import (
    write_odelia_ambl_dataset,
)
from meisenmeister.utils.file_utils import (
    load_dataset_json,
    verify_training_files_present,
)


class TrainingCasePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def test_verify_training_files_present_uses_training_cases_paths(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        shared_dir = self.root / "shared"
        absolute_dir = self.root / "absolute"
        relative_image = shared_dir / "case_001_0000.nii.gz"
        absolute_image = absolute_dir / "case_001_0001.nii.gz"
        relative_image.parent.mkdir(parents=True)
        absolute_image.parent.mkdir(parents=True)
        relative_image.write_text("c0", encoding="utf-8")
        absolute_image.write_text("c1", encoding="utf-8")

        self._write_json(
            dataset_dir / "dataset.json",
            {
                "channel_names": {"0": "pre", "1": "post1"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "healthy", "1": "malignant"},
                "numTraining": 1,
                "training_cases": {
                    "case_001": {
                        "0": "../shared/case_001_0000.nii.gz",
                        "1": str(absolute_image),
                    }
                },
            },
        )

        dataset_json = load_dataset_json(dataset_dir)
        resolved = verify_training_files_present(dataset_dir, dataset_json)

        self.assertEqual(sorted(resolved), ["case_001"])
        self.assertEqual(
            resolved["case_001"],
            [relative_image, absolute_image],
        )

    def test_load_dataset_json_rejects_missing_declared_channel(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        image_path = self.root / "shared" / "case_001_0000.nii.gz"
        image_path.parent.mkdir(parents=True)
        image_path.write_text("c0", encoding="utf-8")

        self._write_json(
            dataset_dir / "dataset.json",
            {
                "channel_names": {"0": "pre", "1": "post1"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "healthy", "1": "malignant"},
                "numTraining": 1,
                "training_cases": {
                    "case_001": {
                        "0": "../shared/case_001_0000.nii.gz",
                    }
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "must define exactly channels"):
            load_dataset_json(dataset_dir)

    def test_write_odelia_ambl_dataset_builds_combined_training_cases(self) -> None:
        odelia_dataset_dir = self.root / "mm_raw" / "Dataset_002_Odelia"
        odelia_images_dir = odelia_dataset_dir / "imagesTr"
        odelia_images_dir.mkdir(parents=True)
        for filename in (
            "ODELIA_001_0000.nii.gz",
            "ODELIA_001_0001.nii.gz",
            "ODELIA_001_0002.nii.gz",
        ):
            (odelia_images_dir / filename).write_text(filename, encoding="utf-8")

        self._write_json(
            odelia_dataset_dir / "dataset.json",
            {
                "channel_names": {"0": "pre", "1": "post1", "2": "post2"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "healthy", "1": "benign", "2": "malignant"},
                "numTraining": 1,
            },
        )
        self._write_json(
            odelia_dataset_dir / "labelsTr.json",
            {
                "ODELIA_001_left": [1, 0, 0],
                "ODELIA_001_right": [0, 1, 0],
            },
        )

        ambl_dataset_dir = self.root / "AMBL"
        ambl_images_dir = ambl_dataset_dir / "imagesTr"
        ambl_images_dir.mkdir(parents=True)
        for filename in (
            "AMBL-001_0000.nii.gz",
            "AMBL-001_0001.nii.gz",
            "AMBL-001_0002.nii.gz",
            "AMBL-001_0003.nii.gz",
        ):
            (ambl_images_dir / filename).write_text(filename, encoding="utf-8")
        self._write_json(
            ambl_dataset_dir / "labelsTr.json",
            {
                "AMBL-001_left": [0, 0, 1],
                "AMBL-001_right": [1, 0, 0],
            },
        )

        output_dataset_dir = self.root / "mm_raw" / "Dataset_003_Odelia_AMBL"
        write_odelia_ambl_dataset(
            odelia_dataset_dir=odelia_dataset_dir,
            ambl_dataset_dir=ambl_dataset_dir,
            output_dataset_dir=output_dataset_dir,
        )

        dataset_json = json.loads((output_dataset_dir / "dataset.json").read_text())
        labels_json = json.loads((output_dataset_dir / "labelsTr.json").read_text())

        self.assertEqual(dataset_json["numTraining"], 2)
        self.assertEqual(
            sorted(dataset_json["training_cases"]), ["AMBL-001", "ODELIA_001"]
        )
        self.assertEqual(
            dataset_json["training_cases"]["ODELIA_001"],
            {
                "0": "../Dataset_002_Odelia/imagesTr/ODELIA_001_0000.nii.gz",
                "1": "../Dataset_002_Odelia/imagesTr/ODELIA_001_0001.nii.gz",
                "2": "../Dataset_002_Odelia/imagesTr/ODELIA_001_0002.nii.gz",
            },
        )
        self.assertEqual(
            dataset_json["training_cases"]["AMBL-001"],
            {
                "0": "../../AMBL/imagesTr/AMBL-001_0000.nii.gz",
                "1": "../../AMBL/imagesTr/AMBL-001_0001.nii.gz",
                "2": "../../AMBL/imagesTr/AMBL-001_0002.nii.gz",
            },
        )
        self.assertNotIn("training_cases", load_dataset_json(odelia_dataset_dir))
        self.assertEqual(
            labels_json,
            {
                "ODELIA_001_left": [1, 0, 0],
                "ODELIA_001_right": [0, 1, 0],
                "AMBL-001_left": [0, 0, 1],
                "AMBL-001_right": [1, 0, 0],
            },
        )


if __name__ == "__main__":
    unittest.main()
