import json
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_label_value(label_value):
    if isinstance(label_value, list):
        if len(label_value) == 0:
            raise ValueError("Encountered empty label list")
        return int(np.argmax(np.asarray(label_value)))
    if isinstance(label_value, int):
        return label_value
    raise TypeError(f"Unsupported label value type: {type(label_value).__name__}")


def _parse_sample_id(path: Path) -> tuple[str, str, str]:
    sample_id = path.stem
    if "_" not in sample_id:
        raise ValueError(
            f"Malformed sample filename '{path.name}'. Expected <case_id>_<roi>.b2nd"
        )

    case_id, roi_name = sample_id.rsplit("_", 1)
    if not case_id or not roi_name:
        raise ValueError(
            f"Malformed sample filename '{path.name}'. Expected <case_id>_<roi>.b2nd"
        )
    return sample_id, case_id, roi_name


class MeisenmeisterROIDataset(Dataset):
    def __init__(self, preprocessed_dataset_dir: Path):
        self.preprocessed_dataset_dir = preprocessed_dataset_dir
        self.plans = _load_json(preprocessed_dataset_dir / "mmPlans.json")
        self.labels = _load_json(preprocessed_dataset_dir / "labelsTr.json")
        self.data_dir = preprocessed_dataset_dir / self.plans["output_folder_name"]
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Missing preprocessed data directory: {self.data_dir}"
            )

        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        samples = []
        for path in sorted(self.data_dir.glob("*.b2nd")):
            sample_id, case_id, roi_name = _parse_sample_id(path)
            if sample_id not in self.labels:
                raise KeyError(
                    f"Missing label for sample '{sample_id}' in labelsTr.json"
                )

            samples.append(
                {
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "roi_name": roi_name,
                    "path": path,
                    "label": _normalize_label_value(self.labels[sample_id]),
                }
            )

        if not samples:
            raise ValueError(f"No .b2nd samples found in {self.data_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = np.asarray(blosc2.open(str(sample["path"]), mode="r"), dtype=np.float32)
        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "roi_name": sample["roi_name"],
        }
