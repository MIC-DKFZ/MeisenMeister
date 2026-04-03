import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def _load_mm_plans(plans_path: Path) -> dict:
    if not plans_path.is_file():
        raise FileNotFoundError(
            f"Missing mmPlans.json in {plans_path.parent}. Run mm_plan_experment first."
        )

    with plans_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_case_channel_files(case_files: list[Path], dataset_json: dict) -> list[Path]:
    file_ending = dataset_json["file_ending"]
    sorted_channel_ids = sorted(
        int(channel_id) for channel_id in dataset_json["channel_names"]
    )
    ordered_files: list[Path] = []
    for channel_id in sorted_channel_ids:
        suffix = f"_{channel_id:04d}{file_ending}"
        matched = next(
            (path for path in case_files if path.name.endswith(suffix)), None
        )
        if matched is None:
            raise FileNotFoundError(f"Missing channel file with suffix {suffix}")
        ordered_files.append(matched)
    return ordered_files


def _compute_bbox(
    mask_array: np.ndarray, roi_label: int, margin_mm: list[float], spacing: list[float]
):
    roi_voxels = np.argwhere(mask_array == roi_label)
    if roi_voxels.size == 0:
        raise ValueError(f"ROI label {roi_label} not present in mask")

    min_corner = roi_voxels.min(axis=0)
    max_corner = roi_voxels.max(axis=0)
    margin_voxels = np.ceil(
        np.asarray(margin_mm, dtype=np.float64) / np.asarray(spacing, dtype=np.float64)
    ).astype(int)
    lower = np.maximum(min_corner - margin_voxels, 0)
    upper = np.minimum(
        max_corner + margin_voxels + 1, np.asarray(mask_array.shape, dtype=int)
    )
    return lower, upper


def _resample_array(
    array: np.ndarray,
    input_spacing: list[float],
    target_spacing: list[float],
    is_mask: bool,
) -> np.ndarray:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(reversed(input_spacing)))

    input_size = image.GetSize()
    output_size = [
        max(1, int(round(size * spacing / target)))
        for size, spacing, target in zip(
            input_size, reversed(input_spacing), reversed(target_spacing), strict=True
        )
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(reversed(target_spacing)))
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    result = resampler.Execute(image)
    return sitk.GetArrayFromImage(result)


def _fit_to_target_shape(array: np.ndarray, target_shape: list[int]) -> np.ndarray:
    fitted = array
    for axis, target in enumerate(target_shape):
        current = fitted.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * fitted.ndim
            slices[axis] = slice(start, end)
            fitted = fitted[tuple(slices)]

    pad_width = []
    for current, target in zip(fitted.shape, target_shape, strict=True):
        total_pad = max(0, target - current)
        before = total_pad // 2
        after = total_pad - before
        pad_width.append((before, after))
    return np.pad(fitted, pad_width, mode="constant", constant_values=0)


def _zscore_per_channel(data: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    normalized = data.astype(np.float32, copy=True)
    for channel_idx in range(normalized.shape[0]):
        channel = normalized[channel_idx]
        foreground_values = channel[foreground_mask]
        if foreground_values.size == 0:
            raise ValueError("Foreground mask is empty during z-score normalization")
        mean = float(foreground_values.mean())
        std = float(foreground_values.std())
        normalized[channel_idx, foreground_mask] = (foreground_values - mean) / max(
            std, 1e-8
        )
        normalized[channel_idx, ~foreground_mask] = 0.0
    return normalized


def _preprocess_case(
    case_id: str,
    case_file_paths: list[Path],
    dataset_json: dict,
    masks_tr_dir: Path,
    file_ending: str,
    plans: dict,
    output_dir: Path,
) -> None:
    mask_path = masks_tr_dir / f"{case_id}{file_ending}"
    mask_image = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing = [float(i) for i in mask_image.GetSpacing()[::-1]]
    channel_files = _get_case_channel_files(case_file_paths, dataset_json)
    channel_arrays = [
        sitk.GetArrayFromImage(sitk.ReadImage(str(channel_file)))
        for channel_file in channel_files
    ]
    image_data = np.stack(channel_arrays, axis=0)

    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    margin_mm = [float(i) for i in plans["margin_mm"]]

    for roi_name, roi_label in plans["roi_labels"].items():
        lower, upper = _compute_bbox(mask_array, roi_label, margin_mm, spacing)
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(lower, upper, strict=True)
        )

        cropped_channels = np.stack([channel[slices] for channel in image_data], axis=0)
        cropped_mask = (mask_array[slices] == roi_label).astype(np.uint8)
        cropped_channels = np.stack(
            [
                _resample_array(channel, spacing, target_spacing, is_mask=False)
                for channel in cropped_channels
            ],
            axis=0,
        )
        cropped_mask = _resample_array(
            cropped_mask,
            spacing,
            target_spacing,
            is_mask=True,
        )
        fitted_channels = np.stack(
            [
                _fit_to_target_shape(channel, target_shape)
                for channel in cropped_channels
            ],
            axis=0,
        )
        fitted_mask = _fit_to_target_shape(cropped_mask, target_shape) > 0
        normalized_channels = _zscore_per_channel(fitted_channels, fitted_mask)

        output_path = output_dir / f"{case_id}_{roi_name}.b2nd"
        blosc2.asarray(
            normalized_channels.astype(np.float32),
            urlpath=str(output_path),
            mode="w",
        )


@require_global_paths_set
def preprocess(d: int, num_workers: int = 4) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]

    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    preprocessed_dataset_dir = mm_preprocessed / dataset_dir.name
    plans = _load_mm_plans(preprocessed_dataset_dir / "mmPlans.json")
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)

    masks_tr_dir = dataset_dir / "masksTr"
    output_dir = preprocessed_dataset_dir / plans["output_folder_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_path = dataset_dir / "labelsTr.json"
    if labels_tr_path.is_file():
        shutil.copy2(labels_tr_path, preprocessed_dataset_dir / "labelsTr.json")

    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    margin_mm = [float(i) for i in plans["margin_mm"]]
    file_ending = dataset_json["file_ending"]
    _ = target_spacing, target_shape, margin_mm

    sorted_case_ids = sorted(case_files)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _preprocess_case,
                case_id,
                case_files[case_id],
                dataset_json,
                masks_tr_dir,
                file_ending,
                plans,
                output_dir,
            ): case_id
            for case_id in sorted_case_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing ROI crops",
        ):
            future.result()

    print(output_dir)
    return output_dir
