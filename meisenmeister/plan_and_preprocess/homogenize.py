from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def _resample_to_reference(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_image)


@require_global_paths_set
def homogenize(d: int) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    mm_raw = verify_required_global_paths_set()["mm_raw"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    file_ending = dataset_json["file_ending"]
    reference_suffix = f"_0000{file_ending}"

    for case_id, files in tqdm(sorted(case_files.items()), desc="Homogenizing cases"):
        reference_path = next(
            (path for path in files if path.name.endswith(reference_suffix)),
            None,
        )
        if reference_path is None:
            raise FileNotFoundError(
                f"Missing reference image ending with '{reference_suffix}' for case {case_id}"
            )

        reference_image = sitk.ReadImage(str(reference_path))
        for path in files:
            if path == reference_path:
                continue
            moving_image = sitk.ReadImage(str(path))
            resampled_image = _resample_to_reference(moving_image, reference_image)
            sitk.WriteImage(resampled_image, str(path))

    print(dataset_dir / "imagesTr")
    return dataset_dir / "imagesTr"
