from pathlib import Path

from breastdivider import BreastDividerPredictor

from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def get_breast_segmentation_predictor():
    return BreastDividerPredictor(device="cuda")


def predict_breast_segmentation(predictor, input_path, output_path):
    predictor.predict(
        input_path=input_path,
        output_path=output_path,
    )


def create_breast_segmentations(d: int) -> tuple[Path, Path]:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    mm_raw = verify_required_global_paths_set()["mm_raw"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    images_tr_dir = dataset_dir / "imagesTr"
    masks_tr_dir = dataset_dir / "masksTr"

    if not images_tr_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory in {dataset_dir}")

    masks_tr_dir.mkdir(parents=True, exist_ok=True)
    file_ending = dataset_json["file_ending"]
    input_suffix = f"_0000{file_ending}"
    predictor = get_breast_segmentation_predictor()

    for case_id, files in sorted(case_files.items()):
        input_file = next(
            (path for path in files if path.name.endswith(input_suffix)),
            None,
        )
        if input_file is None:
            raise FileNotFoundError(
                f"Missing input file ending with '{input_suffix}' for case {case_id}"
            )

        output_file = masks_tr_dir / f"{case_id}{file_ending}"
        predict_breast_segmentation(
            predictor=predictor,
            input_path=str(input_file),
            output_path=str(output_file),
        )

    return images_tr_dir, masks_tr_dir
