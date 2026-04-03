import json
from pathlib import Path

REQUIRED_DATASET_JSON_KEYS = (
    "channel_names",
    "file_ending",
    "problem_type",
    "labels",
    "numTraining",
)


def load_dataset_json(dataset_dir: Path) -> dict:
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Missing dataset.json in {dataset_dir}")

    with dataset_json_path.open("r", encoding="utf-8") as file:
        dataset_json = json.load(file)

    missing_keys = [
        key for key in REQUIRED_DATASET_JSON_KEYS if key not in dataset_json
    ]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(
            f"dataset.json in {dataset_dir} is missing required keys: {missing_keys_str}"
        )

    return dataset_json


def verify_training_files_present(dataset_dir: Path, dataset_json: dict) -> dict[str, list[Path]]:
    images_tr_dir = dataset_dir / "imagesTr"
    if not images_tr_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory in {dataset_dir}")

    channel_names = dataset_json["channel_names"]
    file_ending = dataset_json["file_ending"]
    num_training = dataset_json["numTraining"]

    print(
        f"Verifying training files in {images_tr_dir} for {num_training} cases "
        f"with channels {channel_names} and file ending '{file_ending}'"
    )

    channel_ids = sorted(int(channel_id) for channel_id in channel_names)
    expected_suffixes = [f"_{channel_id:04d}{file_ending}" for channel_id in channel_ids]
    expected_suffix_set = set(expected_suffixes)

    case_files: dict[str, list[Path]] = {}
    unexpected_files: list[str] = []
    for path in sorted(images_tr_dir.iterdir()):
        if not path.is_file():
            continue

        matched = False
        for suffix in expected_suffixes:
            if path.name.endswith(suffix):
                case_id = path.name[: -len(suffix)]
                if case_id:
                    case_files.setdefault(case_id, []).append(path)
                    matched = True
                break
        if not matched:
            unexpected_files.append(path.name)

    found_case_ids = sorted(case_files)
    print(f"Found case ids in imagesTr: {found_case_ids}")

    if unexpected_files:
        unexpected_files_str = ", ".join(unexpected_files)
        raise ValueError(
            f"Found unexpected files in {images_tr_dir}: {unexpected_files_str}"
        )

    if len(case_files) != num_training:
        raise ValueError(
            f"Expected {num_training} training cases in {images_tr_dir}, found "
            f"{len(case_files)}: {found_case_ids}"
        )

    expected_channel_count = len(channel_ids)
    invalid_cases: list[str] = []
    for case_id, files in case_files.items():
        present_suffixes = sorted(path.name[len(case_id) :] for path in files)
        present_suffix_set = set(present_suffixes)
        missing_suffixes = sorted(
            suffix for suffix in expected_suffixes if suffix not in present_suffix_set
        )
        if len(files) != expected_channel_count:
            invalid_cases.append(
                f"{case_id}: found {len(files)} channel files, expected "
                f"{expected_channel_count}; present={present_suffixes}"
            )
            continue

        extra_suffixes = sorted(
            suffix for suffix in present_suffix_set if suffix not in expected_suffix_set
        )
        if missing_suffixes or extra_suffixes:
            parts: list[str] = [f"{case_id}: present={present_suffixes}"]
            if missing_suffixes:
                parts.append(f"missing={missing_suffixes}")
            if extra_suffixes:
                parts.append(f"unexpected={extra_suffixes}")
            invalid_cases.append("; ".join(parts))

    if invalid_cases:
        raise ValueError(
            "Training files are incomplete or inconsistent in "
            f"{images_tr_dir}: {'; '.join(invalid_cases)}"
        )

    return case_files
