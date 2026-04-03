from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_training_files_present,
    verify_required_global_paths_set,
)


@require_global_paths_set
def extract_dataset_fingerprint(d: int):
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    mm_raw = verify_required_global_paths_set()["mm_raw"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    print(dataset_dir)
    print(dataset_json)
    print(list(case_files.keys()))
