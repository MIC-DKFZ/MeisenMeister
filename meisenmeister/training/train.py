from meisenmeister.training.registry import get_trainer_class
from meisenmeister.training.splits import get_fold_sample_ids
from meisenmeister.utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)


@require_global_paths_set
def train(
    d: int,
    fold: int,
    trainer_name: str = "mmTrainer",
    architecture_name: str = "ResNet3D18",
    continue_training: bool = False,
) -> None:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if fold < 0:
        raise ValueError(f"Fold must be non-negative, got {fold}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]
    mm_results = paths["mm_results"]

    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    preprocessed_dataset_dir = mm_preprocessed / dataset_dir.name

    get_fold_sample_ids(preprocessed_dataset_dir, fold)
    trainer_class = get_trainer_class(trainer_name)
    trainer = trainer_class(
        dataset_id=dataset_id,
        fold=fold,
        dataset_dir=dataset_dir,
        preprocessed_dataset_dir=preprocessed_dataset_dir,
        results_dir=mm_results,
        architecture_name=architecture_name,
        continue_training=continue_training,
    )
    trainer.fit()
