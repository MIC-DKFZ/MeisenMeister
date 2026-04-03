import torch
from torch.utils.data import DataLoader

from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)


@require_global_paths_set
def train(d: int) -> None:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]

    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    preprocessed_dataset_dir = mm_preprocessed / dataset_dir.name

    dataset = MeisenmeisterROIDataset(preprocessed_dataset_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset: {dataset_dir.name}")
    print(f"Samples: {len(dataset)}")

    for batch_idx, batch in enumerate(dataloader, start=1):
        print(
            f"Batch {batch_idx}: image_shape={tuple(batch['image'].shape)}, "
            f"sample_ids={list(batch['sample_id'])}"
        )

    print("DONE")
