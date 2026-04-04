from __future__ import annotations

from meisenmeister.training.trainers.mm_trainer import mmTrainer


class mmTrainer_Debug(mmTrainer):
    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir,
        preprocessed_dataset_dir,
        architecture_name: str = "ResNet3D18",
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            fold=fold,
            dataset_dir=dataset_dir,
            preprocessed_dataset_dir=preprocessed_dataset_dir,
            architecture_name=architecture_name,
            num_epochs=2,
            batch_size=1,
            num_workers=0,
            shuffle=True,
        )


class NotATrainer:
    pass
