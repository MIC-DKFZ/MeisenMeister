from __future__ import annotations

from meisenmeister.architectures import get_architecture_class
from meisenmeister.training.trainers.mm_trainer import mmTrainer


class mmTrainer_NNUNetEncoder(mmTrainer):
    ARCHITECTURE_NAME = "ResidualEncoderClsNetwork"

    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir,
        preprocessed_dataset_dir,
        results_dir,
        architecture_name: str | None = None,
        continue_training: bool = False,
        weights_path=None,
        experiment_postfix: str | None = None,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            fold=fold,
            dataset_dir=dataset_dir,
            preprocessed_dataset_dir=preprocessed_dataset_dir,
            results_dir=results_dir,
            architecture_name=architecture_name or self.ARCHITECTURE_NAME,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
        )

    def fit(self) -> None:
        self._validate_target_shape()
        super().fit()

    def _validate_target_shape(self) -> None:
        plans = self.get_preprocessing_plans()
        target_shape = plans.get("target_shape")
        if target_shape is None:
            raise ValueError(
                "mmPlans.json must define target_shape for mmTrainer_NNUNetEncoder"
            )

        architecture_class = get_architecture_class(self.architecture_name)
        required_divisibility = getattr(
            architecture_class,
            "ENCODER_INPUT_DIVISIBILITY",
            None,
        )
        if required_divisibility is None:
            return

        resolved_target_shape = [int(axis) for axis in target_shape]
        incompatible_axes = [
            axis
            for axis, divisor in zip(
                resolved_target_shape,
                required_divisibility,
                strict=True,
            )
            if axis % int(divisor) != 0
        ]
        if incompatible_axes:
            raise ValueError(
                f"{self.architecture_name} requires target_shape divisible by "
                f"{list(required_divisibility)}, got {resolved_target_shape}"
            )
