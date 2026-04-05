from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MultiplicativeBrightness3D:
    def __init__(
        self,
        probability: float,
        multiplier_range: Sequence[float] = (0.5, 2.0),
        p_per_channel: float = 1.0,
        synchronize_channels: bool = False,
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"MultiplicativeBrightness3D probability must be between 0 and 1, got {probability!r}"
            )
        if not 0.0 <= float(p_per_channel) <= 1.0:
            raise ValueError(
                f"MultiplicativeBrightness3D p_per_channel must be between 0 and 1, got {p_per_channel!r}"
            )
        if len(multiplier_range) != 2:
            raise ValueError(
                "MultiplicativeBrightness3D multiplier_range must contain exactly two values"
            )

        multiplier_min = float(multiplier_range[0])
        multiplier_max = float(multiplier_range[1])
        if multiplier_min > multiplier_max:
            raise ValueError(
                "MultiplicativeBrightness3D multiplier_range must be ordered as (min, max)"
            )

        self.probability = float(probability)
        self.multiplier_range = (multiplier_min, multiplier_max)
        self.p_per_channel = float(p_per_channel)
        self.synchronize_channels = bool(synchronize_channels)

    def _sample_multiplier(self) -> float:
        low, high = self.multiplier_range
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        brightened_image = image.copy()

        synchronized_multiplier = (
            self._sample_multiplier() if self.synchronize_channels else None
        )
        for channel_index in range(brightened_image.shape[0]):
            if np.random.random() >= self.p_per_channel:
                continue

            multiplier = (
                synchronized_multiplier
                if synchronized_multiplier is not None
                else self._sample_multiplier()
            )
            brightened_image[channel_index] = (
                brightened_image[channel_index] * multiplier
            )

        return {**sample, "image": brightened_image.astype(image.dtype, copy=False)}
