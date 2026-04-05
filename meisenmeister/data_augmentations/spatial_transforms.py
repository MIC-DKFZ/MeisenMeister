from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FlipAxes3D:
    def __init__(self, probability: float, axes: Sequence[int]) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"FlipAxes3D probability must be between 0 and 1, got {probability!r}"
            )

        normalized_axes = tuple(int(axis) for axis in axes)
        invalid_axes = [axis for axis in normalized_axes if axis not in (0, 1, 2)]
        if invalid_axes:
            raise ValueError(
                f"FlipAxes3D axes must be chosen from (0, 1, 2), got {normalized_axes!r}"
            )

        self.probability = float(probability)
        self.axes = normalized_axes

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0 or not self.axes:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        flipped_image = np.flip(
            image, axis=tuple(axis + 1 for axis in self.axes)
        ).copy()
        return {**sample, "image": flipped_image}
