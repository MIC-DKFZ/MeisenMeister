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


class RandomShiftWithinMargin3D:
    def __init__(
        self,
        probability: float,
        max_shift_voxels: Sequence[int],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomShiftWithinMargin3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(max_shift_voxels) != 3:
            raise ValueError(
                "RandomShiftWithinMargin3D max_shift_voxels must contain exactly three values"
            )

        normalized_shifts = tuple(int(axis) for axis in max_shift_voxels)
        if any(axis < 0 for axis in normalized_shifts):
            raise ValueError(
                "RandomShiftWithinMargin3D max_shift_voxels values must be non-negative"
            )

        self.probability = float(probability)
        self.max_shift_voxels = normalized_shifts

    def _sample_shift(self) -> tuple[int, int, int]:
        sampled_shift = []
        for max_shift in self.max_shift_voxels:
            if max_shift == 0:
                sampled_shift.append(0)
                continue
            sampled_shift.append(int(np.random.randint(-max_shift, max_shift + 1)))
        return tuple(sampled_shift)

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        shifted_image = np.zeros_like(image)
        spatial_shape = image.shape[1:]
        sampled_shift = self._sample_shift()

        source_slices = []
        destination_slices = []
        for size, shift in zip(spatial_shape, sampled_shift, strict=True):
            src_start = max(0, -shift)
            src_stop = min(size, size - shift) if shift >= 0 else size
            dst_start = max(0, shift)
            dst_stop = dst_start + max(0, src_stop - src_start)
            source_slices.append(slice(src_start, src_stop))
            destination_slices.append(slice(dst_start, dst_stop))

        shifted_image[
            :,
            destination_slices[0],
            destination_slices[1],
            destination_slices[2],
        ] = image[
            :,
            source_slices[0],
            source_slices[1],
            source_slices[2],
        ]
        return {**sample, "image": shifted_image}
