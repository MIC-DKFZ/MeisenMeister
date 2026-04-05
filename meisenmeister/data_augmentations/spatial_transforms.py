from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk


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


def _resample_channel_with_scale(channel: np.ndarray, scale: float) -> np.ndarray:
    depth, height, width = channel.shape
    center = (np.asarray(channel.shape, dtype=np.float32) - 1.0) / 2.0

    z = np.arange(depth, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    src_z = center[0] + (zz - center[0]) / scale
    src_y = center[1] + (yy - center[1]) / scale
    src_x = center[2] + (xx - center[2]) / scale

    valid = (
        (src_z >= 0.0)
        & (src_z <= depth - 1)
        & (src_y >= 0.0)
        & (src_y <= height - 1)
        & (src_x >= 0.0)
        & (src_x <= width - 1)
    )

    z0 = np.floor(src_z).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    z1 = np.clip(z0 + 1, 0, depth - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)

    wz = src_z - z0
    wy = src_y - y0
    wx = src_x - x0

    c000 = channel[z0, y0, x0]
    c001 = channel[z0, y0, x1]
    c010 = channel[z0, y1, x0]
    c011 = channel[z0, y1, x1]
    c100 = channel[z1, y0, x0]
    c101 = channel[z1, y0, x1]
    c110 = channel[z1, y1, x0]
    c111 = channel[z1, y1, x1]

    output = (
        c000 * (1.0 - wz) * (1.0 - wy) * (1.0 - wx)
        + c001 * (1.0 - wz) * (1.0 - wy) * wx
        + c010 * (1.0 - wz) * wy * (1.0 - wx)
        + c011 * (1.0 - wz) * wy * wx
        + c100 * wz * (1.0 - wy) * (1.0 - wx)
        + c101 * wz * (1.0 - wy) * wx
        + c110 * wz * wy * (1.0 - wx)
        + c111 * wz * wy * wx
    ).astype(channel.dtype, copy=False)
    output[~valid] = 0.0
    return output


def _rotate_channel_with_angles(
    channel: np.ndarray,
    angles_degrees: tuple[float, float, float],
) -> np.ndarray:
    rotation_z, rotation_y, rotation_x = angles_degrees
    image = sitk.GetImageFromArray(channel.astype(np.float32, copy=False))
    center_index = [(axis_size - 1) / 2.0 for axis_size in image.GetSize()]
    center_point = image.TransformContinuousIndexToPhysicalPoint(center_index)

    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_point)
    # Public augmentation parameters use NumPy spatial axis order (z, y, x).
    # SimpleITK Euler3DTransform expects rotations around physical axes (x, y, z).
    transform.SetRotation(
        float(np.deg2rad(rotation_x)),
        float(np.deg2rad(rotation_y)),
        float(np.deg2rad(rotation_z)),
    )

    rotated = sitk.Resample(
        image,
        image,
        transform,
        sitk.sitkLinear,
        0.0,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(rotated).astype(channel.dtype, copy=False)


class RandomScaling3D:
    def __init__(
        self,
        probability: float,
        scaling: Sequence[float],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomScaling3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(scaling) != 2:
            raise ValueError("RandomScaling3D scaling must contain exactly two values")

        scaling_min = float(scaling[0])
        scaling_max = float(scaling[1])
        if scaling_min <= 0.0 or scaling_max <= 0.0:
            raise ValueError("RandomScaling3D scaling values must be positive")
        if scaling_min > scaling_max:
            raise ValueError("RandomScaling3D scaling must be ordered as (min, max)")

        self.probability = float(probability)
        self.scaling = (scaling_min, scaling_max)

    def _sample_scale(self) -> float:
        low, high = self.scaling
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        scale = self._sample_scale()
        scaled_image = np.stack(
            [_resample_channel_with_scale(channel, scale) for channel in image],
            axis=0,
        )
        return {**sample, "image": scaled_image.astype(image.dtype, copy=False)}


class RandomRotation3D:
    def __init__(
        self,
        probability: float,
        max_rotation_degrees: Sequence[float],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomRotation3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(max_rotation_degrees) != 3:
            raise ValueError(
                "RandomRotation3D max_rotation_degrees must contain exactly three values"
            )

        normalized_angles = tuple(float(angle) for angle in max_rotation_degrees)
        if any(angle < 0.0 for angle in normalized_angles):
            raise ValueError(
                "RandomRotation3D max_rotation_degrees values must be non-negative"
            )

        self.probability = float(probability)
        self.max_rotation_degrees = normalized_angles

    def _sample_angles(self) -> tuple[float, float, float]:
        sampled_angles = []
        for max_angle in self.max_rotation_degrees:
            if max_angle == 0.0:
                sampled_angles.append(0.0)
                continue
            sampled_angles.append(float(np.random.uniform(-max_angle, max_angle)))
        return tuple(sampled_angles)

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        sampled_angles = self._sample_angles()
        rotated_image = np.stack(
            [_rotate_channel_with_angles(channel, sampled_angles) for channel in image],
            axis=0,
        )
        return {**sample, "image": rotated_image.astype(image.dtype, copy=False)}
