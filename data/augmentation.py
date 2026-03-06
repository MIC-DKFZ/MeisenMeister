import math
import random

import torch
import torch.nn.functional as F


class Random3DAugmentation:

    def __init__(
        self,
        p_rotation=0.2,
        rotation_range_degrees=(-30.0, 30.0),
        p_scaling=0.2,
        scaling=(0.7, 1.4),
        p_noise=0.1,
        noise_variance=(0.0, 0.1),
        p_brightness=0.15,
        brightness_range=(0.75, 1.25),
        p_contrast=0.15,
        contrast_range=(0.75, 1.25),
        mirror_axes=(0, 1, 2),
    ):
        self.p_rotation = p_rotation
        self.rotation_range_degrees = rotation_range_degrees
        self.p_scaling = p_scaling
        self.scaling = scaling
        self.p_noise = p_noise
        self.noise_variance = noise_variance
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.mirror_axes = mirror_axes

    @staticmethod
    def _rand_uniform(lo, hi):
        return lo + (hi - lo) * random.random()

    @staticmethod
    def _rotation_matrix(ax, ay, az, device, dtype):
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        rx = torch.tensor(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=dtype
        )
        ry = torch.tensor(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=dtype
        )
        rz = torch.tensor(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=dtype
        )
        return rz @ ry @ rx

    def _spatial(self, x):
        apply_rot = random.random() < self.p_rotation
        apply_scale = random.random() < self.p_scaling
        if not apply_rot and not apply_scale:
            return x

        min_deg, max_deg = self.rotation_range_degrees
        if apply_rot:
            ax = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            ay = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            az = self._rand_uniform(min_deg, max_deg) * torch.pi / 180.0
            rot = self._rotation_matrix(ax, ay, az, x.device, x.dtype)
        else:
            rot = torch.eye(3, device=x.device, dtype=x.dtype)

        scale = (
            self._rand_uniform(self.scaling[0], self.scaling[1]) if apply_scale else 1.0
        )
        theta = torch.zeros((1, 3, 4), device=x.device, dtype=x.dtype)
        theta[0, :3, :3] = rot * scale

        x_b = x.unsqueeze(0)
        grid = F.affine_grid(theta, size=x_b.shape, align_corners=False)
        x_b = F.grid_sample(
            x_b,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return x_b.squeeze(0)

    def __call__(self, x):
        x = self._spatial(x)

        if self.mirror_axes is not None:
            for axis in self.mirror_axes:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[axis + 1])

        if random.random() < self.p_noise:
            noise_std = self._rand_uniform(
                self.noise_variance[0], self.noise_variance[1]
            )
            x = x + torch.randn_like(x) * noise_std

        if random.random() < self.p_brightness:
            brightness = self._rand_uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            x = x * brightness

        if random.random() < self.p_contrast:
            contrast = self._rand_uniform(
                self.contrast_range[0], self.contrast_range[1]
            )
            mean = x.mean()
            x = (x - mean) * contrast + mean

        return x
