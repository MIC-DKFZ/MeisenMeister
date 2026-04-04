from __future__ import annotations

import unittest

import torch

from meisenmeister.architectures import (
    BaseArchitecture,
    ResNet3D18,
    get_architecture_class,
    get_architecture_registry,
)


class ArchitectureTests(unittest.TestCase):
    def test_architecture_registry_finds_resnet3d18(self) -> None:
        registry = get_architecture_registry()

        self.assertIn("ResNet3D18", registry)

    def test_unknown_architecture_error_lists_available_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown architecture 'missing'.*ResNet3D18",
        ):
            get_architecture_class("missing")

    def test_resnet3d18_satisfies_base_interface(self) -> None:
        architecture = ResNet3D18(in_channels=3, num_classes=2)

        self.assertIsInstance(architecture, BaseArchitecture)

    def test_resnet3d18_forward_shape(self) -> None:
        architecture = ResNet3D18(in_channels=3, num_classes=4)
        x = torch.randn(2, 3, 32, 32, 32)

        y = architecture(x)

        self.assertEqual(tuple(y.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
