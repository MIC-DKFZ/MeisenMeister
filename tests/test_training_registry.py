from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister.training import BaseTrainer, get_trainer_class, get_trainer_registry
from meisenmeister.training.trainers.mm_trainer import mmTrainer


class TrainingRegistryTests(unittest.TestCase):
    def test_registry_finds_default_and_derived_trainers(self) -> None:
        registry = get_trainer_registry()

        self.assertIn("mmTrainer", registry)
        self.assertIn("mmTrainer_Debug", registry)

    def test_registry_ignores_non_trainer_classes(self) -> None:
        registry = get_trainer_registry()

        self.assertNotIn("NotATrainer", registry)

    def test_unknown_trainer_error_lists_available_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown trainer 'missing'.*mmTrainer.*mmTrainer_Debug",
        ):
            get_trainer_class("missing")

    def test_base_trainer_cannot_be_instantiated_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(TypeError):
                BaseTrainer("001", 0, root, root)

    def test_default_trainer_satisfies_base_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
                return_value={
                    "train": ["case_001_left", "case_001_right"],
                    "val": ["case_002_left", "case_002_right"],
                },
            ):
                trainer = mmTrainer("001", 0, root, root)

        self.assertIsInstance(trainer, BaseTrainer)


if __name__ == "__main__":
    unittest.main()
