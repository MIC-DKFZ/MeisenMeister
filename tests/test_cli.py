from __future__ import annotations

import unittest
from unittest.mock import patch

from meisenmeister import cli


class CliTests(unittest.TestCase):
    def test_mm_train_uses_default_trainer(self) -> None:
        with (
            patch("sys.argv", ["mm_train", "-d", "1"]),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(1, trainer_name="mmTrainer")

    def test_mm_train_passes_explicit_trainer(self) -> None:
        with (
            patch("sys.argv", ["mm_train", "-d", "1", "--trainer", "mmTrainer_Debug"]),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(1, trainer_name="mmTrainer_Debug")


if __name__ == "__main__":
    unittest.main()
