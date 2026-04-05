from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_LOGGER_CACHE: dict[Path, logging.Logger] = {}


def build_experiment_paths(
    *,
    results_dir: Path,
    dataset_name: str,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    fold: int,
) -> dict[str, Path]:
    experiment_name = f"{trainer_name}_{architecture_name}"
    if experiment_postfix:
        experiment_name = f"{experiment_name}_{experiment_postfix}"
    experiment_dir = results_dir / dataset_name / experiment_name
    fold_dir = experiment_dir / f"fold_{fold}"
    return {
        "experiment_dir": experiment_dir,
        "fold_dir": fold_dir,
        "log_path": fold_dir / "train.log",
        "last_checkpoint_path": fold_dir / "model_last.pt",
        "best_checkpoint_path": fold_dir / "model_best.pt",
        "eval_last_path": fold_dir / "eval_last.json",
        "eval_best_path": fold_dir / "eval_best.json",
        "plot_path": fold_dir / "training_curves.png",
    }


def prepare_output_dir(
    *,
    fold_dir: Path,
    log_path: Path,
    continue_training: bool,
) -> str | None:
    fold_dir_exists = fold_dir.exists()
    fold_dir.mkdir(parents=True, exist_ok=True)
    if not continue_training:
        log_path.write_text("", encoding="utf-8")
    if fold_dir_exists and not continue_training:
        return f"WARNING: YOU ARE OVERWRITING EXISTING TRAINING OUTPUT IN {fold_dir}"
    return None


def log_message(message: str, log_path: Path) -> None:
    logger = _get_training_logger(log_path)
    if message.startswith("WARNING:"):
        logger.warning(message)
        return
    logger.info(message)


def _get_training_logger(log_path: Path) -> logging.Logger:
    resolved_path = log_path.resolve()
    logger = _LOGGER_CACHE.get(resolved_path)
    if logger is None:
        logger_name = "meisenmeister.training." + str(resolved_path).replace(
            "/", "."
        ).replace(":", "_")
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _LOGGER_CACHE[resolved_path] = logger

    needs_reconfigure = True
    if len(logger.handlers) == 2:
        stream_handler, file_handler = logger.handlers
        stream_matches = (
            isinstance(stream_handler, logging.StreamHandler)
            and getattr(stream_handler, "stream", None) is sys.stdout
        )
        file_matches = (
            isinstance(file_handler, logging.FileHandler)
            and Path(file_handler.baseFilename) == resolved_path
        )
        needs_reconfigure = not (stream_matches and file_matches)

    if needs_reconfigure:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def save_training_curves(history: dict[str, list[float]], plot_path: Path) -> None:
    epochs = history["epoch"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].plot(
        epochs,
        history["val_balanced_accuracy"],
        label="val_balanced_accuracy",
    )
    axes[0].plot(epochs, history["val_macro_auc"], label="val_macro_auc")
    axes[0].plot(
        epochs,
        history["ema_val_balanced_accuracy"],
        label="ema_val_balanced_accuracy",
    )
    axes[0].set_ylabel("metrics")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["epoch_time_sec"], label="epoch_time_sec")
    axes[1].set_ylabel("seconds")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["lr"], label="lr")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
