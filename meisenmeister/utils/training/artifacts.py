from __future__ import annotations

import logging
import math
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

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
        "da_preview_path": fold_dir / "da_preview.png",
    }


def ensure_portable_inference_metadata(
    *,
    dataset_dir: Path,
    preprocessed_dataset_dir: Path,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "dataset.json": dataset_dir / "dataset.json",
        "mmPlans.json": preprocessed_dataset_dir / "mmPlans.json",
    }
    for filename, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required inference metadata file not found: {source_path}"
            )
        destination_path = experiment_dir / filename
        if destination_path.exists():
            continue
        shutil.copy2(source_path, destination_path)


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


def save_da_preview(
    samples: list[dict],
    plot_path: Path,
    *,
    slice_titles: tuple[str, str, str] = ("axial", "coronal", "sagittal"),
) -> None:
    if not samples:
        raise ValueError("At least one sample is required to save a DA preview")

    first_image = np.asarray(_to_numpy_image(samples[0]["image"]))
    if first_image.ndim != 4:
        raise ValueError(
            "DA preview expects channel-first 4D images with shape (C, D, H, W)"
        )
    num_channels = int(first_image.shape[0])
    num_rows = len(samples) * num_channels
    figure, axes = plt.subplots(
        num_rows,
        3,
        figsize=(9, max(3, num_rows * 2)),
        squeeze=False,
    )

    row_index = 0
    for sample in samples:
        image = np.asarray(_to_numpy_image(sample["image"]), dtype=np.float32)
        if image.ndim != 4:
            raise ValueError(
                "DA preview expects channel-first 4D images with shape (C, D, H, W)"
            )
        if int(image.shape[0]) != num_channels:
            raise ValueError("All DA preview samples must have the same channel count")

        sample_id = str(sample.get("sample_id", f"sample_{row_index}"))
        for channel_index in range(num_channels):
            channel = image[channel_index]
            for column_index, slice_2d in enumerate(_extract_mid_slices(channel)):
                axis = axes[row_index, column_index]
                axis.imshow(slice_2d, cmap="gray")
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(slice_titles[column_index])
            axes[row_index, 0].set_ylabel(
                f"{sample_id}\nch {channel_index}",
                rotation=0,
                labelpad=32,
                va="center",
            )
            row_index += 1

    figure.tight_layout()
    figure.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def _to_numpy_image(image) -> np.ndarray:
    if hasattr(image, "detach"):
        return image.detach().cpu().numpy()
    return np.asarray(image)


def _extract_mid_slices(
    channel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if channel.ndim != 3:
        raise ValueError(
            f"DA preview expects 3D channel data with shape (D, H, W), got {channel.shape!r}"
        )
    depth_mid = channel.shape[0] // 2
    height_mid = channel.shape[1] // 2
    width_mid = channel.shape[2] // 2
    axial = channel[depth_mid, :, :]
    coronal = channel[:, height_mid, :]
    sagittal = channel[:, :, width_mid]
    return axial, coronal, sagittal
