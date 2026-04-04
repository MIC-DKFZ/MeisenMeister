from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
    print(message)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{message}\n")


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
