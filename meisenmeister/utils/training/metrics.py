from __future__ import annotations

import math

import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def create_empty_history() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_auc": [],
        "ema_val_balanced_accuracy": [],
        "lr": [],
        "epoch_time_sec": [],
    }


def aggregate_epoch_metrics(metrics: list[dict]) -> tuple[float, float]:
    total_samples = sum(metric["num_samples"] for metric in metrics)
    if total_samples == 0:
        raise ValueError("Cannot aggregate metrics for an empty epoch")

    total_loss = sum(metric["loss"] * metric["num_samples"] for metric in metrics)
    total_correct = sum(metric["num_correct"] for metric in metrics)
    return total_loss / total_samples, total_correct / total_samples


def aggregate_validation_classification_metrics(
    metrics: list[dict],
) -> tuple[float, float, bool]:
    labels = torch.cat([metric["labels"] for metric in metrics], dim=0).numpy()
    predictions = torch.cat(
        [metric["predictions"] for metric in metrics], dim=0
    ).numpy()
    probabilities = torch.cat(
        [metric["probabilities"] for metric in metrics],
        dim=0,
    ).numpy()
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    macro_auc = math.nan
    macro_auc_defined = True
    try:
        macro_auc = float(
            roc_auc_score(
                labels,
                probabilities,
                average="macro",
                multi_class="ovr",
                labels=list(range(probabilities.shape[1])),
            )
        )
    except ValueError:
        macro_auc_defined = False
    if math.isnan(macro_auc):
        macro_auc_defined = False
    return balanced_accuracy, macro_auc, macro_auc_defined


def compute_ema(
    history: list[float],
    current_value: float,
    *,
    alpha: float,
) -> float:
    if not history:
        return current_value
    previous_value = history[-1]
    return alpha * current_value + (1 - alpha) * previous_value


def append_history(
    history: dict[str, list[float]],
    *,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    val_balanced_accuracy: float,
    val_macro_auc: float,
    ema_val_balanced_accuracy: float,
    lr: float,
    epoch_time_sec: float,
) -> None:
    history["epoch"].append(int(epoch))
    history["train_loss"].append(float(train_loss))
    history["train_accuracy"].append(float(train_accuracy))
    history["val_loss"].append(float(val_loss))
    history["val_accuracy"].append(float(val_accuracy))
    history["val_balanced_accuracy"].append(float(val_balanced_accuracy))
    history["val_macro_auc"].append(float(val_macro_auc))
    history["ema_val_balanced_accuracy"].append(float(ema_val_balanced_accuracy))
    history["lr"].append(float(lr))
    history["epoch_time_sec"].append(float(epoch_time_sec))


def should_update_best(
    best_state: dict[str, float | int | None],
    ema_val_balanced_accuracy: float,
    val_loss: float,
    *,
    tolerance: float,
) -> bool:
    best_ema = best_state["ema_val_balanced_accuracy"]
    best_val_loss = best_state["val_loss"]
    if best_ema is None:
        return True
    if ema_val_balanced_accuracy > best_ema + tolerance:
        return True
    if abs(ema_val_balanced_accuracy - best_ema) <= tolerance:
        return best_val_loss is None or val_loss < best_val_loss
    return False
