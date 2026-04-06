from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from meisenmeister.architectures import get_architecture_class
from meisenmeister.plan_and_preprocess.create_breast_seg import (
    get_breast_segmentation_predictor,
    predict_breast_segmentation,
)
from meisenmeister.plan_and_preprocess.preprocessing_utils import (
    get_case_channel_files,
    load_case_image_data,
    load_mm_plans,
    preprocess_roi_array,
)
from meisenmeister.utils import (
    build_experiment_paths,
    discover_case_files,
    find_dataset_dir,
    load_dataset_json,
    maybe_compile_model,
    require_global_paths_set,
    verify_required_global_paths_set,
)


def _resolve_trainer_architecture_name(trainer_name: str) -> str:
    from meisenmeister.training.registry import get_trainer_class

    trainer_class = get_trainer_class(trainer_name)
    return getattr(trainer_class, "ARCHITECTURE_NAME", "ResNet3D18")


def _validate_folds(folds: list[int]) -> list[int]:
    if not folds:
        raise ValueError("At least one fold must be provided")
    if any(fold < 0 for fold in folds):
        raise ValueError(f"All folds must be non-negative, got {folds}")
    unique_folds = sorted(set(int(fold) for fold in folds))
    if len(unique_folds) != len(folds):
        raise ValueError(f"Folds must be unique, got {folds}")
    return unique_folds


def _get_flip_axes(use_tta: bool) -> list[tuple[int, ...]]:
    if not use_tta:
        return [()]
    spatial_axes = (1, 2, 3)
    variants = [()]
    for subset_size in range(1, len(spatial_axes) + 1):
        variants.extend(combinations(spatial_axes, subset_size))
    return variants


def _average_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("Cannot average empty probability list")
    stacked = np.stack(probabilities, axis=0).astype(np.float64, copy=False)
    return stacked.mean(axis=0).astype(np.float32, copy=False)


def _predict_roi_with_tta(
    model: torch.nn.Module,
    roi_tensor: torch.Tensor,
    *,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    probability_vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for axes in _get_flip_axes(use_tta):
            augmented = torch.flip(roi_tensor, dims=axes) if axes else roi_tensor
            logits = model(
                augmented.unsqueeze(0).to(
                    device, dtype=torch.float32, non_blocking=True
                )
            )
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            probability_vectors.append(probabilities)
    return _average_probabilities(probability_vectors)


def _save_binary_mask(
    mask_image: sitk.Image, binary_mask: np.ndarray, output_path: Path
) -> None:
    binary_image = sitk.GetImageFromArray(binary_mask.astype(np.uint8, copy=False))
    binary_image.CopyInformation(mask_image)
    sitk.WriteImage(binary_image, str(output_path))


def _prepare_case_prediction_inputs(
    *,
    case_id: str,
    case_files: list[Path],
    dataset_json: dict,
    plans: dict,
    predictor,
    output_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    ordered_case_files = get_case_channel_files(case_files, dataset_json)
    file_ending = dataset_json["file_ending"]
    primary_input_path = ordered_case_files[0]
    breast_mask_path = output_dir / f"{case_id}_breast_mask{file_ending}"
    predict_breast_segmentation(
        predictor=predictor,
        input_path=str(primary_input_path),
        output_path=str(breast_mask_path),
    )

    mask_image = sitk.ReadImage(str(breast_mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing = [float(i) for i in mask_image.GetSpacing()[::-1]]
    image_data = load_case_image_data(case_files, dataset_json)
    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    zero_margin = [0.0, 0.0, 0.0]

    roi_tensors: dict[str, torch.Tensor] = {}
    artifact_paths = {"breast_mask": str(breast_mask_path)}
    for roi_name, roi_label in plans["roi_labels"].items():
        if not np.any(mask_array == roi_label):
            raise ValueError(
                f"ROI label {roi_label} ({roi_name}) not present for {case_id}"
            )
        roi_output_path = output_dir / f"{case_id}_{roi_name}_mask{file_ending}"
        _save_binary_mask(mask_image, mask_array == roi_label, roi_output_path)
        artifact_paths[f"{roi_name}_mask"] = str(roi_output_path)
        roi_array = preprocess_roi_array(
            image_data,
            mask_array,
            spacing,
            roi_label=int(roi_label),
            target_spacing=target_spacing,
            target_shape=target_shape,
            margin_mm=zero_margin,
        )
        roi_tensors[roi_name] = torch.from_numpy(roi_array)

    return roi_tensors, artifact_paths


def _resolve_checkpoint_path(*, fold_dir: Path, checkpoint: str) -> Path:
    return fold_dir / ("model_best.pt" if checkpoint == "best" else "model_last.pt")


def _load_checkpoint_payload(checkpoint_path: Path) -> dict:
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def _load_model_from_checkpoint_payload(
    *,
    checkpoint_payload: dict,
    architecture_name: str,
    device: torch.device,
    compile_model: bool,
) -> tuple[torch.nn.Module, bool, str]:
    trainer_config = checkpoint_payload.get("trainer_config", {})
    in_channels = trainer_config.get("in_channels")
    num_classes = trainer_config.get("num_classes")
    if in_channels is None or num_classes is None:
        raise ValueError(
            "Checkpoint is missing architecture metadata required for portable inference"
        )
    architecture_class = get_architecture_class(architecture_name)
    model = architecture_class(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    model, compile_applied, compile_status_message = maybe_compile_model(
        model,
        device=device,
        enabled=compile_model,
    )
    return model, compile_applied, compile_status_message


def _load_fold_predictors_from_experiment_dir(
    *,
    experiment_dir: Path,
    architecture_name: str,
    folds: list[int],
    checkpoint: str,
    compile_model: bool,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = []
    for fold in folds:
        fold_dir = experiment_dir / f"fold_{fold}"
        checkpoint_path = _resolve_checkpoint_path(
            fold_dir=fold_dir,
            checkpoint=checkpoint,
        )
        checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
        model, compile_applied, compile_status_message = (
            _load_model_from_checkpoint_payload(
                checkpoint_payload=checkpoint_payload,
                architecture_name=architecture_name,
                device=device,
                compile_model=compile_model,
            )
        )
        predictors.append(
            {
                "fold": fold,
                "device": device,
                "checkpoint_path": str(checkpoint_path),
                "compile_applied": compile_applied,
                "compile_status_message": compile_status_message,
                "model": model,
            }
        )
    return predictors


def _load_fold_predictors(
    *,
    dataset_id: str,
    dataset_dir: Path,
    preprocessed_dataset_dir: Path,
    results_dir: Path,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int],
    checkpoint: str,
    compile_model: bool,
) -> list[dict]:
    experiment_dir = build_experiment_paths(
        results_dir=results_dir,
        dataset_name=dataset_dir.name,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        fold=folds[0],
    )["experiment_dir"]
    return _load_fold_predictors_from_experiment_dir(
        experiment_dir=experiment_dir,
        architecture_name=architecture_name,
        folds=folds,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )


def _build_prediction_payload(
    *,
    dataset_id: str | None,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int],
    checkpoint: str,
    use_tta: bool,
    compile_enabled: bool,
) -> dict:
    return {
        "config": {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "trainer_name": trainer_name,
            "architecture_name": architecture_name,
            "experiment_postfix": experiment_postfix,
            "folds": folds,
            "checkpoint": checkpoint,
            "tta_enabled": bool(use_tta),
            "compile_enabled": bool(compile_enabled),
        },
        "cases": {},
    }


def _run_prediction(
    *,
    dataset_id: str | None,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    dataset_json: dict,
    plans: dict,
    fold_predictors: list[dict],
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int],
    checkpoint: str,
    use_tta: bool,
    compile_model: bool,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    case_files_by_case_id = discover_case_files(input_path, dataset_json)
    predictor = get_breast_segmentation_predictor()
    payload = _build_prediction_payload(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        input_path=input_path,
        output_path=output_path,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=folds,
        checkpoint=checkpoint,
        use_tta=use_tta,
        compile_enabled=all(
            fold_predictor.get("compile_applied", False)
            for fold_predictor in fold_predictors
        ),
    )

    for case_id, case_files in sorted(case_files_by_case_id.items()):
        roi_tensors, artifact_paths = _prepare_case_prediction_inputs(
            case_id=case_id,
            case_files=case_files,
            dataset_json=dataset_json,
            plans=plans,
            predictor=predictor,
            output_dir=output_path,
        )
        roi_results = {}
        for roi_name, roi_tensor in roi_tensors.items():
            per_fold = {}
            fold_probabilities: list[np.ndarray] = []
            for fold_predictor in fold_predictors:
                probabilities = _predict_roi_with_tta(
                    fold_predictor["model"],
                    roi_tensor,
                    device=fold_predictor["device"],
                    use_tta=use_tta,
                )
                per_fold[str(fold_predictor["fold"])] = {
                    "checkpoint_path": fold_predictor["checkpoint_path"],
                    "probabilities": [float(value) for value in probabilities.tolist()],
                    "prediction": int(np.argmax(probabilities)),
                }
                fold_probabilities.append(probabilities)
            final_probabilities = _average_probabilities(fold_probabilities)
            roi_results[roi_name] = {
                "per_fold": per_fold,
                "probabilities": [
                    float(value) for value in final_probabilities.tolist()
                ],
                "prediction": int(np.argmax(final_probabilities)),
            }

        payload["cases"][case_id] = {
            "artifacts": artifact_paths,
            "rois": roi_results,
        }

    predictions_path = output_path / "predictions.json"
    predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return predictions_path


def _get_experiment_metadata(
    experiment_dir: Path, folds: list[int], checkpoint: str
) -> dict:
    checkpoint_path = _resolve_checkpoint_path(
        fold_dir=experiment_dir / f"fold_{folds[0]}",
        checkpoint=checkpoint,
    )
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    trainer_config = checkpoint_payload.get("trainer_config", {})
    dataset_name = trainer_config.get("dataset_name", experiment_dir.parent.name)
    trainer_name = trainer_config.get("trainer_name")
    architecture_name = trainer_config.get("architecture_name")
    if trainer_name is None or architecture_name is None:
        raise ValueError(
            "Checkpoint is missing trainer_name or architecture_name required for portable inference"
        )
    return {
        "dataset_id": trainer_config.get("dataset_id"),
        "dataset_name": dataset_name,
        "trainer_name": trainer_name,
        "architecture_name": architecture_name,
        "experiment_postfix": trainer_config.get("experiment_postfix"),
    }


@require_global_paths_set
def predict(
    d: int,
    input_dir: str,
    output_dir: str,
    folds: list[int],
    trainer_name: str = "mmTrainer",
    experiment_postfix: str | None = None,
    checkpoint: str = "best",
    use_tta: bool = True,
    compile_model: bool = True,
) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if checkpoint not in {"best", "last"}:
        raise ValueError(
            f"checkpoint must be one of ('best', 'last'), got {checkpoint!r}"
        )

    fold_values = _validate_folds(folds)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    dataset_dir = find_dataset_dir(paths["mm_raw"], dataset_id)
    preprocessed_dataset_dir = paths["mm_preprocessed"] / dataset_dir.name
    results_dir = paths["mm_results"]
    architecture_name = _resolve_trainer_architecture_name(trainer_name)
    dataset_json = load_dataset_json(dataset_dir)
    plans = load_mm_plans(preprocessed_dataset_dir / "mmPlans.json")
    fold_predictors = _load_fold_predictors(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        preprocessed_dataset_dir=preprocessed_dataset_dir,
        results_dir=results_dir,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=fold_values,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )
    return _run_prediction(
        dataset_id=dataset_id,
        dataset_name=dataset_dir.name,
        input_path=input_path,
        output_path=output_path,
        dataset_json=dataset_json,
        plans=plans,
        fold_predictors=fold_predictors,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=fold_values,
        checkpoint=checkpoint,
        use_tta=use_tta,
        compile_model=compile_model,
    )


def predict_from_modelfolder(
    model_folder: str,
    input_dir: str,
    output_dir: str,
    folds: list[int],
    checkpoint: str = "best",
    use_tta: bool = True,
    compile_model: bool = True,
) -> Path:
    if checkpoint not in {"best", "last"}:
        raise ValueError(
            f"checkpoint must be one of ('best', 'last'), got {checkpoint!r}"
        )

    fold_values = _validate_folds(folds)
    experiment_dir = Path(model_folder)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    dataset_json = load_dataset_json(experiment_dir)
    plans = load_mm_plans(experiment_dir / "mmPlans.json")
    metadata = _get_experiment_metadata(experiment_dir, fold_values, checkpoint)
    fold_predictors = _load_fold_predictors_from_experiment_dir(
        experiment_dir=experiment_dir,
        architecture_name=metadata["architecture_name"],
        folds=fold_values,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )
    return _run_prediction(
        dataset_id=metadata["dataset_id"],
        dataset_name=metadata["dataset_name"],
        input_path=input_path,
        output_path=output_path,
        dataset_json=dataset_json,
        plans=plans,
        fold_predictors=fold_predictors,
        trainer_name=metadata["trainer_name"],
        architecture_name=metadata["architecture_name"],
        experiment_postfix=metadata["experiment_postfix"],
        folds=fold_values,
        checkpoint=checkpoint,
        use_tta=use_tta,
        compile_model=compile_model,
    )
