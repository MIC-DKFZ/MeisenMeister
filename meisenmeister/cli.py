import argparse

from meisenmeister.plan_and_preprocess.create_breast_seg import (
    create_breast_segmentations,
)
from meisenmeister.plan_and_preprocess.extract_dataset_fingerprint import (
    extract_dataset_fingerprint,
)
from meisenmeister.plan_and_preprocess.homogenize import homogenize
from meisenmeister.plan_and_preprocess.plan_and_preprocess import plan_and_preprocess
from meisenmeister.plan_and_preprocess.plan_experiment import plan_experiment
from meisenmeister.plan_and_preprocess.preprocess import preprocess
from meisenmeister.training.splits import create_five_fold_splits
from meisenmeister.training.train import train
from meisenmeister.utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)


def mm_extract_dataset_fingerprint() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_extract_dataset_fingerprint",
        description="Run the extract_dataset_fingerprint entrypoint.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for ROI fingerprint extraction.",
    )
    args = parser.parse_args()
    extract_dataset_fingerprint(args.d, num_workers=args.num_workers)


def mm_create_breast_segmentations() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_create_breast_segmentations",
        description="Create breast segmentations in masksTr next to imagesTr.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()
    create_breast_segmentations(args.d)


def mm_homogenize() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_homogenize",
        description="Resample all non-_0000 images to the _0000 space and overwrite the raw NIfTI files.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()

    confirmation = input(
        "ARE YOU SURE YOU WANT TO OVERWRITE YOUR RAW DATA? Type 'yes' to continue: "
    ).strip()
    if confirmation != "yes":
        raise SystemExit("Aborted.")

    homogenize(args.d)


def mm_plan_experment() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_plan_experment",
        description="Create mmPlans.json from dataset_fingerprint.json.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()
    plan_experiment(args.d)


def mm_preprocess() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_preprocess",
        description="Preprocess ROI crops into blosc2 b2nd format from mmPlans.json.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for preprocessing.",
    )
    args = parser.parse_args()
    preprocess(args.d, num_workers=args.num_workers)


def mm_plan_and_preprocess() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_plan_and_preprocess",
        description="Run planning and preprocessing for a dataset.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for preprocessing.",
    )
    args = parser.parse_args()
    plan_and_preprocess(args.d, num_workers=args.num_workers)


def mm_train() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_train",
        description="Resolve a trainer, build its training components, and call fit().",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="Fold index from splits.json.",
    )
    parser.add_argument(
        "--trainer",
        default="mmTrainer",
        help="Trainer class name registered under meisenmeister.training.trainers.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default="ResNet3D18",
        help="Architecture class name registered under meisenmeister.architectures.",
    )
    parser.add_argument(
        "-c",
        "--continue-training",
        action="store_true",
        help="Continue training from model_last.pt in the computed experiment fold directory.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Path to pretrained weights for starting a fresh fine-tuning run.",
    )
    parser.add_argument(
        "--postfix",
        help="Optional suffix appended to the experiment name.",
    )
    parser.add_argument(
        "--val",
        nargs="?",
        const="last",
        choices=("last", "best"),
        help="Skip training and run final validation only. `--val` uses model_last.pt and writes eval_last.json; `--val best` uses model_best.pt and writes eval_best.json.",
    )
    args = parser.parse_args()
    train(
        args.d,
        fold=args.fold,
        trainer_name=args.trainer,
        architecture_name=args.architecture,
        continue_training=args.continue_training,
        weights_path=args.weights,
        experiment_postfix=args.postfix,
        val=args.val,
    )


@require_global_paths_set
def mm_create_5fold() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_create_5fold",
        description="Create a leakage-safe 5-fold splits.json in the preprocessed dataset folder.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()

    if not 0 <= args.d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {args.d}")

    dataset_id = f"{args.d:03d}"
    paths = verify_required_global_paths_set()
    dataset_dir = find_dataset_dir(paths["mm_raw"], dataset_id)
    preprocessed_dataset_dir = paths["mm_preprocessed"] / dataset_dir.name
    output_path = create_five_fold_splits(preprocessed_dataset_dir)
    print(output_path)


if __name__ == "__main__":
    mm_extract_dataset_fingerprint()
