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


if __name__ == "__main__":
    mm_extract_dataset_fingerprint()
