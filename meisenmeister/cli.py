import argparse

from meisenmeister.plan_and_preprocess.create_breast_seg import (
    create_breast_segmentations,
)
from meisenmeister.plan_and_preprocess.extract_dataset_fingerprint import (
    extract_dataset_fingerprint,
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


if __name__ == "__main__":
    mm_extract_dataset_fingerprint()
