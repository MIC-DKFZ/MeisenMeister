import argparse

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
    args = parser.parse_args()
    extract_dataset_fingerprint(args.d)


if __name__ == "__main__":
    mm_extract_dataset_fingerprint()
