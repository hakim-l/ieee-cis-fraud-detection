import argparse
import sys

from loguru import logger

from src.prepare_dataset.processor import DatasetProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the dataset from raw CSV files to interim Parquet format."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing interim Parquet files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose logging output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO" if args.verbose else "WARNING")

    processor = DatasetProcessor()
    processor.process_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
