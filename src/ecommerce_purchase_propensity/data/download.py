"""Download and validate the raw Online Shoppers dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from ecommerce_purchase_propensity.data.schema import EXPECTED_COLUMNS
from ecommerce_purchase_propensity.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)

ARCHIVE_MEMBER = "online_shoppers_intention.csv"

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"

EXPECTED_ROWS = 12_330


def validate_dataset(dataset_path: Path) -> None:
    """Validate the raw dataset structure."""
    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError("Downloaded dataset is empty.")

    if list(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match the expected schema.")

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(dataframe)}.")

    LOGGER.info(
        "Dataset validation successful: %s rows and %s columns",
        len(dataframe),
        len(dataframe.columns),
    )


def _download_archive(destination: Path) -> None:
    """Download the dataset archive from the UCI repository."""
    LOGGER.info("Downloading dataset from UCI repository")

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("wb") as archive_file,
        ):
            shutil.copyfileobj(response, archive_file)
    except urllib.error.URLError as error:
        raise RuntimeError("Failed to download dataset from UCI repository.") from error


def _extract_dataset(archive_path: Path, output_path: Path) -> None:
    """Extract the dataset CSV from the downloaded archive."""
    LOGGER.info("Extracting %s", ARCHIVE_MEMBER)

    with zipfile.ZipFile(archive_path) as archive:
        if ARCHIVE_MEMBER not in archive.namelist():
            raise ValueError(
                f"{ARCHIVE_MEMBER} was not found in the downloaded archive."
            )

        with (
            archive.open(ARCHIVE_MEMBER) as source,
            output_path.open("wb") as destination,
        ):
            shutil.copyfileobj(source, destination)


def download_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    force: bool = False,
) -> Path:
    """Download, extract, and validate the raw dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        LOGGER.info("Dataset already exists at %s", output_path)
        validate_dataset(output_path)
        return output_path

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "dataset.zip"

        _download_archive(archive_path)
        _extract_dataset(archive_path, output_path)

    validate_dataset(output_path)

    LOGGER.info("Dataset saved to %s", output_path)

    return output_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the Online Shoppers dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the dataset even if it already exists.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the dataset download command."""
    setup_logging()

    args = _parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
