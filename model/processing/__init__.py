"""Contains utility functions for dataset initialization"""
from pathlib import Path
import pandas as pd
import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = PACKAGE_DIR / "config.yml"
DATASET_DIR = PACKAGE_DIR / "dataset"


def load() -> DataFrame:
    """Loads the training data file into a dataframe.

    Returns:
        DataFrame: Training records
    """
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(config)

    df = pd.read_csv(DATASET_DIR / config["train"]["dataset_file_name"])

    return df
