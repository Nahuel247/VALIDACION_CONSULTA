from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SPLIT_PATH,
    VALIDATION_SIZE,
    VALIDATION_SPLIT_PATH,
    TEST_SPLIT_PATH,
)
from src.utils.io import write_dataframe


def create_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )
    relative_validation_size = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_validation_size,
        stratify=temp_df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "train": TRAIN_SPLIT_PATH,
        "validation": VALIDATION_SPLIT_PATH,
        "test": TEST_SPLIT_PATH,
    }
    for name, path in mapping.items():
        write_dataframe(path, splits[name])


def load_saved_splits() -> dict[str, pd.DataFrame] | None:
    paths: dict[str, Path] = {
        "train": TRAIN_SPLIT_PATH,
        "validation": VALIDATION_SPLIT_PATH,
        "test": TEST_SPLIT_PATH,
    }
    if not all(path.exists() for path in paths.values()):
        return None
    return {name: pd.read_csv(path, encoding="utf-8-sig") for name, path in paths.items()}
