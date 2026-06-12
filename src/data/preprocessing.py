from __future__ import annotations

import pandas as pd

from src.config import LABEL_COLUMN, SUBTYPE_COLUMN, TEXT_COLUMN


VALID_LABELS = {"valida", "no_valida"}


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].astype(str).str.strip()
    normalized[LABEL_COLUMN] = normalized[LABEL_COLUMN].astype(str).str.strip().str.lower()
    normalized[SUBTYPE_COLUMN] = normalized[SUBTYPE_COLUMN].astype(str).str.strip().str.lower()
    normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].str.replace(r"\s+", " ", regex=True)
    return normalized


def validate_labels(df: pd.DataFrame) -> None:
    labels = set(df[LABEL_COLUMN].unique())
    unexpected = labels.difference(VALID_LABELS)
    if unexpected:
        raise ValueError(f"Se encontraron etiquetas no soportadas: {sorted(unexpected)}")
