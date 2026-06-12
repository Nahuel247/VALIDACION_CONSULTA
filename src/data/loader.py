from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, SUBTYPE_COLUMN, TEXT_COLUMN


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=";")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"El dataset debe contener la columna '{TEXT_COLUMN}'.")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"El dataset debe contener la columna '{LABEL_COLUMN}'.")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("").astype(str).str.strip()

    if SUBTYPE_COLUMN not in df.columns:
        df[SUBTYPE_COLUMN] = "sin_subtipo"
    else:
        df[SUBTYPE_COLUMN] = df[SUBTYPE_COLUMN].fillna("sin_subtipo").astype(str).str.strip()

    invalid_rows = df[(df[TEXT_COLUMN] == "") | (df[LABEL_COLUMN] == "")]
    if not invalid_rows.empty:
        preview = invalid_rows.head(5).to_dict(orient="records")
        raise ValueError(f"El dataset contiene filas vacias. Ejemplos: {preview}")

    return df
