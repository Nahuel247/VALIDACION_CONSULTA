import pandas as pd

from src.data.preprocessing import normalize_dataset, validate_labels


def test_normalize_dataset_lowercases_labels() -> None:
    df = pd.DataFrame({"text": [" Hola   mundo "], "label": ["VALIDA"], "subtype": ["General"]})
    normalized = normalize_dataset(df)
    assert normalized.loc[0, "text"] == "Hola mundo"
    assert normalized.loc[0, "label"] == "valida"


def test_validate_labels_accepts_binary_labels() -> None:
    df = pd.DataFrame({"text": ["A", "B"], "label": ["valida", "no_valida"], "subtype": ["x", "y"]})
    validate_labels(df)
