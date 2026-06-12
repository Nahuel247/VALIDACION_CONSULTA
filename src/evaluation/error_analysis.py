from __future__ import annotations

import pandas as pd

from src.config import MISCLASSIFIED_PATH


def save_misclassified_examples(test_df: pd.DataFrame, predicted_labels: list[str]) -> pd.DataFrame:
    analyzed = test_df.copy().reset_index(drop=True)
    analyzed["predicted_label"] = predicted_labels
    misclassified = analyzed[analyzed["label"] != analyzed["predicted_label"]].copy()
    misclassified.to_csv(MISCLASSIFIED_PATH, index=False, encoding="utf-8-sig")
    return misclassified
