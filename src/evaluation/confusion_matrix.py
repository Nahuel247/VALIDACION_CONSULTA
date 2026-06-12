from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import CONFUSION_MATRIX_FIGURE_PATH, CONFUSION_MATRIX_PATH


def save_confusion_matrix(labels_true, labels_pred, target_names: list[str]) -> None:
    matrix = confusion_matrix(labels_true, labels_pred, labels=target_names)
    df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    df.to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Prediccion")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_FIGURE_PATH, dpi=150)
        plt.close()
    except Exception:
        pass
