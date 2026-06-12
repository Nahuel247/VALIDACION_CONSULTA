import numpy as np

from src.training.metrics import compute_metrics


def test_compute_metrics_returns_accuracy_and_f1() -> None:
    logits = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
