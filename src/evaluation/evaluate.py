from __future__ import annotations

import json

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import Trainer

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    LABELS_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SPLIT_PATH,
    TRAINED_MODEL_DIR,
    ensure_project_dirs,
)
from src.data.preprocessing import normalize_dataset
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.error_analysis import save_misclassified_examples
from src.training.metrics import compute_metrics
from src.training.trainer_factory import build_tokenizer
from src.utils.io import write_json
from src.utils.seed import set_seed
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding


def load_label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    label2id = {key: int(value) for key, value in payload["label2id"].items()}
    id2label = {int(key): value for key, value in payload["id2label"].items()}
    return label2id, id2label


def main() -> None:
    ensure_project_dirs()
    set_seed(RANDOM_STATE)

    if not TRAINED_MODEL_DIR.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado en {TRAINED_MODEL_DIR}")
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError("No existe test.csv. Ejecuta primero el entrenamiento.")

    test_df = pd.read_csv(TEST_SPLIT_PATH, encoding="utf-8-sig")
    test_df = normalize_dataset(test_df)
    label2id, id2label = load_label_mapping()

    tokenizer = build_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(str(TRAINED_MODEL_DIR), local_files_only=True)

    hf_test = Dataset.from_pandas(
        test_df[["text", "label"]].assign(labels=test_df["label"].map(label2id))[["text", "labels"]],
        preserve_index=False,
    )
    tokenized_test = hf_test.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=128), batched=True)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_test)
    test_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    write_json(METRICS_PATH, test_metrics)

    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    predicted_labels = [id2label[int(idx)] for idx in predicted_ids]
    true_labels = [id2label[int(idx)] for idx in predictions.label_ids]

    report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_PATH, encoding="utf-8-sig")
    save_misclassified_examples(test_df=test_df, predicted_labels=predicted_labels)
    save_confusion_matrix(labels_true=true_labels, labels_pred=predicted_labels, target_names=sorted(label2id.keys()))

    print("Metricas:")
    print(test_metrics)
    print(f"Reporte guardado en: {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
