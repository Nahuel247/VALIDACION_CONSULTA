from __future__ import annotations

from datasets import Dataset

from src.config import LABELS_PATH, RAW_DATASET_PATH, TRAINED_MODEL_DIR, ensure_project_dirs
from src.data.loader import load_dataset
from src.data.preprocessing import normalize_dataset, validate_labels
from src.data.split import create_splits, save_splits
from src.training.save_artifacts import save_label_mapping
from src.training.trainer_factory import (
    build_label_maps,
    build_model,
    build_tokenizer,
    build_trainer,
    tokenize_splits,
)
from src.utils.seed import set_seed
from src.config import RANDOM_STATE


def main() -> None:
    ensure_project_dirs()
    set_seed(RANDOM_STATE)

    df = load_dataset(RAW_DATASET_PATH)
    df = normalize_dataset(df)
    validate_labels(df)

    splits = create_splits(df)
    save_splits(splits)

    labels = sorted(df["label"].unique())
    label2id, id2label = build_label_maps(labels)

    hf_splits = {
        name: Dataset.from_pandas(split_df[["text", "label"]].assign(labels=split_df["label"].map(label2id))[["text", "labels"]], preserve_index=False)
        for name, split_df in splits.items()
    }

    tokenizer = build_tokenizer()
    tokenized_splits = tokenize_splits(hf_splits, tokenizer)
    model = build_model(label2id=label2id, id2label=id2label)
    trainer = build_trainer(model=model, tokenizer=tokenizer, tokenized_splits=tokenized_splits)

    trainer.train()

    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(TRAINED_MODEL_DIR))
    tokenizer.save_pretrained(str(TRAINED_MODEL_DIR))
    save_label_mapping(LABELS_PATH, label2id=label2id, id2label=id2label)

    print(f"Modelo exportado en: {TRAINED_MODEL_DIR}")


if __name__ == "__main__":
    main()
