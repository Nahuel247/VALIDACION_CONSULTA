from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BASE_MODEL_NAME_OR_PATH,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_LENGTH,
    NUM_TRAIN_EPOCHS,
    REPORT_TO,
    SAVE_TOTAL_LIMIT,
    TRAIN_BATCH_SIZE,
    TRAINED_MODEL_DIR,
    WEIGHT_DECAY,
)
from src.training.metrics import compute_metrics


def use_local_files_only() -> bool:
    return Path(BASE_MODEL_NAME_OR_PATH).exists()


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(sorted(labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME_OR_PATH, local_files_only=use_local_files_only())


def tokenize_splits(splits: dict[str, Dataset], tokenizer) -> DatasetDict:
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    dataset_dict = DatasetDict(splits)
    return dataset_dict.map(tokenize_batch, batched=True)


def build_model(label2id: dict[str, int], id2label: dict[int, str]):
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=use_local_files_only(),
    )


def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(TRAINED_MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=REPORT_TO,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )


def build_trainer(model, tokenizer, tokenized_splits: DatasetDict) -> Trainer:
    return Trainer(
        model=model,
        args=build_training_args(),
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
