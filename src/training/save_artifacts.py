from __future__ import annotations

import json
from pathlib import Path

from src.utils.io import write_json


def save_label_mapping(path: Path, label2id: dict[str, int], id2label: dict[int, str]) -> None:
    payload = {
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    write_json(path, payload)
