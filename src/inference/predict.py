from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import LABELS_PATH, MAX_LENGTH, TRAINED_MODEL_DIR


def load_id2label() -> dict[int, str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {int(key): value for key, value in payload["id2label"].items()}


def load_inference_artifacts():
    if not TRAINED_MODEL_DIR.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado en {TRAINED_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(TRAINED_MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(TRAINED_MODEL_DIR), local_files_only=True)
    id2label = load_id2label()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, id2label, device


def predict_with_loaded_model(text: str, tokenizer, model, id2label, device) -> dict[str, object]:
    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu()

    predicted_idx = int(torch.argmax(probabilities).item())
    predicted_label = id2label[predicted_idx]
    return {
        "text": text,
        "predicted_label": predicted_label,
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {id2label[idx]: float(score) for idx, score in enumerate(probabilities.tolist())},
    }


def print_prediction(result: dict[str, object]) -> None:
    print(f"Texto: {result['text']}")
    print(f"Prediccion: {result['predicted_label']}")
    print(f"Confianza: {result['confidence']:.4f}")
    print(result["probabilities"])


def predict(text: str) -> None:
    tokenizer, model, id2label, device = load_inference_artifacts()
    result = predict_with_loaded_model(text, tokenizer, model, id2label, device)
    print_prediction(result)


def interactive_predict() -> None:
    tokenizer, model, id2label, device = load_inference_artifacts()
    print("Modelo cargado. Escribe una pregunta para clasificarla.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        text = input("Pregunta> ").strip()
        if not text:
            continue
        if text.lower() in {"salir", "exit", "quit"}:
            print("Sesion terminada.")
            break

        result = predict_with_loaded_model(text, tokenizer, model, id2label, device)
        print_prediction(result)
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inferencia rapida para el clasificador municipal")
    parser.add_argument("--text", help="Texto a clasificar")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mantiene el modelo cargado y permite clasificar multiples preguntas.",
    )
    args = parser.parse_args(argv)

    if args.interactive:
        interactive_predict()
        return

    if not args.text:
        parser.error("Debes indicar --text o usar --interactive.")
    predict(args.text)


if __name__ == "__main__":
    main(sys.argv[1:])
