from __future__ import annotations

import argparse
import sys

from src.evaluation.evaluate import main as evaluate_main
from src.inference.predict import main as predict_main
from src.training.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Punto de entrada para entrenar, evaluar o clasificar preguntas municipales."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "interactive"],
        default="predict",
        help="Operacion a ejecutar.",
    )
    parser.add_argument(
        "--text",
        help="Texto a clasificar cuando el modo es 'predict'.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "train":
        train_main()
        return

    if args.mode == "evaluate":
        evaluate_main()
        return

    if args.mode == "interactive":
        predict_main(["--interactive"])
        return

    if not args.text:
        parser.error("Debes indicar --text cuando uses --mode predict.")
    predict_main(["--text", args.text])


if __name__ == "__main__":
    main(sys.argv[1:])
