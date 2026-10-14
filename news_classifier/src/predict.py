"""Predict a label for a single news text using a saved model."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib


def predict(model_path: Path, text: str) -> tuple[str, float]:
    model = joblib.load(model_path)
    probabilities = model.predict_proba([text])[0]
    index = probabilities.argmax()
    return str(model.classes_[index]), float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/fake_news_model.joblib"))
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    label, confidence = predict(args.model, args.text)
    print(f"Prediction: {label.upper()} | model confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
