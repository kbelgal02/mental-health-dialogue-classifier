"""Run inference with a trained model.

Usage:
    python -m src.predict --model models/mental_health_classifier --text "I can't stop worrying"
    python -m src.predict --model models/mental_health_classifier --file data/to_classify.csv
"""
import argparse
import json
import sys
import pandas as pd
from pathlib import Path

from .classifier import MentalHealthClassifier


def predict_text(classifier: MentalHealthClassifier, text: str) -> None:
    result = classifier.predict(text)
    print(f"\nInput: {text!r}")
    print(f"Label: {result['label'].upper()}  (confidence: {result['confidence']:.1%})")
    print(f"Description: {result['description']}")
    print("\nAll probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {label:<22} {prob:.1%}  {bar}")


def predict_file(classifier: MentalHealthClassifier, csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        print("Error: CSV must have a 'text' column.", file=sys.stderr)
        sys.exit(1)

    print(f"Classifying {len(df)} rows...")
    results = classifier.predict_batch(df["text"].tolist())

    df["predicted_label"] = [r["label"] for r in results]
    df["confidence"] = [round(r["confidence"], 4) for r in results]
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mental health dialogue classifier inference")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--text", help="Single text string to classify")
    parser.add_argument("--file", help="CSV file with a 'text' column to classify")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (used with --file)")
    parser.add_argument("--device", default=None, help="Force device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide --text or --file")

    print(f"Loading model from: {args.model}")
    classifier = MentalHealthClassifier(args.model, device=args.device)

    if args.text:
        predict_text(classifier, args.text)
    if args.file:
        predict_file(classifier, args.file, args.output)


if __name__ == "__main__":
    main()
