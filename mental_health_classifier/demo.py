"""Interactive prediction demo for the mental health classifier.

Usage (run from the mental_health_classifier/ directory):
    python demo.py --model models/mental_health_classifier
    python demo.py --model models/mental_health_classifier --device cpu
"""
import argparse
import sys

CRISIS_RESOURCES = (
    "\n  !! CRISIS DETECTED — Please reach out for help immediately:\n"
    "  * National Suicide Prevention Lifeline : 988 (call or text, US)\n"
    "  * Crisis Text Line                     : Text HOME to 741741\n"
    "  * International resources              : https://www.iasp.info/resources/Crisis_Centres/\n"
)

_COLORS = {
    "normal":     "\033[92m",
    "depression": "\033[94m",
    "anxiety":    "\033[93m",
    "stress":     "\033[95m",
    "crisis":     "\033[91m",
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _c(label: str, text: str) -> str:
    return f"{_COLORS.get(label, '')}{text}{_RESET}"


def _display(text: str, result: dict) -> None:
    label = result["label"]
    conf  = result["confidence"]
    print(f"\n{_BOLD}Input:{_RESET} {text!r}")
    print(f"{_BOLD}Label:{_RESET} {_c(label, label.upper())}  ({conf:.1%} confidence)")
    print(f"{_BOLD}Desc: {_RESET} {result['description']}")
    print(f"\n{_BOLD}Probabilities:{_RESET}")
    for lbl, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {_c(lbl, f'{lbl:<22}')} {prob:>5.1%}  {bar}")
    if label == "crisis":
        print(_c("crisis", CRISIS_RESOURCES))


def _run_interactive(classifier) -> None:
    print(f"\n{_BOLD}Mental Health Dialogue Classifier — Interactive Demo{_RESET}")
    print("Type a message and press Enter to classify it.")
    print("Commands: 'q' or 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        result = classifier.predict(text)
        _display(text, result)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive mental health classifier demo")
    parser.add_argument("--model",  required=True, help="Path to trained model directory")
    parser.add_argument("--device", default=None,  help="'cpu' or 'cuda' (auto-detected if omitted)")
    args = parser.parse_args()

    try:
        from src.classifier import MentalHealthClassifier
    except ImportError:
        sys.exit("ImportError: run this script from the mental_health_classifier/ project root.")

    print(f"Loading model from: {args.model} ...")
    classifier = MentalHealthClassifier(args.model, device=args.device)
    _run_interactive(classifier)


if __name__ == "__main__":
    main()
