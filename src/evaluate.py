import numpy as np
from typing import Optional
import evaluate as hf_evaluate
from sklearn.metrics import classification_report, confusion_matrix

from .config import ID2LABEL

_accuracy = hf_evaluate.load("accuracy")
_f1 = hf_evaluate.load("f1")


def compute_metrics(eval_pred) -> dict:
    """Used by HuggingFace Trainer during training."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = _accuracy.compute(predictions=predictions, references=labels)
    f1 = _f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def full_report(predictions: list[int], labels: list[int], output_path: Optional[str] = None) -> str:
    """Print and optionally save a detailed classification report."""
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    report = classification_report(labels, predictions, target_names=target_names, digits=4)
    cm = confusion_matrix(labels, predictions)

    cm_str = _format_confusion_matrix(cm, target_names)
    full = f"Classification Report\n{'='*60}\n{report}\n\nConfusion Matrix\n{'='*60}\n{cm_str}"

    print(full)
    if output_path:
        with open(output_path, "w") as f:
            f.write(full)
    return full


def _format_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    header = "Predicted →\n" + " " * 22 + "  ".join(f"{l[:8]:>8}" for l in labels)
    rows = [header]
    for i, row in enumerate(cm):
        row_str = f"Actual {labels[i]:<14}" + "  ".join(f"{v:>8}" for v in row)
        rows.append(row_str)
    return "\n".join(rows)
