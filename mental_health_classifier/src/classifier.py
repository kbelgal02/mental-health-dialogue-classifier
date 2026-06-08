import torch
from pathlib import Path
from typing import Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import LABEL2ID, ID2LABEL, LABEL_DESCRIPTIONS, TrainingConfig


class MentalHealthClassifier:
    """Wraps a fine-tuned DistilBERT model for mental health dialogue classification."""

    def __init__(self, model_path_or_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path_or_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Return the predicted label, confidence, and per-class probabilities."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu()
        predicted_id = int(probs.argmax())
        label = ID2LABEL[predicted_id]

        return {
            "label": label,
            "confidence": float(probs[predicted_id]),
            "description": LABEL_DESCRIPTIONS[label],
            "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu()
            for row in probs:
                predicted_id = int(row.argmax())
                label = ID2LABEL[predicted_id]
                results.append({
                    "label": label,
                    "confidence": float(row[predicted_id]),
                    "description": LABEL_DESCRIPTIONS[label],
                    "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(row)},
                })
        return results

    def save(self, path: Union[str, Path]) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None) -> "MentalHealthClassifier":
        return cls(path, device=device)
