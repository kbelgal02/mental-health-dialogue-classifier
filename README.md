# Mental Health Dialogue Classifier

A fine-tuned DistilBERT model that classifies short text passages from mental health dialogues into five clinical categories: **normal**, **depression**, **anxiety**, **stress**, and **crisis**. Designed as a research tool for triage support and content moderation in mental health applications.

---

## Why This Matters

Mental health crises are often first expressed in text — support chat logs, app check-ins, forum posts, helpline transcripts. Manually reviewing this volume of text is not scalable. An automated classifier that can flag high-risk language (especially **crisis** signals like suicidal ideation) allows human counselors to prioritize the conversations that need immediate attention.

This project demonstrates that a relatively small transformer model (DistilBERT, 66M parameters) can learn to distinguish between nuanced emotional states from short utterances — a practical trade-off between accuracy and deployment cost for real-time systems.

---

## Technical Approach

| Component | Details |
|---|---|
| Base model | `distilbert-base-uncased` (HuggingFace) |
| Task | 5-class sequence classification |
| Training framework | HuggingFace `Trainer` with `EarlyStoppingCallback` |
| Optimizer | AdamW, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1 |
| Best-model metric | Weighted F1 |
| Data splits | Stratified 75% train / 15% val / 10% test |
| Data augmentation | EDA (Wei & Zou, 2019): synonym replacement, random deletion, random swap |
| Evaluation output | Per-class precision/recall/F1 + confusion matrix saved to `test_report.txt` |

### Labels

| Label | Description |
|---|---|
| `normal` | General conversation with no significant clinical concern |
| `depression` | Indicators of depressive symptoms (sadness, hopelessness, low energy) |
| `anxiety` | Indicators of anxiety (excessive worry, fear, panic attacks) |
| `stress` | Overwhelm, burnout, or situational distress |
| `crisis` | Acute crisis, suicidal ideation, or urgent safety concern |

---

## Project Structure

```
mental_health_classifier/
├── data/
│   └── sample_data.csv        # Example training data (text, label columns)
├── src/
│   ├── config.py              # Label maps, TrainingConfig dataclass
│   ├── dataset.py             # CSV loading, stratified splits, tokenization
│   ├── augment.py             # EDA augmentation utilities
│   ├── train.py               # Training entrypoint
│   ├── evaluate.py            # Metrics and classification report
│   ├── classifier.py          # Inference wrapper (single and batch)
│   └── predict.py             # Prediction CLI
├── demo.py                    # Interactive terminal demo
└── requirements.txt
```

---

## Setup

```bash
cd mental_health_classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU is optional — CPU works for inference and small training sets.

---

## Training

Your CSV must have a `text` column and a `label` column. Labels must be one of: `normal`, `depression`, `anxiety`, `stress`, `crisis`.

```bash
# Basic training run
python -m src.train --data data/sample_data.csv

# Custom settings
python -m src.train \
  --data data/sample_data.csv \
  --model_name distilbert-base-uncased \
  --output_dir models/my_model \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --fp16          # enable mixed precision (requires CUDA)
```

Training saves the best checkpoint (by weighted F1) to `output_dir` and writes a full test-set classification report to `output_dir/test_report.txt`.

### Data Augmentation

To expand a small training set before training:

```python
from src.augment import augment_dataframe
import pandas as pd

df = pd.read_csv("data/sample_data.csv")
df_augmented = augment_dataframe(df, ops_per_sample=2)
df_augmented.to_csv("data/augmented.csv", index=False)
```

---

## Inference

### Python API

```python
from src.classifier import MentalHealthClassifier

clf = MentalHealthClassifier("models/mental_health_classifier")
result = clf.predict("I haven't left my room in days and everything feels pointless.")

print(result["label"])        # e.g. "depression"
print(result["confidence"])   # e.g. 0.91
print(result["description"])  # human-readable label description
print(result["probabilities"]) # per-class softmax probabilities
```

Batch prediction:

```python
results = clf.predict_batch(["I'm fine.", "I can't stop worrying about everything."])
```

### Interactive Demo

```bash
python demo.py --model models/mental_health_classifier
```

The demo displays color-coded predictions and probability bars. If the `crisis` label is predicted, it immediately prints emergency resources:

- **988 Suicide & Crisis Lifeline** (call or text, US)
- **Crisis Text Line** — text HOME to 741741
- **IASP international crisis centre directory**

---

## Limitations

- **Not a diagnostic tool.** This model predicts surface-level text patterns, not clinical diagnoses. It should never be used as a substitute for professional mental health assessment.
- **English only.** The base model and training data are English-language only.
- **Short-text optimized.** Input is truncated at 128 tokens. Long passages may lose relevant context.
- **Distribution sensitivity.** Performance degrades on text that differs significantly from training data (e.g., highly formal clinical notes vs. casual chat).
- **Class overlap.** Depression, anxiety, and stress share lexical patterns. Borderline cases may be misclassified.
- **Crisis recall is critical.** In safety-critical deployments, threshold tuning to maximize `crisis` recall (at the cost of precision) is strongly recommended over using argmax predictions.

---

## Ethical Considerations

- **Automated flagging is a tool, not a decision-maker.** Any crisis or high-risk prediction should route to a qualified human reviewer — never trigger automated responses without human oversight.
- **Privacy.** Mental health text is highly sensitive. Ensure data collection, storage, and model inference comply with applicable regulations (HIPAA, GDPR) and that users provide informed consent.
- **Bias.** The model reflects patterns in its training data. It may perform unevenly across dialects, cultural expressions of distress, or demographic groups not well-represented in training.
- **Stigma.** Labels should be used to route support, not to permanently tag or profile individuals.
- **Transparency.** Inform users when an automated system is analyzing their messages for mental health content.

---

## References

- Wei, J. & Zou, K. (2019). [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196). EMNLP 2019.
- Sanh, V. et al. (2019). [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108).
- Wolf, T. et al. (2020). [HuggingFace Transformers](https://arxiv.org/abs/1910.03771).
