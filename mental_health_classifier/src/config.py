from dataclasses import dataclass, field
from typing import Optional

LABEL2ID = {
    "normal": 0,
    "depression": 1,
    "anxiety": 2,
    "stress": 3,
    "crisis": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_DESCRIPTIONS = {
    "normal": "General conversation with no significant clinical concern",
    "depression": "Indicators of depressive symptoms (sadness, hopelessness, low energy)",
    "anxiety": "Indicators of anxiety (excessive worry, fear, panic attacks)",
    "stress": "Overwhelm, burnout, or situational distress",
    "crisis": "Acute crisis, suicidal ideation, or urgent safety concern",
}


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = len(LABEL2ID)
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./models/mental_health_classifier"
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    seed: int = 42
    fp16: bool = False
    val_split: float = 0.15
    test_split: float = 0.10
