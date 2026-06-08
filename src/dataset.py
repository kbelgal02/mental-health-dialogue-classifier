import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .config import LABEL2ID, TrainingConfig


class MentalHealthDataset:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def load_from_csv(
        self,
        path: Union[str, Path],
        text_col: str = "text",
        label_col: str = "label",
    ) -> DatasetDict:
        df = pd.read_csv(path)
        _validate_columns(df, text_col, label_col)
        df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
        df["label"] = df[label_col].map(LABEL2ID)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return self._split_and_tokenize(df, text_col)

    def load_from_dataframe(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> DatasetDict:
        _validate_columns(df, text_col, label_col)
        df = df.copy().dropna(subset=[text_col, label_col]).reset_index(drop=True)
        df["label"] = df[label_col].map(LABEL2ID).astype(int)
        return self._split_and_tokenize(df, text_col)

    def _split_and_tokenize(self, df: pd.DataFrame, text_col: str) -> DatasetDict:
        val_size = self.config.val_split + self.config.test_split
        train_df, temp_df = train_test_split(
            df, test_size=val_size, random_state=self.config.seed, stratify=df["label"]
        )
        relative_test = self.config.test_split / val_size
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test, random_state=self.config.seed, stratify=temp_df["label"]
        )

        splits = {
            "train": Dataset.from_dict({"text": train_df[text_col].tolist(), "label": train_df["label"].tolist()}),
            "validation": Dataset.from_dict({"text": val_df[text_col].tolist(), "label": val_df["label"].tolist()}),
            "test": Dataset.from_dict({"text": test_df[text_col].tolist(), "label": test_df["label"].tolist()}),
        }

        tokenized = {
            split: ds.map(self._tokenize_batch, batched=True, remove_columns=["text"])
            for split, ds in splits.items()
        }
        return DatasetDict(tokenized)

    def _tokenize_batch(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )

    def tokenize_single(self, text: str) -> dict:
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )


def _validate_columns(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}. Found: {list(df.columns)}")
