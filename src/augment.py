"""Easy Data Augmentation (EDA) utilities for expanding small training sets.

Implements three label-preserving augmentation operations from Wei & Zou (2019):
  - Synonym replacement via WordNet
  - Random word deletion
  - Random word swap

Usage:
    from src.augment import augment_dataframe
    augmented_df = augment_dataframe(df, ops_per_sample=2)
"""
import random
import re
from typing import Optional

import nltk
import pandas as pd

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet


def _synonyms(word: str) -> list[str]:
    clean = re.sub(r"[^a-zA-Z]", "", word)
    syns: set[str] = set()
    for syn in wordnet.synsets(clean):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != clean.lower():
                syns.add(candidate)
    return list(syns)


def synonym_replacement(text: str, n: int = 2) -> str:
    words = text.split()
    replaceable = [i for i, w in enumerate(words) if _synonyms(w)]
    random.shuffle(replaceable)
    for idx in replaceable[:n]:
        syns = _synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
    return " ".join(words)


def random_deletion(text: str, p: float = 0.15) -> str:
    words = text.split()
    if len(words) == 1:
        return text
    kept = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else random.choice(words)


def random_swap(text: str, n: int = 1) -> str:
    words = text.split()
    for _ in range(n):
        if len(words) < 2:
            break
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


_OPS = [synonym_replacement, random_deletion, random_swap]


def augment_text(text: str, n: int = 1) -> list[str]:
    """Return `n` augmented variants of `text` using randomly chosen EDA ops."""
    return [random.choice(_OPS)(text) for _ in range(n)]


def augment_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    ops_per_sample: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Return original rows + `ops_per_sample` augmented copies per row, shuffled."""
    random.seed(seed)
    new_rows = [
        {text_col: aug, label_col: row[label_col]}
        for _, row in df.iterrows()
        for aug in augment_text(row[text_col], n=ops_per_sample)
    ]
    return (
        pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
