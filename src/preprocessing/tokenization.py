"""
Tokenizer fitting, sequence padding, and label encoding for the BiLSTM
input pipeline.

Kept separate from dataset_builder.py / text_cleaning.py so each concern
(dataset unification, text cleaning, tokenization) can be re-run
independently -- e.g. you can refit the tokenizer with a different
vocab size without re-running label mapping.
"""

import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocessing.label_mapping import TARGET_CLASSES

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80
OOV_TOKEN = "<OOV>"


def fit_tokenizer(texts) -> Tokenizer:
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(tokenizer: Tokenizer, texts) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


def encode_labels(labels) -> np.ndarray:
    """
    Encodes emotion strings to integer indices using the fixed order in
    TARGET_CLASSES (not sklearn's alphabetical LabelEncoder default),
    so the mapping is stable and human-readable regardless of which
    classes happen to be present in a given run.
    """
    class_to_index = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    return np.array([class_to_index[label] for label in labels])


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
