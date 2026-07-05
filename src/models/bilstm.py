"""
BiLSTM architecture for emotion classification: Embedding -> Bidirectional
LSTM -> Dropout -> Dense(softmax).

Matches the Epic 2 T3 specification (Embedding 128-dim, Bidirectional LSTM
128 units). Vocabulary size is read from the actual fitted tokenizer
rather than hardcoded at 30,000, since the real dataset almost certainly
uses fewer unique tokens than the tokenizer's cap -- see the training
script for why the resulting parameter count won't exactly match the
doc's reference figure of 4.1M.
"""

from tensorflow.keras import layers, models


def build_bilstm_model(vocab_size: int, embedding_dim: int = 128,
                        lstm_units: int = 128, max_seq_len: int = 80,
                        num_classes: int = 5, dropout_rate: float = 0.3) -> models.Sequential:
    model = models.Sequential([
        # Explicit Input layer forces the model to build its shapes
        # immediately, so model.summary() shows real parameter counts
        # before training starts (Keras 3 no longer infers this from
        # Embedding's input_length, which is deprecated).
        layers.Input(shape=(max_seq_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
