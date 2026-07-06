"""
BiLSTM architecture for emotion classification: Embedding -> Bidirectional
LSTM -> Dropout -> Dense(softmax).

The vocabulary size is determined from the fitted tokenizer created
during preprocessing.
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
