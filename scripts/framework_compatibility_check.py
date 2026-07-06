"""
Utility script used during development to verify that TensorFlow and
PyTorch models can be loaded in the same Python process.

Not required for the main project workflow.
"""

from tensorflow import keras
import torch
from transformers import AutoModelForSequenceClassification

print("Imports OK")

keras.models.load_model("models/bilstm/model.keras", compile=False)
print("TensorFlow model loaded")

AutoModelForSequenceClassification.from_pretrained(
    "models/bert_emotion_model_final"
)
print("BERT loaded")

print("Both models loaded successfully.")