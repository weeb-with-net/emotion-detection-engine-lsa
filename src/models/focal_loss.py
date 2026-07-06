"""
Implementation of Sparse Categorical Focal Loss for multi-class
emotion classification.

This loss function helps reduce the impact of class imbalance by
placing greater emphasis on harder-to-classify examples.

Class weights are computed dynamically from the training data during
training rather than being hardcoded.
"""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Warn when a class receives an unusually large focal-loss weight.
ALPHA_WARNING_THRESHOLD = 15.0

def compute_alpha_from_labels(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute per-class alpha weights from the training labels.

    Uses scikit-learn's balanced class-weight formula.
    """
    class_indices = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=class_indices, y=y_train)

    print("\n== Automatically computed class weights (alpha) ==")
    for idx, w in zip(class_indices, weights):
        flag = "  [WARNING: high weight, may cause noisy training]" if w > ALPHA_WARNING_THRESHOLD else ""
        print(f"  class {idx}: {w:.3f}{flag}")

    return weights


def sparse_categorical_focal_loss(alpha: np.ndarray, gamma: float = 2.0, num_classes: int = 5):
    """
    Create a Keras-compatible sparse categorical focal loss function.

    Parameters:
        alpha: Per-class weighting factors.
        gamma: Focusing parameter for difficult examples.
        num_classes: Number of target classes.
    """
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        y_true_onehot = tf.one_hot(y_true, depth=num_classes)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)

        modulating_factor = tf.pow(1.0 - y_pred, gamma)
        weighted = alpha_tensor * modulating_factor * cross_entropy

        # Sum over classes: only the true class's term is non-zero per row
        # (one-hot zeroes out everything else), so this collapses to
        # alpha_true * (1 - p_true)^gamma * -log(p_true) per example.
        per_example_loss = tf.reduce_sum(weighted, axis=1)
        return tf.reduce_mean(per_example_loss)

    return loss_fn
