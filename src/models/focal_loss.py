"""
Sparse categorical focal loss for multi-class classification with severe
class imbalance.

Focal loss extends cross-entropy with a (1 - p_t)^gamma
term that down-weights already-easy, well-classified examples so the model
spends more of its gradient budget on hard/rare examples. The per-class
`alpha` term further re-weights each class's contribution to the loss.

Per project decision: alpha is ALWAYS computed automatically from the
actual training labels via sklearn's compute_class_weight (see
compute_alpha_from_labels below) -- never hardcoded. If the upstream
preprocessing changes and the class distribution shifts, alpha updates
automatically the next time this is called, rather than silently going
stale.
"""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Any single class ending up with an automatically-computed weight above
# this is flagged (not altered) -- a very large weight on a very rare
# class combined with focal loss's own down-weighting of easy examples
# can make training noisy on the few batches that contain that class.
ALPHA_WARNING_THRESHOLD = 15.0


def compute_alpha_from_labels(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Computes per-class alpha weights from the actual training labels
    using sklearn's 'balanced' scheme: weight_c = n_samples / (n_classes * n_c).

    This must be called on the TRAINING split only (not the full dataset),
    so the weighting reflects what the model actually trains on.
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
    Returns a Keras compatible loss function: loss(y_true, y_pred) where
    y_true is integer class indices (shape (batch,)) and y_pred is softmax
    probabilities (shape (batch, num_classes)).

    gamma=2.0 matches the project specification (Epic 2 T3 doc).
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
