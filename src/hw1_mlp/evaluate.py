from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from .model import MLPClassifier, compute_accuracy, softmax_cross_entropy


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    logits = model.forward(x).logits
    loss, _ = softmax_cross_entropy(logits.copy(), y)
    accuracy = compute_accuracy(logits, y)
    preds = logits.argmax(axis=1)
    return loss, accuracy, preds


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
