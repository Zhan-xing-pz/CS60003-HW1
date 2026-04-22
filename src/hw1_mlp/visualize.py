from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_weight_visualizations(
    first_layer_weight: np.ndarray,
    image_shape: tuple[int, int, int],
    output_path: str | Path,
    max_filters: int = 16,
) -> None:
    num_filters = min(max_filters, first_layer_weight.shape[1])
    cols = 4
    rows = int(np.ceil(num_filters / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx in range(num_filters):
        weight = first_layer_weight[:, idx].reshape(image_shape)
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(weight)
        plt.axis("off")
        plt.title(f"Neuron {idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_error_analysis(
    x_test: np.ndarray,
    y_test: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    image_shape: tuple[int, int, int],
    output_path: str | Path,
    max_examples: int = 12,
) -> list[dict[str, str]]:
    misclassified = np.where(preds != y_test)[0][:max_examples]
    examples: list[dict[str, str]] = []
    if len(misclassified) == 0:
        return examples

    cols = 4
    rows = int(np.ceil(len(misclassified) / cols))
    plt.figure(figsize=(3.5 * cols, 3.5 * rows))
    for plot_idx, sample_idx in enumerate(misclassified, start=1):
        image = x_test[sample_idx].reshape(image_shape)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        true_name = class_names[int(y_test[sample_idx])]
        pred_name = class_names[int(preds[sample_idx])]
        examples.append({"true": true_name, "pred": pred_name, "index": str(int(sample_idx))})
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"T:{true_name}\nP:{pred_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return examples
