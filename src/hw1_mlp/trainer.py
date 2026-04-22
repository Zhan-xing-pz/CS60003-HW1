from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .evaluate import evaluate
from .model import MLPClassifier, compute_accuracy, softmax_cross_entropy


@dataclass
class TrainConfig:
    output_dir: str
    hidden_dim1: int
    hidden_dim2: int
    activation: str
    learning_rate: float
    lr_decay: float
    weight_decay: float
    batch_size: int
    epochs: int
    seed: int


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


class SGDOptimizer:
    def __init__(self, model: MLPClassifier, learning_rate: float, weight_decay: float):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self) -> None:
        for param, grad in self.model.parameters_and_grads():
            if grad.shape == param.shape and grad.ndim >= 2:
                param -= self.learning_rate * (grad + self.weight_decay * param)
            else:
                param -= self.learning_rate * grad


def train_model(
    model: MLPClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    optimizer = SGDOptimizer(model, config.learning_rate, config.weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_state = model.state_dict()
    best_val_acc = -np.inf

    for epoch in tqdm(range(config.epochs), desc="Training", unit="epoch"):
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        for batch_x, batch_y in iterate_minibatches(x_train, y_train, config.batch_size, rng):
            logits = model.forward(batch_x).logits
            ce_loss, grad_logits = softmax_cross_entropy(logits.copy(), batch_y)
            l2_penalty = 0.5 * config.weight_decay * (
                np.sum(model.fc1.weight**2)
                + np.sum(model.fc2.weight**2)
                + np.sum(model.fc3.weight**2)
            )
            loss = ce_loss + l2_penalty
            model.backward(grad_logits)
            optimizer.step()
            epoch_losses.append(float(loss))
            epoch_accs.append(compute_accuracy(logits, batch_y))

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))
        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.learning_rate)

        tqdm.write(f"  epoch {epoch+1:3d}  loss {train_loss:.4f}  acc {train_acc:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            np.savez(output_dir / "best_model.npz", **best_state)

        optimizer.learning_rate *= config.lr_decay

    model.load_state_dict(best_state)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    return history, best_state
