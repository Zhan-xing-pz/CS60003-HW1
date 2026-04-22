from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        self.bias = np.zeros((1, out_features), dtype=np.float32)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self.inputs: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return x @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.inputs.shape[0]
        self.grad_weight[...] = self.inputs.T @ grad_output / batch_size
        self.grad_bias[...] = grad_output.mean(axis=0, keepdims=True)
        return grad_output @ self.weight.T

    def parameters_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]


class ReLU(Layer):
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self.mask


class Sigmoid(Layer):
    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self.output * (1.0 - self.output)


class Tanh(Layer):
    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * (1.0 - self.output**2)


def build_activation(name: str) -> Layer:
    normalized = name.lower()
    if normalized == "relu":
        return ReLU()
    if normalized == "sigmoid":
        return Sigmoid()
    if normalized == "tanh":
        return Tanh()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class ForwardResult:
    logits: np.ndarray
    hidden1: np.ndarray
    hidden2: np.ndarray


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(input_dim, hidden_dim1, rng)
        self.act1 = build_activation(activation)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, rng)
        self.act2 = build_activation(activation)
        self.fc3 = Linear(hidden_dim2, num_classes, rng)

    def forward(self, x: np.ndarray) -> ForwardResult:
        hidden1_linear = self.fc1.forward(x)
        hidden1 = self.act1.forward(hidden1_linear)
        hidden2_linear = self.fc2.forward(hidden1)
        hidden2 = self.act2.forward(hidden2_linear)
        logits = self.fc3.forward(hidden2)
        return ForwardResult(logits=logits, hidden1=hidden1, hidden2=hidden2)

    def backward(self, grad_logits: np.ndarray) -> None:
        grad_hidden2 = self.fc3.backward(grad_logits)
        grad_hidden2 = self.act2.backward(grad_hidden2)
        grad_hidden1 = self.fc2.backward(grad_hidden2)
        grad_hidden1 = self.act1.backward(grad_hidden1)
        self.fc1.backward(grad_hidden1)

    def parameters_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return (
            self.fc1.parameters_and_grads()
            + self.fc2.parameters_and_grads()
            + self.fc3.parameters_and_grads()
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "fc1_weight": self.fc1.weight.copy(),
            "fc1_bias": self.fc1.bias.copy(),
            "fc2_weight": self.fc2.weight.copy(),
            "fc2_bias": self.fc2.bias.copy(),
            "fc3_weight": self.fc3.weight.copy(),
            "fc3_bias": self.fc3.bias.copy(),
        }

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        self.fc1.weight[...] = state_dict["fc1_weight"]
        self.fc1.bias[...] = state_dict["fc1_bias"]
        self.fc2.weight[...] = state_dict["fc2_weight"]
        self.fc2.bias[...] = state_dict["fc2_bias"]
        self.fc3.weight[...] = state_dict["fc3_weight"]
        self.fc3.bias[...] = state_dict["fc3_bias"]


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch_indices = np.arange(targets.shape[0])
    loss = -np.log(probs[batch_indices, targets] + 1e-12).mean()
    grad = probs
    grad[batch_indices, targets] -= 1.0
    return float(loss), grad.astype(np.float32)


def compute_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == targets).mean())
