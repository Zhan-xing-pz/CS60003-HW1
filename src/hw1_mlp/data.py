from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    image_shape: tuple[int, int, int]


def load_eurosat(root: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, int, int]]:
    root = Path(root)
    class_names = sorted([path.name for path in root.iterdir() if path.is_dir()])
    images: list[np.ndarray] = []
    labels: list[int] = []
    image_shape: tuple[int, int, int] | None = None

    for class_idx, class_name in enumerate(class_names):
        for image_path in sorted((root / class_name).glob("*.jpg")):
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
            if image_shape is None:
                image_shape = tuple(image.shape)
            images.append(image.reshape(-1))
            labels.append(class_idx)

    if image_shape is None:
        raise ValueError(f"No images found under {root}")

    return np.stack(images), np.asarray(labels, dtype=np.int64), class_names, image_shape


def standardize_from_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


def make_splits(
    root: str | Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    standardize: bool = True,
) -> SplitData:
    x, y, class_names, image_shape = load_eurosat(root)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=y_train_val,
    )
    if standardize:
        x_train, x_val, x_test = standardize_from_train(x_train, x_val, x_test)
    return SplitData(
        x_train=x_train.astype(np.float32),
        y_train=y_train,
        x_val=x_val.astype(np.float32),
        y_val=y_val,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
        class_names=class_names,
        image_shape=image_shape,
    )
