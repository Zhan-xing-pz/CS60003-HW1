from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.evaluate import build_confusion_matrix, evaluate
from src.hw1_mlp.model import MLPClassifier
from src.hw1_mlp.visualize import plot_confusion_matrix


def load_config(weights_path: Path) -> dict:
    config_path = weights_path.parent / "train_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test set.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--weights", required=True, help="Path to best_model.npz")
    parser.add_argument("--output-dir", default=None, help="Where to save confusion_matrix.png (default: same dir as weights)")
    parser.add_argument("--hidden-dim1", type=int, default=None)
    parser.add_argument("--hidden-dim2", type=int, default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # auto-load model config from train_config.json if not specified
    saved_config = load_config(weights_path)
    hidden_dim1 = args.hidden_dim1 or saved_config.get("hidden_dim1")
    hidden_dim2 = args.hidden_dim2 or saved_config.get("hidden_dim2")
    activation = args.activation or saved_config.get("activation")

    if hidden_dim1 is None or hidden_dim2 is None or activation is None:
        parser.error(
            "Could not find train_config.json next to weights. "
            "Please provide --hidden-dim1, --hidden-dim2, --activation explicitly."
        )

    output_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = ["Loading dataset", "Building model", "Running inference", "Computing metrics"]
    with tqdm(steps, desc="Testing", unit="step") as pbar:
        pbar.set_description("Loading dataset")
        split_data = make_splits(args.data_root, seed=args.seed)
        pbar.update(1)

        pbar.set_description("Building model")
        model = MLPClassifier(
            input_dim=split_data.x_test.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            num_classes=len(split_data.class_names),
            activation=activation,
            seed=args.seed,
        )
        state_dict = dict(np.load(weights_path))
        model.load_state_dict(state_dict)
        pbar.update(1)

        pbar.set_description("Running inference")
        test_loss, test_acc, preds = evaluate(model, split_data.x_test, split_data.y_test)
        pbar.update(1)

        pbar.set_description("Computing metrics")
        cm = build_confusion_matrix(split_data.y_test, preds)
        pbar.update(1)

    print(f"Test Accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")

    plot_confusion_matrix(cm, split_data.class_names, output_dir / "confusion_matrix.png")
    print(f"confusion_matrix.png saved to: {output_dir}")


if __name__ == "__main__":
    main()
