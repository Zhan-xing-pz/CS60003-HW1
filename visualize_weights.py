from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.model import MLPClassifier
from src.hw1_mlp.visualize import save_weight_visualizations


def load_config(weights_path: Path) -> dict:
    config_path = weights_path.parent / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"train_config.json not found next to {weights_path}")
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize first-layer weights of a trained MLP as image patterns.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--weights", required=True, help="Path to best_model.npz")
    parser.add_argument("--output-dir", default=None, help="Default: same directory as weights")
    parser.add_argument("--max-filters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = ["Loading config", "Loading dataset metadata", "Building model", "Saving visualization"]
    with tqdm(steps, desc="Weight visualization", unit="step") as pbar:
        pbar.set_description("Loading config")
        config = load_config(weights_path)
        pbar.update(1)

        pbar.set_description("Loading dataset metadata")
        split_data = make_splits(args.data_root, seed=args.seed)
        pbar.update(1)

        pbar.set_description("Building model")
        model = MLPClassifier(
            input_dim=split_data.x_train.shape[1],
            hidden_dim1=int(config["hidden_dim1"]),
            hidden_dim2=int(config["hidden_dim2"]),
            num_classes=len(split_data.class_names),
            activation=str(config["activation"]),
            seed=args.seed,
        )
        model.load_state_dict(dict(np.load(weights_path)))
        pbar.update(1)

        pbar.set_description("Saving visualization")
        save_weight_visualizations(
            model.fc1.weight,
            split_data.image_shape,
            output_dir / "first_layer_weights.png",
            max_filters=args.max_filters,
        )
        pbar.update(1)

    summary = {
        "weights": str(weights_path),
        "output_image": str(output_dir / "first_layer_weights.png"),
        "hidden_dim1": int(config["hidden_dim1"]),
        "hidden_dim2": int(config["hidden_dim2"]),
        "activation": str(config["activation"]),
        "max_filters": args.max_filters,
    }
    with open(output_dir / "weight_visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved weight visualization to: {output_dir / 'first_layer_weights.png'}")


if __name__ == "__main__":
    main()
