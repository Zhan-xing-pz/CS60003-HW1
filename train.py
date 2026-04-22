from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.evaluate import evaluate
from src.hw1_mlp.model import MLPClassifier
from src.hw1_mlp.trainer import TrainConfig, train_model
from src.hw1_mlp.visualize import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Single training run with fixed hyperparameters.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs/single")
    parser.add_argument("--hidden-dim1", type=int, default=512)
    parser.add_argument("--hidden-dim2", type=int, default=256)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data = make_splits(args.data_root, seed=args.seed)

    config = TrainConfig(
        output_dir=str(output_dir),
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        activation=args.activation,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    model = MLPClassifier(
        input_dim=split_data.x_train.shape[1],
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        num_classes=len(split_data.class_names),
        activation=args.activation,
        seed=args.seed,
    )

    history, _ = train_model(
        model,
        split_data.x_train,
        split_data.y_train,
        split_data.x_val,
        split_data.y_val,
        config,
    )
    plot_training_curves(history, output_dir)

    _, val_acc, _ = evaluate(model, split_data.x_val, split_data.y_val)
    print(f"val_accuracy: {val_acc:.4f}")
    print(f"weights saved to: {output_dir / 'best_model.npz'}")


if __name__ == "__main__":
    main()
