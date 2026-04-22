from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.search import run_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search over predefined configurations.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data = make_splits(args.data_root, seed=args.seed)

    all_results, best_result, _ = run_search(
        split_data=split_data,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_decay=args.lr_decay,
        seed=args.seed,
        max_trials=args.max_trials,
    )

    with open(output_dir / "search_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    best_weights = Path(best_result["trial_dir"]) / "best_model.npz"
    print(f"best val_accuracy: {best_result['val_accuracy']:.4f}")
    print(f"best weights: {best_weights}")
    print(f"run test:  python test.py --data-root {args.data_root} --weights {best_weights} --output-dir {output_dir}")


if __name__ == "__main__":
    main()
