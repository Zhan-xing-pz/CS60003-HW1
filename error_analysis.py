from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.evaluate import evaluate
from src.hw1_mlp.model import MLPClassifier


def load_config(weights_path: Path) -> dict:
    config_path = weights_path.parent / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"train_config.json not found next to {weights_path}")
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)


def save_misclassified_gallery(
    raw_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    image_shape: tuple[int, int, int],
    output_path: Path,
    max_examples: int,
) -> list[dict[str, int | str]]:
    misclassified = np.where(y_pred != y_true)[0][:max_examples]
    examples: list[dict[str, int | str]] = []
    if len(misclassified) == 0:
        return examples

    cols = 4
    rows = int(np.ceil(len(misclassified) / cols))
    plt.figure(figsize=(3.8 * cols, 3.8 * rows))
    for plot_idx, sample_idx in enumerate(misclassified, start=1):
        image = raw_images[sample_idx].reshape(image_shape)
        true_name = class_names[int(y_true[sample_idx])]
        pred_name = class_names[int(y_pred[sample_idx])]
        examples.append(
            {
                "test_index": int(sample_idx),
                "true_label": true_name,
                "pred_label": pred_name,
            }
        )
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(np.clip(image, 0.0, 1.0))
        plt.axis("off")
        plt.title(f"True: {true_name}\nPred: {pred_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return examples


def save_confusion_pair_chart(confusion_pairs: list[tuple[str, str, int]], output_path: Path, top_k: int = 10) -> None:
    top_pairs = confusion_pairs[:top_k]
    if not top_pairs:
        return
    labels = [f"{true}->{pred}" for true, pred, _ in top_pairs]
    counts = [count for _, _, count in top_pairs]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(counts)), counts)
    plt.xticks(np.arange(len(counts)), labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Top Misclassification Pairs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export misclassified test examples for error analysis.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--weights", required=True, help="Path to best_model.npz")
    parser.add_argument("--output-dir", default=None, help="Default: same directory as weights")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = ["Loading config", "Loading processed split", "Loading raw split", "Building model", "Running inference", "Saving analysis"]
    with tqdm(steps, desc="Error analysis", unit="step") as pbar:
        pbar.set_description("Loading config")
        config = load_config(weights_path)
        pbar.update(1)

        pbar.set_description("Loading processed split")
        split_data = make_splits(args.data_root, seed=args.seed, standardize=True)
        pbar.update(1)

        pbar.set_description("Loading raw split")
        raw_split_data = make_splits(args.data_root, seed=args.seed, standardize=False)
        pbar.update(1)

        pbar.set_description("Building model")
        model = MLPClassifier(
            input_dim=split_data.x_test.shape[1],
            hidden_dim1=int(config["hidden_dim1"]),
            hidden_dim2=int(config["hidden_dim2"]),
            num_classes=len(split_data.class_names),
            activation=str(config["activation"]),
            seed=args.seed,
        )
        model.load_state_dict(dict(np.load(weights_path)))
        pbar.update(1)

        pbar.set_description("Running inference")
        test_loss, test_acc, preds = evaluate(model, split_data.x_test, split_data.y_test)
        pbar.update(1)

        pbar.set_description("Saving analysis")
        examples = save_misclassified_gallery(
            raw_images=raw_split_data.x_test,
            y_true=split_data.y_test,
            y_pred=preds,
            class_names=split_data.class_names,
            image_shape=split_data.image_shape,
            output_path=output_dir / "error_cases.png",
            max_examples=args.max_examples,
        )

        pair_counter = Counter()
        for true_idx, pred_idx in zip(split_data.y_test, preds):
            if true_idx != pred_idx:
                pair_counter[(split_data.class_names[int(true_idx)], split_data.class_names[int(pred_idx)])] += 1
        confusion_pairs = [
            (true_name, pred_name, count)
            for (true_name, pred_name), count in pair_counter.most_common()
        ]
        save_confusion_pair_chart(confusion_pairs, output_dir / "top_confusion_pairs.png")
        pbar.update(1)

    summary = {
        "weights": str(weights_path),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_errors": int(np.sum(preds != split_data.y_test)),
        "saved_examples": examples,
        "top_confusion_pairs": [
            {"true_label": true_name, "pred_label": pred_name, "count": count}
            for true_name, pred_name, count in confusion_pairs[:10]
        ],
    }
    with open(output_dir / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved error gallery to: {output_dir / 'error_cases.png'}")
    print(f"Saved confusion-pair chart to: {output_dir / 'top_confusion_pairs.png'}")


if __name__ == "__main__":
    main()
