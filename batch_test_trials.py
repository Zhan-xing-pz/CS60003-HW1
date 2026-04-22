from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.hw1_mlp.data import make_splits
from src.hw1_mlp.evaluate import build_confusion_matrix, evaluate
from src.hw1_mlp.model import MLPClassifier
from src.hw1_mlp.visualize import plot_confusion_matrix


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def discover_trial_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("trial_")])


def build_model_from_config(split_data, config: dict, seed: int) -> MLPClassifier:
    return MLPClassifier(
        input_dim=split_data.x_test.shape[1],
        hidden_dim1=int(config["hidden_dim1"]),
        hidden_dim2=int(config["hidden_dim2"]),
        num_classes=len(split_data.class_names),
        activation=str(config["activation"]),
        seed=seed,
    )


def evaluate_trial(trial_dir: Path, split_data, seed: int, save_confusion: bool) -> dict:
    config_path = trial_dir / "train_config.json"
    weights_path = trial_dir / "best_model.npz"
    summary_path = trial_dir / "summary.json"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing files in {trial_dir}")

    config = load_json(config_path)
    model = build_model_from_config(split_data, config, seed)
    state_dict = dict(np.load(weights_path))
    model.load_state_dict(state_dict)

    val_loss, val_acc, _ = evaluate(model, split_data.x_val, split_data.y_val)
    test_loss, test_acc, preds = evaluate(model, split_data.x_test, split_data.y_test)

    result = {
        "trial_name": trial_dir.name,
        "trial_dir": str(trial_dir),
        "hidden_dim1": int(config["hidden_dim1"]),
        "hidden_dim2": int(config["hidden_dim2"]),
        "activation": str(config["activation"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "epochs": int(config["epochs"]),
        "lr_decay": float(config["lr_decay"]),
        "val_loss_recomputed": float(val_loss),
        "val_accuracy_recomputed": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    if summary_path.exists():
        summary = load_json(summary_path)
        result["val_loss_saved"] = float(summary.get("val_loss", val_loss))
        result["val_accuracy_saved"] = float(summary.get("val_accuracy", val_acc))

    if save_confusion:
        cm = build_confusion_matrix(split_data.y_test, preds)
        plot_confusion_matrix(cm, split_data.class_names, trial_dir / "confusion_matrix_test.png")

    with open(trial_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def save_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sorted_test_accuracy(rows: list[dict], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["test_accuracy"], reverse=True)
    labels = [row["trial_name"] for row in sorted_rows]
    scores = [row["test_accuracy"] for row in sorted_rows]

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(scores)), scores)
    plt.xticks(np.arange(len(scores)), labels, rotation=75, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy of All Trials")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_val_test_scatter(rows: list[dict], output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for row in rows:
        plt.scatter(row["val_accuracy_recomputed"], row["test_accuracy"], s=60)
        plt.text(row["val_accuracy_recomputed"], row["test_accuracy"], row["trial_name"], fontsize=7)
    plt.xlabel("Validation Accuracy")
    plt.ylabel("Test Accuracy")
    plt.title("Validation vs Test Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def unique_preserve_order(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def plot_heatmaps(rows: list[dict], output_dir: Path) -> None:
    hidden_pairs = sorted({(row["hidden_dim1"], row["hidden_dim2"]) for row in rows})
    for hidden_dim1, hidden_dim2 in hidden_pairs:
        subset = [
            row for row in rows
            if row["hidden_dim1"] == hidden_dim1 and row["hidden_dim2"] == hidden_dim2
        ]
        if not subset:
            continue

        learning_rates = unique_preserve_order(sorted([row["learning_rate"] for row in subset], reverse=True))
        weight_decays = unique_preserve_order(sorted([row["weight_decay"] for row in subset]))
        matrix = np.full((len(learning_rates), len(weight_decays)), np.nan, dtype=np.float32)

        for row in subset:
            i = learning_rates.index(row["learning_rate"])
            j = weight_decays.index(row["weight_decay"])
            matrix[i, j] = row["test_accuracy"]

        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Test Accuracy")
        plt.xticks(np.arange(len(weight_decays)), [f"{wd:.0e}" for wd in weight_decays])
        plt.yticks(np.arange(len(learning_rates)), [str(lr) for lr in learning_rates])
        plt.xlabel("Weight Decay")
        plt.ylabel("Learning Rate")
        plt.title(f"Test Accuracy Heatmap (h1={hidden_dim1}, h2={hidden_dim2})")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_h{hidden_dim1}_{hidden_dim2}.png", dpi=200)
        plt.close()


def save_markdown_summary(rows: list[dict], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["test_accuracy"], reverse=True)
    lines = [
        "# Trial Test Summary",
        "",
        "| Rank | Trial | h1 | h2 | lr | weight_decay | Val Acc | Test Acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(sorted_rows, start=1):
        lines.append(
            f"| {idx} | {row['trial_name']} | {row['hidden_dim1']} | {row['hidden_dim2']} | "
            f"{row['learning_rate']} | {row['weight_decay']} | "
            f"{row['val_accuracy_recomputed']:.4f} | {row['test_accuracy']:.4f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test all trial checkpoints and summarize their performance.")
    parser.add_argument("--data-root", default="EuroSAT_RGB")
    parser.add_argument("--trials-root", default="outputs")
    parser.add_argument("--summary-dir", default="outputs/batch_test_summary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-confusion", action="store_true", help="Save one test confusion matrix per trial.")
    args = parser.parse_args()

    trials_root = Path(args.trials_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    trial_dirs = discover_trial_dirs(trials_root)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial_* directories found under {trials_root}")

    split_data = make_splits(args.data_root, seed=args.seed)
    rows = []
    for trial_dir in tqdm(trial_dirs, desc="Batch testing trials", unit="trial"):
        rows.append(evaluate_trial(trial_dir, split_data, args.seed, args.save_confusion))

    rows_sorted = sorted(rows, key=lambda row: row["test_accuracy"], reverse=True)
    with open(summary_dir / "trial_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)
    save_csv(rows_sorted, summary_dir / "trial_test_summary.csv")
    save_markdown_summary(rows_sorted, summary_dir / "trial_test_summary.md")
    plot_sorted_test_accuracy(rows_sorted, summary_dir / "sorted_test_accuracy.png")
    plot_val_test_scatter(rows_sorted, summary_dir / "val_vs_test_scatter.png")
    plot_heatmaps(rows_sorted, summary_dir)

    best = rows_sorted[0]
    print(f"Evaluated {len(rows_sorted)} trials.")
    print(f"Best test accuracy: {best['test_accuracy']:.4f}")
    print(f"Best trial: {best['trial_name']}")
    print(f"Summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
