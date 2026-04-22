from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

import numpy as np

from .data import SplitData
from .evaluate import evaluate
from .model import MLPClassifier
from .trainer import TrainConfig, train_model
from .visualize import plot_training_curves


def search_space() -> list[dict[str, float | int | str]]:
    learning_rates = [0.03, 0.01, 0.003]
    hidden_dims = [(256, 128), (512, 256), (1024, 512)]
    weight_decays = [1e-3, 1e-4, 1e-5]

    configs = []
    for lr in learning_rates:
        for h1, h2 in hidden_dims:
            for wd in weight_decays:
                configs.append({
                    "hidden_dim1": h1,
                    "hidden_dim2": h2,
                    "activation": "relu",
                    "learning_rate": lr,
                    "weight_decay": wd,
                })
    return configs


def run_trial(
    split_data: SplitData,
    base_output_dir: Path,
    trial_id: int,
    hidden_dim1: int,
    hidden_dim2: int,
    activation: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    lr_decay: float,
    seed: int,
) -> tuple[dict, MLPClassifier]:
    trial_dir = base_output_dir / f"trial_{trial_id:02d}"
    config = TrainConfig(
        output_dir=str(trial_dir),
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        activation=activation,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )
    model = MLPClassifier(
        input_dim=split_data.x_train.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=len(split_data.class_names),
        activation=activation,
        seed=seed,
    )
    history, _ = train_model(
        model,
        split_data.x_train,
        split_data.y_train,
        split_data.x_val,
        split_data.y_val,
        config,
    )
    plot_training_curves(history, trial_dir)
    val_loss, val_acc, _ = evaluate(model, split_data.x_val, split_data.y_val)
    result = {
        "trial_dir": str(trial_dir),
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "activation": activation,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    with open(trial_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result, model


def run_search(
    split_data: SplitData,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr_decay: float,
    seed: int,
    max_trials: int | None = None,
) -> tuple[list[dict], dict, MLPClassifier]:
    trial_configs = search_space()
    if max_trials is not None:
        trial_configs = trial_configs[:max_trials]

    all_results = []
    best_model = None
    best_result = None

    for trial_id, params in tqdm(enumerate(trial_configs, start=1), total=len(trial_configs), desc="Trials", unit="trial"):
        tqdm.write(f"\n--- Trial {trial_id}/{len(trial_configs)}: {params} ---")
        result, model = run_trial(
            split_data=split_data,
            base_output_dir=output_dir,
            trial_id=trial_id,
            epochs=epochs,
            batch_size=batch_size,
            lr_decay=lr_decay,
            seed=seed,
            **params,
        )
        all_results.append(result)
        if best_result is None or result["val_accuracy"] > best_result["val_accuracy"]:
            best_result = result
            best_model = model

    if best_result is None or best_model is None:
        raise RuntimeError("No trial completed.")

    return all_results, best_result, best_model
