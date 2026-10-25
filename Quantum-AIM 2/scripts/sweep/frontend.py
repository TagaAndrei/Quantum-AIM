"""Random search over the classical frontend. The circuit is not trained.

Run from the project root::

    python -m scripts.sweep.frontend --trials 150
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from data.creditcard import load_transactions, prepare_splits
from models.runtime import as_quantum
from models.two_phase import TwoPhaseConfig, TwoPhaseVQC, feature_map_coordinates
from paths import RESULTS
from scripts.visualize.angle_separation import plot_best_sweep

SEARCH_SPACE = {
    "hidden_size": (4, 8, 16, 32, 64),
    "mlp_epochs": (20, 40, 80, 120),
    "mlp_learning_rate": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "mlp_weight_decay": (0.0, 1e-5, 1e-4, 1e-3),
    "separation_weight": (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    "spread_weight": (0.001, 0.01, 0.05, 0.1),
    "separation_margin": (0.5, 1.0, 2.0, 4.0),
    "spread_target_std": (0.05, 0.15, 0.25, 0.5),
    "mlp_batch_size": (16, 32, 64, 128),
}

DEFAULT_CONFIG = TwoPhaseConfig()
DEFAULT_OUTPUT = RESULTS / "frontend_sweep.csv"
DEFAULT_PLOT = RESULTS / "frontend_sweep_best.png"


def config_candidates(trials: int, seed: int) -> list[TwoPhaseConfig]:
    parameter_names = tuple(SEARCH_SPACE)
    candidate_count = int(np.prod([len(SEARCH_SPACE[name]) for name in parameter_names]))

    def signature(values: dict) -> tuple:
        return tuple(values[name] for name in parameter_names)

    default_values = {
        "hidden_size": DEFAULT_CONFIG.hidden_size,
        "mlp_epochs": DEFAULT_CONFIG.mlp_epochs,
        "mlp_learning_rate": DEFAULT_CONFIG.mlp_learning_rate,
        "mlp_weight_decay": DEFAULT_CONFIG.mlp_weight_decay,
        "separation_weight": DEFAULT_CONFIG.separation_weight,
        "spread_weight": DEFAULT_CONFIG.spread_weight,
        "separation_margin": DEFAULT_CONFIG.separation_margin,
        "spread_target_std": DEFAULT_CONFIG.spread_target_std,
        "mlp_batch_size": DEFAULT_CONFIG.mlp_batch_size,
    }
    default_signature = signature(default_values)
    candidates = [default_values]
    if trials > candidate_count:
        raise ValueError(f"trials must be at most {candidate_count}.")
    rng = random.Random(seed)
    seen = {default_signature}
    while len(candidates) < trials:
        values = {name: rng.choice(SEARCH_SPACE[name]) for name in parameter_names}
        if signature(values) not in seen:
            candidates.append(values)
            seen.add(signature(values))

    configs = []
    for values in candidates:
        configs.append(
            replace(
                DEFAULT_CONFIG,
                hidden_size=values["hidden_size"],
                mlp_epochs=values["mlp_epochs"],
                mlp_learning_rate=values["mlp_learning_rate"],
                mlp_weight_decay=values["mlp_weight_decay"],
                mlp_batch_size=values["mlp_batch_size"],
                separation_weight=values["separation_weight"],
                spread_weight=values["spread_weight"],
                separation_margin=values["separation_margin"],
                spread_target_std=values["spread_target_std"],
            )
        )
    return configs


def prototype_balanced_accuracy(
    model: TwoPhaseVQC,
    features_train: pd.DataFrame,
    labels_train: np.ndarray,
    features_valid: pd.DataFrame,
    labels_valid: np.ndarray,
) -> float:
    train_angles = as_quantum(model.encode(features_train))
    valid_angles = as_quantum(model.encode(features_valid))
    connectivity = model.config.vqc.connectivity
    train_rotation, train_pairs = feature_map_coordinates(train_angles, connectivity)
    valid_rotation, valid_pairs = feature_map_coordinates(valid_angles, connectivity)
    train_rotation = train_rotation.cpu().numpy()
    train_pairs = train_pairs.cpu().numpy()
    valid_rotation = valid_rotation.cpu().numpy()
    valid_pairs = valid_pairs.cpu().numpy()

    rotation_prototypes = np.stack([train_rotation[labels_train == label].mean(axis=0) for label in (0, 1)])
    pair_prototypes = np.stack([train_pairs[labels_train == label].mean(axis=0) for label in (0, 1)])
    rotation_distances = np.square(valid_rotation[:, None, :] - rotation_prototypes[None, :, :]).mean(axis=-1)
    pair_distances = np.square(valid_pairs[:, None, :] - pair_prototypes[None, :, :]).mean(axis=-1)
    predictions = np.argmin(0.5 * (rotation_distances + pair_distances), axis=1)
    recalls = [np.mean(predictions[labels_valid == label] == label) for label in (0, 1)]
    return float(np.mean(recalls))


def evaluate_config(
    config: TwoPhaseConfig,
    seeds: list[int],
    features_train: pd.DataFrame,
    labels_train: np.ndarray,
    features_valid: pd.DataFrame,
    labels_valid: np.ndarray,
) -> dict:
    validation_auprc = []
    validation_boundary_accuracy = []
    prototype_accuracy = []
    best_epochs = []

    for seed in seeds:
        model = TwoPhaseVQC(config)
        history = model.fit_frontend(
            features_train,
            labels_train,
            features_valid,
            labels_valid,
            seed=seed,
            verbose=False,
        )
        best_row = history.loc[history["validation AUPRC"].idxmax()]
        validation_auprc.append(float(best_row["validation AUPRC"]))
        best_epochs.append(int(best_row["epoch"]))
        valid_probabilities = model.predict_head(features_valid)
        validation_boundary_accuracy.append(
            float(balanced_accuracy_score(labels_valid, valid_probabilities >= 0.5))
        )
        prototype_accuracy.append(
            prototype_balanced_accuracy(model, features_train, labels_train, features_valid, labels_valid)
        )

    return {
        "hidden_size": config.hidden_size,
        "mlp_epochs": config.mlp_epochs,
        "mlp_learning_rate": config.mlp_learning_rate,
        "mlp_weight_decay": config.mlp_weight_decay,
        "separation_weight": config.separation_weight,
        "spread_weight": config.spread_weight,
        "separation_margin": config.separation_margin,
        "spread_target_std": config.spread_target_std,
        "mlp_batch_size": config.mlp_batch_size,
        "validation_AUPRC_mean": float(np.mean(validation_auprc)),
        "validation_AUPRC_std": float(np.std(validation_auprc)),
        "classifier_boundary_balanced_accuracy_mean": float(np.mean(validation_boundary_accuracy)),
        "classifier_boundary_balanced_accuracy_std": float(np.std(validation_boundary_accuracy)),
        "prototype_balanced_accuracy_mean": float(np.mean(prototype_accuracy)),
        "prototype_balanced_accuracy_std": float(np.std(prototype_accuracy)),
        "best_epoch_mean": float(np.mean(best_epochs)),
        "seeds": ",".join(map(str, seeds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the classical MLP frontend; rank decision-boundary quality on validation."
    )
    parser.add_argument("--data", type=Path, help="CSV path (defaults to data/creditcard.csv).")
    parser.add_argument("--trials", type=int, default=150, help="Candidates, including current defaults.")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    configs = config_candidates(args.trials, args.search_seed)
    x_train, y_train, x_valid, y_valid, _, _ = prepare_splits(load_transactions(args.data), seed=42)
    rows = []
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] hidden={config.hidden_size}, "
            f"lr={config.mlp_learning_rate:g}, separation={config.separation_weight:g}, "
            f"spread={config.spread_weight:g}, epochs={config.mlp_epochs}, "
            f"batch={config.mlp_batch_size}",
            flush=True,
        )
        rows.append(evaluate_config(config, args.seeds, x_train, y_train, x_valid, y_valid))

    results = pd.DataFrame(rows).sort_values(
        ["classifier_boundary_balanced_accuracy_mean", "validation_AUPRC_mean"],
        ascending=False,
        kind="stable",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\nSweep results, ranked by validation classifier-boundary balanced accuracy:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(results)} candidates to {args.output.resolve()}")
    print("\nRecommended configuration:")
    print(results.iloc[0].to_string())
    best_config = configs[int(results.index[0])]
    plot_best_sweep(
        best_config,
        x_train,
        y_train,
        x_valid,
        y_valid,
        seed=args.seeds[0],
        output_path=args.plot_output,
    )


if __name__ == "__main__":
    main()
