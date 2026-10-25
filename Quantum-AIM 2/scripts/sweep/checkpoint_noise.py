"""Noise and cost sweep over every saved ``model.pth``.

For each checkpoint under ``models/trained``:

1. Score validation and test under depolarizing noise from ``p = 0`` through
   ``p = 0.5`` (default grid: clean, 0.001, then steps up to 0.5).
2. Write the full computational cost of that model: Adam parameter-shift
   ledger, genetic forwards when the name or checkpoint stores a GA, frontend
   size, and noise locations.

Run from the project root::

    python -m scripts.sweep.checkpoint_noise
    python -m scripts.runs.run_grid --experiments checkpoint_noise
    python -m scripts.sweep.checkpoint_noise --noise-step 0.01
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from data.creditcard import fixed_split, load_transactions, undersample
from evolutionary import GeneticConfig, forward_executions
from paths import RESULTS, TRAINED_MODELS
from scripts.runs.checkpoints import LoadedCheckpoint, discover_checkpoints, load_checkpoint
from scripts.runs.evaluation import (
    SHOTS_PER_EXECUTION,
    append_rows,
    choose_threshold,
    cost_ledger,
    evaluate,
    evaluate_under_noise,
    noise_grid,
)

DEFAULT_METRICS = RESULTS / "checkpoint_noise_metrics.csv"
DEFAULT_COST = RESULTS / "checkpoint_cost.csv"
GENETIC_NAME = re.compile(
    r"pop(?P<pop>\d+)_gen(?P<gen>\d+)_mut(?P<mut>[0-9.]+)_elite(?P<elite>\d+)"
)


def genetic_from_checkpoint(loaded: LoadedCheckpoint) -> GeneticConfig | None:
    if loaded.genetic is not None:
        return loaded.genetic
    match = GENETIC_NAME.search(loaded.name)
    if match is None:
        return None
    return GeneticConfig(
        population_size=int(match.group("pop")),
        generations=int(match.group("gen")),
        mutation_std=float(match.group("mut")),
        elite_count=int(match.group("elite")),
    )


def finished_metric_cells(path: Path, levels: tuple[float, ...]) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    required = set(levels)
    done = set()
    for (model, seed), group in frame.groupby(["model", "seed"]):
        if required <= set(group["p"].astype(float)):
            done.add((str(model), int(seed)))
    return done


def finished_cost_cells(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    return {(str(row.model), int(row.seed)) for row in frame.itertuples()}


def encode_splits(loaded: LoadedCheckpoint, splits, seed: int):
    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    x_balanced, y_balanced = undersample(x_train, y_train, seed)
    return {
        "x_balanced": x_balanced,
        "x_valid": x_valid,
        "x_test": x_test,
        "train": loaded.encode(x_balanced),
        "valid": loaded.encode(x_valid),
        "test": loaded.encode(x_test),
        "y_balanced": y_balanced,
        "y_valid": y_valid,
        "y_test": y_test,
        "n_train": len(y_balanced),
        "n_valid": len(y_valid),
        "n_test": len(y_test),
    }


def cost_row(loaded: LoadedCheckpoint, encoded: dict) -> dict:
    genetic = genetic_from_checkpoint(loaded)
    if loaded.kind == "head" or loaded.circuit is None:
        frontend_epochs = getattr(loaded.config, "mlp_epochs", 0)
        frontend_parameters = 0
        if loaded.head is not None and loaded.head.encoder.frontend is not None:
            frontend_parameters = sum(
                parameter.numel() for parameter in loaded.head.encoder.frontend.parameters()
            )
        return {
            "model": loaded.name,
            "seed": loaded.seed,
            "kind": loaded.kind,
            "path": str(loaded.path),
            "n_parameters": 5,
            "epochs": frontend_epochs,
            "executions_per_sample": 0,
            "training_executions": 0,
            "validation_executions": 0,
            "test_executions": 0,
            "circuit_executions": 0,
            "genetic_forward_executions": 0,
            "circuit_executions_total": 0,
            "shots": 0,
            "shots_total": 0,
            "noise_locations": 0,
            "frontend_parameters": frontend_parameters,
            "n_train": encoded["n_train"],
            "n_valid": encoded["n_valid"],
            "n_test": encoded["n_test"],
            "population_size": None,
            "generations": None,
            "mutation_std": None,
            "elite_count": None,
        }

    circuit = loaded.circuit
    config = circuit.config
    adam = cost_ledger(
        n_train=encoded["n_train"],
        n_valid=encoded["n_valid"],
        n_test=encoded["n_test"],
        n_parameters=config.n_parameters,
        epochs=config.epochs,
    )
    ga_forwards = 0
    if genetic is not None:
        ga_forwards = forward_executions(genetic, encoded["n_train"], encoded["n_valid"])
    sample = torch.as_tensor(encoded["train"][0], dtype=torch.float64)
    weights = loaded.weights.detach().cpu() if loaded.weights is not None else None
    noise_locations = circuit.noise_locations(sample, weights)
    frontend_parameters = 0
    if hasattr(loaded.config, "hidden_size"):
        # Approximate AngleMLP size: 30→H, H→4, 4→1.
        hidden = int(loaded.config.hidden_size)
        frontend_parameters = 30 * hidden + hidden + hidden * 4 + 4 + 4 + 1
    total = adam["circuit_executions"] + ga_forwards
    return {
        "model": loaded.name,
        "seed": loaded.seed,
        "kind": loaded.kind,
        "path": str(loaded.path),
        **adam,
        "genetic_forward_executions": ga_forwards,
        "circuit_executions_total": total,
        "shots_total": total * SHOTS_PER_EXECUTION,
        "noise_locations": noise_locations,
        "frontend_parameters": frontend_parameters,
        "n_train": encoded["n_train"],
        "n_valid": encoded["n_valid"],
        "n_test": encoded["n_test"],
        "population_size": None if genetic is None else genetic.population_size,
        "generations": None if genetic is None else genetic.generations,
        "mutation_std": None if genetic is None else genetic.mutation_std,
        "elite_count": None if genetic is None else genetic.elite_count,
    }


def score_head(loaded: LoadedCheckpoint, encoded: dict) -> list[dict]:
    if loaded.head is None:
        raise RuntimeError(f"{loaded.name} seed {loaded.seed} has no classification head.")
    valid_scores = loaded.head.predict(encoded["x_valid"])
    test_scores = loaded.head.predict(encoded["x_test"])
    threshold = choose_threshold(encoded["y_valid"], valid_scores)
    return [
        {
            "model": loaded.name,
            "kind": "head",
            "seed": loaded.seed,
            "path": str(loaded.path),
            "p": 0.0,
            **evaluate(encoded["y_test"], test_scores, threshold),
            "Spearman rho": float("nan"),
            "threshold": threshold,
        }
    ]


def score_circuit(loaded: LoadedCheckpoint, encoded: dict, levels: tuple[float, ...]) -> pd.DataFrame:
    if loaded.circuit is None or loaded.weights is None:
        raise RuntimeError(f"{loaded.name} seed {loaded.seed} has no circuit weights.")
    noise = evaluate_under_noise(
        loaded.circuit,
        loaded.weights,
        encoded["valid"],
        encoded["test"],
        encoded["y_valid"],
        encoded["y_test"],
        levels=levels,
    )
    noise.insert(0, "seed", loaded.seed)
    noise.insert(0, "kind", loaded.kind)
    noise.insert(0, "model", loaded.name)
    noise.insert(0, "path", str(loaded.path))
    return noise


def sweep_checkpoints(
    seeds: list[int] | None = None,
    root: Path | None = None,
    metrics_path: Path | None = None,
    cost_path: Path | None = None,
    noise_step: float | None = None,
    names: list[str] | None = None,
) -> None:
    metrics_path = metrics_path or DEFAULT_METRICS
    cost_path = cost_path or DEFAULT_COST
    levels = noise_grid(step=noise_step)
    paths = discover_checkpoints(root or TRAINED_MODELS)
    if names:
        allowed = set(names)
        paths = [path for path in paths if path.parent.parent.name in allowed]
    if not paths:
        print(f"No model.pth files under {root or TRAINED_MODELS}", flush=True)
        return

    splits = fixed_split(load_transactions())
    done_metrics = finished_metric_cells(metrics_path, levels)
    done_cost = finished_cost_cells(cost_path)
    seed_filter = None if seeds is None else set(seeds)
    print(
        f"checkpoint noise sweep: {len(paths)} files, p in {levels[0]}..{levels[-1]} "
        f"({len(levels)} levels), writing {metrics_path} and {cost_path}",
        flush=True,
    )

    for path in paths:
        try:
            loaded = load_checkpoint(path)
        except Exception as error:  # noqa: BLE001 - keep sweeping the rest
            print(f"skip {path}: failed to load ({error})", flush=True)
            continue
        if seed_filter is not None and loaded.seed not in seed_filter:
            continue

        need_metrics = (loaded.name, loaded.seed) not in done_metrics
        need_cost = (loaded.name, loaded.seed) not in done_cost
        if not need_metrics and not need_cost:
            print(f"skip {loaded.name} seed {loaded.seed}: already scored", flush=True)
            continue

        print(f"\n{loaded.name} seed {loaded.seed}: {path}", flush=True)
        started = time.time()
        encoded = encode_splits(loaded, splits, loaded.seed)

        if need_cost:
            append_rows(cost_path, [cost_row(loaded, encoded)])
            done_cost.add((loaded.name, loaded.seed))
            print(f"{loaded.name} seed {loaded.seed}: cost row written", flush=True)

        if need_metrics:
            if loaded.kind == "head":
                rows = score_head(loaded, encoded)
                append_rows(metrics_path, rows)
            else:
                noise = score_circuit(loaded, encoded, levels)
                append_rows(metrics_path, noise.to_dict("records"))
                for probability in (0.0, 0.01, 0.05, 0.5):
                    if any(np.isclose(level, probability) for level in levels):
                        score = float(
                            noise.loc[np.isclose(noise["p"].astype(float), probability), "AUPRC"].iloc[0]
                        )
                        print(
                            f"{loaded.name} seed {loaded.seed}: "
                            f"test AUPRC at p={probability:.3f} is {score:.3f}",
                            flush=True,
                        )
            done_metrics.add((loaded.name, loaded.seed))
            print(f"{loaded.name} seed {loaded.seed}: metrics in {time.time() - started:.0f}s", flush=True)

    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        summary = (
            frame.groupby(["model", "p"])["AUPRC"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "AUPRC mean", "std": "AUPRC std", "count": "seeds"})
            .reset_index()
        )
        print("\nCheckpoint noise AUPRC summary")
        print(summary.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score every saved model.pth under a dense noise grid and write its cost."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Limit to these seeds.")
    parser.add_argument("--root", type=Path, default=TRAINED_MODELS)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument(
        "--noise-step",
        type=float,
        default=None,
        help="If set, use 0.0 plus p from 0.001 to 0.5 in this step. Default is the fixed extended grid.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Only these experiment folder names under models/trained.",
    )
    args = parser.parse_args()
    sweep_checkpoints(
        seeds=args.seeds,
        root=args.root,
        metrics_path=args.metrics_output,
        cost_path=args.cost_output,
        noise_step=args.noise_step,
        names=args.names,
    )


if __name__ == "__main__":
    main()
