"""Hyperparameter sweep for ``two_phase_ga_adam_chained``.

Trains the two-phase frontend, runs a genetic search on the chain circuit, then
Adam fine-tunes. Each trial keeps its own CSV name and ``model.pth`` so finished
cells are not overwritten.

Run from the project root::

    python -m scripts.sweep.chain_ga_hparams --trials 8 --seeds 0
    python -m scripts.runs.run_grid --experiments two_phase_ga_adam_chained_hparams --trials 8 --seeds 0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch

from data.creditcard import fixed_split, load_transactions, undersample
from evolutionary import GeneticConfig, evolve_weights, forward_executions, genetic_tag
from models.two_phase import TwoPhaseConfig, TwoPhaseVQC
from optimizers import make_vqc_optimizer
from paths import RESULTS
from scripts.runs.checkpoints import save_two_phase
from scripts.runs.evaluation import HEADLINE_LEVELS, append_rows, cost_ledger, evaluate_under_noise
from scripts.runs.run_grid import chain_vqc, encoded_angles, setting_tag

DEFAULT_OUTPUT = RESULTS / "two_phase_ga_adam_chained_hparams.csv"
DEFAULT_COST = RESULTS / "two_phase_ga_adam_chained_hparams_cost.csv"
MODEL_PREFIX = "two_phase_ga_adam_chained"

SEARCH_SPACE = {
    "hidden_size": (8, 16, 32),
    "mlp_epochs": (20, 40),
    "mlp_learning_rate": (1e-3, 3e-3, 1e-2),
    "mlp_weight_decay": (0.0, 1e-4),
    "separation_weight": (0.25, 0.5, 1.0),
    "spread_weight": (0.001, 0.01),
    "separation_margin": (0.5, 1.0),
    "mlp_batch_size": (32, 64),
    "population_size": (8, 12, 16),
    "generations": (4, 6),
    "mutation_std": (0.1, 0.25),
    "elite_count": (2, 4),
    "vqc_learning_rate": (0.01, 0.05),
    "vqc_epochs": (20, 30),
}


def default_values() -> dict:
    frontend = TwoPhaseConfig()
    genetic = GeneticConfig()
    vqc = chain_vqc()
    return {
        "hidden_size": frontend.hidden_size,
        "mlp_epochs": frontend.mlp_epochs,
        "mlp_learning_rate": frontend.mlp_learning_rate,
        "mlp_weight_decay": frontend.mlp_weight_decay,
        "separation_weight": frontend.separation_weight,
        "spread_weight": frontend.spread_weight,
        "separation_margin": frontend.separation_margin,
        "mlp_batch_size": frontend.mlp_batch_size,
        "population_size": genetic.population_size,
        "generations": genetic.generations,
        "mutation_std": genetic.mutation_std,
        "elite_count": genetic.elite_count,
        "vqc_learning_rate": vqc.learning_rate,
        "vqc_epochs": vqc.epochs,
    }


def candidate_values(trials: int, search_seed: int) -> list[dict]:
    names = tuple(SEARCH_SPACE)
    defaults = default_values()

    def signature(values: dict) -> tuple:
        return tuple(values[name] for name in names)

    candidates = [defaults]
    seen = {signature(defaults)}
    if trials > 1:
        rng = random.Random(search_seed)
        while len(candidates) < trials:
            values = {name: rng.choice(SEARCH_SPACE[name]) for name in names}
            if values["elite_count"] >= values["population_size"]:
                continue
            if signature(values) not in seen:
                candidates.append(values)
                seen.add(signature(values))
    return candidates


def configs_from_values(values: dict) -> tuple[TwoPhaseConfig, GeneticConfig]:
    vqc = replace(
        chain_vqc(),
        learning_rate=float(values["vqc_learning_rate"]),
        epochs=int(values["vqc_epochs"]),
    )
    frontend = TwoPhaseConfig(
        vqc=vqc,
        hidden_size=int(values["hidden_size"]),
        mlp_epochs=int(values["mlp_epochs"]),
        mlp_learning_rate=float(values["mlp_learning_rate"]),
        mlp_weight_decay=float(values["mlp_weight_decay"]),
        mlp_batch_size=int(values["mlp_batch_size"]),
        separation_weight=float(values["separation_weight"]),
        spread_weight=float(values["spread_weight"]),
        separation_margin=float(values["separation_margin"]),
    )
    genetic = GeneticConfig(
        population_size=int(values["population_size"]),
        generations=int(values["generations"]),
        mutation_std=float(values["mutation_std"]),
        elite_count=int(values["elite_count"]),
    )
    return frontend, genetic


def trial_name(config: TwoPhaseConfig, genetic: GeneticConfig) -> str:
    return f"{MODEL_PREFIX}_{genetic_tag(genetic)}_{setting_tag(config)}_vlr{config.vqc.learning_rate:g}_vep{config.vqc.epochs}"


def finished_cells(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty or "model" not in frame.columns:
        return set()
    required = {0.0, *HEADLINE_LEVELS}
    done = set()
    for (model, seed), group in frame.groupby(["model", "seed"]):
        if required <= set(group["p"].astype(float)):
            done.add((str(model), int(seed)))
    return done


def train_and_score(
    values: dict,
    seed: int,
    splits,
    runs_path: Path,
    cost_path: Path,
) -> None:
    config, genetic = configs_from_values(values)
    name = trial_name(config, genetic)
    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    x_balanced, y_balanced = undersample(x_train, y_train, seed)
    optimizer = make_vqc_optimizer("adam", config.vqc.learning_rate)

    print(f"\n{name} seed {seed}: training on {len(y_balanced)} balanced rows", flush=True)
    started = time.time()
    model = TwoPhaseVQC(config, vqc_optimizer_fn=optimizer)
    model.fit_frontend(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
    if model.frontend is None:
        raise RuntimeError("Frontend training did not produce a network.")
    for parameter in model.frontend.parameters():
        parameter.requires_grad_(False)
    model.frontend.eval()
    angles = encoded_angles(model, x_balanced, x_valid, x_test)
    initial = evolve_weights(
        model.vqc.fraud_probability,
        model.vqc.initialize,
        angles["train"],
        y_balanced,
        seed,
        angles["valid"],
        y_valid,
        genetic,
        verbose=True,
    )
    history = model.vqc.fit(
        angles["train"],
        y_balanced,
        angles["valid"],
        y_valid,
        seed,
        verbose=False,
        optimizer_fn=optimizer,
        initial_weights=initial,
    )
    save_two_phase(name, seed, model, genetic=genetic)
    print(f"{name} seed {seed}: trained in {time.time() - started:.0f}s", flush=True)

    noise = evaluate_under_noise(
        model.vqc,
        model.vqc.weights,
        angles["valid"],
        angles["test"],
        y_valid,
        y_test,
        levels=(0.0, *HEADLINE_LEVELS),
    )
    noise.insert(0, "seed", seed)
    noise.insert(0, "kind", "vqc")
    noise.insert(0, "model", name)
    for key, value in values.items():
        noise[key] = value
    append_rows(runs_path, noise.to_dict("records"))

    adam = cost_ledger(
        n_train=len(y_balanced),
        n_valid=len(y_valid),
        n_test=len(y_test),
        n_parameters=model.vqc.config.n_parameters,
        epochs=model.vqc.config.epochs,
    )
    ga_forwards = forward_executions(genetic, len(y_balanced), len(y_valid))
    ledger = {
        **adam,
        **values,
        "model": name,
        "seed": seed,
        "n_train": len(y_balanced),
        "n_valid": len(y_valid),
        "n_test": len(y_test),
        "noise_locations": model.vqc.noise_locations(
            torch.tensor(angles["train"][0]), model.vqc.weights.detach().cpu()
        ),
        "genetic_forward_executions": ga_forwards,
        "circuit_executions_total": adam["circuit_executions"] + ga_forwards,
        "shots_total": (adam["circuit_executions"] + ga_forwards) * 1024,
        "frontend_parameters": sum(parameter.numel() for parameter in model.frontend.parameters()),
        "best_validation_AUPRC": float(history["validation AUPRC"].max()),
    }
    append_rows(cost_path, [ledger])
    for probability in HEADLINE_LEVELS:
        score = float(noise.loc[noise["p"] == probability, "AUPRC"].iloc[0])
        print(f"{name} seed {seed}: test AUPRC at p={probability:.3f} is {score:.3f}", flush=True)


def sweep_hparams(
    trials: int = 4,
    seeds: list[int] | None = None,
    search_seed: int = 42,
    output: Path | None = None,
    cost_output: Path | None = None,
) -> None:
    seeds = list(seeds or [0])
    output = output or DEFAULT_OUTPUT
    cost_output = cost_output or DEFAULT_COST
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidate_values(trials, search_seed)
    splits = fixed_split(load_transactions())
    done = finished_cells(output)
    print(
        f"two_phase_ga_adam_chained hparam sweep: {len(candidates)} trials, "
        f"seeds {seeds}, writing {output}",
        flush=True,
    )
    for index, values in enumerate(candidates, start=1):
        config, genetic = configs_from_values(values)
        name = trial_name(config, genetic)
        print(
            f"\n[{index}/{len(candidates)}] {name}",
            flush=True,
        )
        for seed in seeds:
            if (name, seed) in done:
                print(f"skip {name} seed {seed}: already in {output}", flush=True)
                continue
            train_and_score(values, seed, splits, output, cost_output)
            done.add((name, seed))
    if output.exists():
        frame = pd.read_csv(output)
        summary = (
            frame.groupby(["model", "p"])["AUPRC"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "AUPRC mean", "std": "AUPRC std", "count": "seeds"})
            .reset_index()
        )
        print("\nHparam sweep AUPRC summary")
        print(summary.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hyperparameters of two_phase_ga_adam_chained.")
    parser.add_argument("--trials", type=int, default=4, help="Candidates, including current defaults.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1.")
    sweep_hparams(
        trials=args.trials,
        seeds=args.seeds,
        search_seed=args.search_seed,
        output=args.output,
        cost_output=args.cost_output,
    )


if __name__ == "__main__":
    main()
