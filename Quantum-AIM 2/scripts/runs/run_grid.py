"""Seed grid for Task 2.

One factory builds every run. Results are appended to ``results/task2_runs.csv``.
The experiment name carries the frontend hyperparameters, so a new setting does
not replace rows already in the file.

Experiments, selected with ``--experiments``:

- ``baseline``: PCA angles and the baseline circuit, Adam on the weights
- ``baseline_chain``: PCA angles, nearest-neighbour ZZ map, one ansatz layer, Adam
- ``baseline_ga``: PCA angles, genetic search for the weights, then Adam
- ``baseline_chain_ga``: PCA angles, chain circuit, genetic search, then Adam
- ``two_phase``: learned angles plus the baseline circuit, Adam on the weights
- ``two_phase_chain``: learned angles, nearest-neighbour ZZ map, one ansatz layer, Adam
- ``two_phase_ga``: learned angles, genetic search for the circuit weights, then Adam
- ``two_phase_chain_ga``: learned angles, chain circuit, genetic search, then Adam
- ``two_phase_ga_adam_chained``: same as ``two_phase_chain_ga``, short stable name
- ``two_phase_ga_adam_chained_hparams``: sweep frontend, Adam, and genetic
  settings of that chained model. Results go to their own CSV.
- ``checkpoint_noise``: reload every saved ``model.pth``, score noise from
  0 to 0.5, and write the full execution cost.
- ``head``: the same frontend and its classification head, no circuit
- ``muon``: the same frontend, Muon on the circuit weights
- ``lightcone``: the same frontend and the projected light-cone kernel.
  ``--optimizer adam``, ``adamw``, or ``muon`` trains the ansatz. Cone weights
  come from the paper's quadratic program.

``--optimizer`` changes only ``lightcone``. Register another builder with
:meth:`RunFactory.add` when a run needs a different model or optimizer.

Run from the project root::

    python -m scripts.runs.run_grid --experiments head muon
    python -m scripts.runs.run_grid --experiments lightcone --optimizer muon
    python -m scripts.runs.run_grid --experiments baseline_ga two_phase_ga
    python -m scripts.runs.run_grid --experiments baseline_chain_ga two_phase_chain_ga
    python -m scripts.runs.run_grid --experiments two_phase_ga_adam_chained
    python -m scripts.runs.run_grid --experiments two_phase_ga_adam_chained_hparams --seeds 0 --trials 4
    python -m scripts.runs.run_grid --experiments checkpoint_noise
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch

from data.creditcard import fixed_split, load_transactions, undersample
from evolutionary import GeneticConfig, evolve_weights, genetic_tag
from models.baseline import AngleEncoder, BaselineVQC
from models.config import VQCConfig
from models.lightcone import LightConeVQC
from models.two_phase import ClassificationHead, TwoPhaseConfig, TwoPhaseVQC
from optimizers import OPTIMIZERS, make_vqc_optimizer
from paths import RESULTS
from scripts.runs.checkpoints import save_baseline, save_head, save_lightcone, save_two_phase
from scripts.runs.evaluation import (
    HEADLINE_LEVELS,
    LEDGER_PATH,
    RUNS_PATH,
    SEEDS,
    append_rows,
    choose_threshold,
    cost_ledger,
    evaluate,
    evaluate_under_noise,
    finished_head_cells,
    finished_vqc_cells,
    logistic_row,
    summarise,
)

SWEEP_KEYS = ("two_phase_ga_adam_chained_hparams", "checkpoint_noise")

Trainer = Callable[..., tuple]


@dataclass(frozen=True)
class GridRun:
    """One named experiment.

    ``kind="vqc"`` trains ``train`` and scores the circuit under noise.
    ``train`` returns ``(circuit, weights, angles, history)``.
    ``kind="head"`` trains the classical head in ``config`` and skips the circuit.
    """

    name: str
    kind: str
    summary: str
    train: Trainer | None = None
    config: TwoPhaseConfig | None = None


class RunFactory:
    """Build named runs from a frontend config and, for light-cone, an optimizer.

    ``add`` registers another experiment. The builder receives this factory and
    returns a :class:`GridRun`.
    """

    def __init__(self, config: TwoPhaseConfig | None = None, optimizer: str = "adam"):
        if optimizer not in OPTIMIZERS:
            known = ", ".join(OPTIMIZERS)
            raise ValueError(f"Unknown optimizer {optimizer}. Known optimizers: {known}")
        self.config = config or TwoPhaseConfig()
        self.optimizer = optimizer
        self.genetic = GeneticConfig()
        self.name_override: str | None = None
        self._builders: dict[str, Callable[[RunFactory], GridRun]] = {}

    def add(self, key: str, builder: Callable[[RunFactory], GridRun]) -> None:
        self._builders[key] = builder

    def build(self, key: str) -> GridRun:
        try:
            builder = self._builders[key]
        except KeyError:
            known = ", ".join(self._builders)
            raise KeyError(f"Unknown experiment {key}. Known experiments: {known}") from None
        return builder(self)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._builders)


def setting_tag(config: TwoPhaseConfig) -> str:
    return (
        f"ep{config.mlp_epochs}"
        f"_lr{config.mlp_learning_rate:g}"
        f"_wd{config.mlp_weight_decay:g}"
        f"_bs{config.mlp_batch_size}"
        f"_sep{config.separation_weight:g}"
        f"_spd{config.spread_weight:g}"
        f"_m{config.separation_margin:g}"
    )


def experiment_name(key: str, config: TwoPhaseConfig, optimizer: str = "adam") -> str:
    """CSV model name. Kept stable so finished seeds are still recognised."""
    if key == "baseline":
        return "baseline"
    if key == "baseline_chain":
        return "baseline_chain_ansatz1"
    if key == "lightcone":
        return f"lightcone_{optimizer}_{setting_tag(config)}"
    prefix = {
        "two_phase": "two_phase",
        "two_phase_chain": "two_phase_chain_ansatz1",
        "head": "head",
        "muon": "two_phase_muon",
    }[key]
    return f"{prefix}_{setting_tag(config)}"


def chain_vqc(config: VQCConfig | None = None) -> VQCConfig:
    """Nearest-neighbour feature map and a single EfficientSU2 layer. Adam stays at 0.05."""
    return replace(config or VQCConfig(), connectivity="chain", ansatz_reps=1)


def ga_summary(name: str, genetic: GeneticConfig) -> str:
    return (
        f"{name}: population {genetic.population_size}, generations {genetic.generations}, "
        f"mutation {genetic.mutation_std:g}, elites {genetic.elite_count}, then Adam"
    )


def frontend_summary(name: str, config: TwoPhaseConfig) -> str:
    return (
        f"{name}: epochs {config.mlp_epochs}, lr {config.mlp_learning_rate:g}, "
        f"wd {config.mlp_weight_decay:g}, batch {config.mlp_batch_size}, "
        f"sep {config.separation_weight:g}, spread {config.spread_weight:g}, "
        f"margin {config.separation_margin:g}"
    )


def encoded_angles(model, x_train, x_valid, x_test) -> dict[str, object]:
    return {"train": model.encode(x_train), "valid": model.encode(x_valid), "test": model.encode(x_test)}


def build_baseline(_factory: RunFactory) -> GridRun:
    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        encoder = AngleEncoder(n_components=4, seed=seed).fit(x_balanced)
        angles = {
            "train": encoder.transform(x_balanced),
            "valid": encoder.transform(x_valid),
            "test": encoder.transform(x_test),
        }
        model = BaselineVQC()
        history = model.fit(angles["train"], y_balanced, angles["valid"], y_valid, seed, verbose=False)
        save_baseline("baseline", seed, encoder, model)
        return model, model.weights, angles, history

    return GridRun(
        name="baseline",
        kind="vqc",
        summary="baseline: PCA angles, EfficientSU2 readout, Adam on the circuit",
        train=train,
    )


def build_baseline_chain(_factory: RunFactory) -> GridRun:
    config = chain_vqc()
    optimizer = make_vqc_optimizer("adam", config.learning_rate)

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        encoder = AngleEncoder(n_components=config.n_qubits, seed=seed).fit(x_balanced)
        angles = {
            "train": encoder.transform(x_balanced),
            "valid": encoder.transform(x_valid),
            "test": encoder.transform(x_test),
        }
        model = BaselineVQC(config)
        history = model.fit(
            angles["train"],
            y_balanced,
            angles["valid"],
            y_valid,
            seed,
            verbose=False,
            optimizer_fn=optimizer,
        )
        save_baseline("baseline_chain_ansatz1", seed, encoder, model)
        return model, model.weights, angles, history

    return GridRun(
        name="baseline_chain_ansatz1",
        kind="vqc",
        summary="baseline_chain_ansatz1: PCA angles, chain ZZ map, one EfficientSU2 layer, Adam",
        train=train,
    )


def build_baseline_ga(_factory: RunFactory) -> GridRun:
    genetic = GeneticConfig()
    optimizer = make_vqc_optimizer("adam", VQCConfig().learning_rate)
    name = f"baseline_ga_{genetic_tag(genetic)}"

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        encoder = AngleEncoder(n_components=4, seed=seed).fit(x_balanced)
        angles = {
            "train": encoder.transform(x_balanced),
            "valid": encoder.transform(x_valid),
            "test": encoder.transform(x_test),
        }
        model = BaselineVQC()
        initial = evolve_weights(
            model.fraud_probability,
            model.initialize,
            angles["train"],
            y_balanced,
            seed,
            angles["valid"],
            y_valid,
            genetic,
            verbose=True,
        )
        history = model.fit(
            angles["train"],
            y_balanced,
            angles["valid"],
            y_valid,
            seed,
            verbose=False,
            optimizer_fn=optimizer,
            initial_weights=initial,
        )
        save_baseline(name, seed, encoder, model)
        return model, model.weights, angles, history

    return GridRun(name=name, kind="vqc", summary=ga_summary(name, genetic), train=train)


def build_baseline_chain_ga(_factory: RunFactory) -> GridRun:
    genetic = GeneticConfig()
    config = chain_vqc()
    optimizer = make_vqc_optimizer("adam", config.learning_rate)
    name = f"baseline_chain_ga_{genetic_tag(genetic)}"

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        encoder = AngleEncoder(n_components=config.n_qubits, seed=seed).fit(x_balanced)
        angles = {
            "train": encoder.transform(x_balanced),
            "valid": encoder.transform(x_valid),
            "test": encoder.transform(x_test),
        }
        model = BaselineVQC(config)
        initial = evolve_weights(
            model.fraud_probability,
            model.initialize,
            angles["train"],
            y_balanced,
            seed,
            angles["valid"],
            y_valid,
            genetic,
            verbose=True,
        )
        history = model.fit(
            angles["train"],
            y_balanced,
            angles["valid"],
            y_valid,
            seed,
            verbose=False,
            optimizer_fn=optimizer,
            initial_weights=initial,
        )
        save_baseline(name, seed, encoder, model)
        return model, model.weights, angles, history

    return GridRun(
        name=name,
        kind="vqc",
        summary=f"{ga_summary(name, genetic)}; chain ZZ map, one EfficientSU2 layer",
        train=train,
    )


def build_two_phase(factory: RunFactory) -> GridRun:
    config = factory.config
    optimizer = make_vqc_optimizer("adam", config.vqc.learning_rate)
    name = experiment_name("two_phase", config)

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        model = TwoPhaseVQC(config, vqc_optimizer_fn=optimizer)
        _, history = model.fit(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
        save_two_phase(name, seed, model)
        return model.vqc, model.vqc.weights, encoded_angles(model, x_balanced, x_valid, x_test), history

    return GridRun(name=name, kind="vqc", summary=frontend_summary(name, config), train=train, config=config)


def build_two_phase_chain(factory: RunFactory) -> GridRun:
    config = replace(factory.config, vqc=chain_vqc(factory.config.vqc))
    optimizer = make_vqc_optimizer("adam", config.vqc.learning_rate)
    name = experiment_name("two_phase_chain", config)

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        model = TwoPhaseVQC(config, vqc_optimizer_fn=optimizer)
        _, history = model.fit(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
        save_two_phase(name, seed, model)
        return model.vqc, model.vqc.weights, encoded_angles(model, x_balanced, x_valid, x_test), history

    return GridRun(
        name=name,
        kind="vqc",
        summary=f"{frontend_summary(name, config)}; chain ZZ map, one EfficientSU2 layer, Adam",
        train=train,
        config=config,
    )


def build_two_phase_ga(factory: RunFactory) -> GridRun:
    config = factory.config
    genetic = GeneticConfig()
    optimizer = make_vqc_optimizer("adam", config.vqc.learning_rate)
    name = f"two_phase_ga_{genetic_tag(genetic)}_{setting_tag(config)}"

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
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
        return model.vqc, model.vqc.weights, angles, history

    return GridRun(
        name=name,
        kind="vqc",
        summary=f"{frontend_summary(name, config)}; {ga_summary('circuit', genetic)}",
        train=train,
        config=config,
    )


def build_two_phase_chain_ga(factory: RunFactory) -> GridRun:
    config = replace(factory.config, vqc=chain_vqc(factory.config.vqc))
    genetic = factory.genetic
    optimizer = make_vqc_optimizer("adam", config.vqc.learning_rate)
    name = factory.name_override or f"two_phase_chain_ga_{genetic_tag(genetic)}_{setting_tag(config)}"

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
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
        return model.vqc, model.vqc.weights, angles, history

    return GridRun(
        name=name,
        kind="vqc",
        summary=(
            f"{frontend_summary(name, config)}; {ga_summary('circuit', genetic)}; "
            "chain ZZ map, one EfficientSU2 layer"
        ),
        train=train,
        config=config,
    )


def build_two_phase_ga_adam_chained(factory: RunFactory) -> GridRun:
    """Short alias for the chain + GA + Adam pipeline used by the hparam sweep."""
    previous = factory.name_override
    factory.name_override = previous or (
        f"two_phase_ga_adam_chained_{genetic_tag(factory.genetic)}_{setting_tag(factory.config)}"
    )
    try:
        return build_two_phase_chain_ga(factory)
    finally:
        factory.name_override = previous


def build_head(factory: RunFactory) -> GridRun:
    config = factory.config
    name = experiment_name("head", config)
    return GridRun(
        name=name,
        kind="head",
        summary=frontend_summary(name, config),
        config=config,
    )


def build_muon(factory: RunFactory) -> GridRun:
    config = factory.config
    optimizer = make_vqc_optimizer("muon", config.vqc.learning_rate)
    name = experiment_name("muon", config)

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        model = TwoPhaseVQC(config, vqc_optimizer_fn=optimizer)
        _, history = model.fit(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
        save_two_phase(name, seed, model)
        return model.vqc, model.vqc.weights, encoded_angles(model, x_balanced, x_valid, x_test), history

    return GridRun(
        name=name,
        kind="vqc",
        summary=f"{frontend_summary(name, config)}; ansatz optimizer muon",
        train=train,
        config=config,
    )


def build_lightcone(factory: RunFactory) -> GridRun:
    config = factory.config
    optimizer = make_vqc_optimizer(factory.optimizer, config.vqc.learning_rate)
    name = experiment_name("lightcone", config, factory.optimizer)

    def train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed: int):
        frontend = TwoPhaseVQC(config)
        frontend.fit_frontend(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
        if frontend.frontend is None:
            raise RuntimeError("Frontend training did not produce a network.")
        for parameter in frontend.frontend.parameters():
            parameter.requires_grad_(False)
        frontend.frontend.eval()
        angles = encoded_angles(frontend, x_balanced, x_valid, x_test)
        circuit = LightConeVQC(config.vqc)
        history = circuit.fit(
            angles["train"],
            y_balanced,
            angles["valid"],
            y_valid,
            seed,
            verbose=False,
            optimizer_fn=optimizer,
        )
        save_lightcone(name, seed, frontend, circuit)
        return circuit, circuit.weights, angles, history

    return GridRun(
        name=name,
        kind="vqc",
        summary=f"{frontend_summary(name, config)}; ansatz optimizer {factory.optimizer}",
        train=train,
        config=config,
    )


def default_factory(config: TwoPhaseConfig | None = None, optimizer: str = "adam") -> RunFactory:
    factory = RunFactory(config, optimizer)
    factory.add("baseline", build_baseline)
    factory.add("baseline_chain", build_baseline_chain)
    factory.add("baseline_ga", build_baseline_ga)
    factory.add("baseline_chain_ga", build_baseline_chain_ga)
    factory.add("two_phase", build_two_phase)
    factory.add("two_phase_chain", build_two_phase_chain)
    factory.add("two_phase_ga", build_two_phase_ga)
    factory.add("two_phase_chain_ga", build_two_phase_chain_ga)
    factory.add("two_phase_ga_adam_chained", build_two_phase_ga_adam_chained)
    factory.add("head", build_head)
    factory.add("muon", build_muon)
    factory.add("lightcone", build_lightcone)
    return factory


def run_circuit(spec: GridRun, seed: int, splits) -> None:
    if spec.train is None:
        raise ValueError(f"{spec.name} is a circuit run and needs a train function")
    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    x_balanced, y_balanced = undersample(x_train, y_train, seed)
    print(f"\n{spec.name} seed {seed}: training on {len(y_balanced)} balanced rows", flush=True)
    started = time.time()
    circuit, weights, angles, history = spec.train(x_balanced, y_balanced, x_valid, y_valid, x_test, seed)
    print(f"{spec.name} seed {seed}: trained in {time.time() - started:.0f}s", flush=True)

    noise = evaluate_under_noise(circuit, weights, angles["valid"], angles["test"], y_valid, y_test)
    noise.insert(0, "seed", seed)
    noise.insert(0, "kind", "vqc")
    noise.insert(0, "model", spec.name)
    reference = logistic_row(angles["train"], y_balanced, angles["valid"], y_valid, angles["test"], y_test)
    reference.update({"model": spec.name, "kind": "logistic", "seed": seed})
    append_rows(RUNS_PATH, [*noise.to_dict("records"), reference])

    ledger = cost_ledger(
        n_train=len(y_balanced),
        n_valid=len(y_valid),
        n_test=len(y_test),
        n_parameters=circuit.config.n_parameters,
        epochs=circuit.config.epochs,
    )
    ledger.update(
        {
            "model": spec.name,
            "seed": seed,
            "n_train": len(y_balanced),
            "n_valid": len(y_valid),
            "n_test": len(y_test),
            "noise_locations": circuit.noise_locations(torch.tensor(angles["train"][0]), weights.detach().cpu()),
            "best_validation_AUPRC": float(history["validation AUPRC"].max()),
        }
    )
    append_rows(LEDGER_PATH, [ledger])
    for probability in HEADLINE_LEVELS:
        score = float(noise.loc[noise["p"] == probability, "AUPRC"].iloc[0])
        print(f"{spec.name} seed {seed}: test AUPRC at p={probability:.3f} is {score:.3f}", flush=True)


def run_head(spec: GridRun, seed: int, splits) -> None:
    """Score the classification head on clean data and append that one row."""
    if spec.config is None:
        raise ValueError(f"{spec.name} is a head run and needs a config")
    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    x_balanced, y_balanced = undersample(x_train, y_train, seed)
    print(f"\n{spec.name} seed {seed}: training on {len(y_balanced)} balanced rows", flush=True)
    started = time.time()
    model = ClassificationHead(spec.config)
    history = model.fit(x_balanced, y_balanced, x_valid, y_valid, seed, verbose=False)
    save_head(spec.name, seed, model)
    print(f"{spec.name} seed {seed}: trained in {time.time() - started:.0f}s", flush=True)

    valid_scores = model.predict(x_valid)
    test_scores = model.predict(x_test)
    threshold = choose_threshold(y_valid, valid_scores)
    row = {
        "model": spec.name,
        "kind": "head",
        "seed": seed,
        "p": 0.0,
        **evaluate(y_test, test_scores, threshold),
        "Spearman rho": float("nan"),
        "threshold": threshold,
    }
    angles = encoded_angles(model, x_balanced, x_valid, x_test)
    reference = logistic_row(angles["train"], y_balanced, angles["valid"], y_valid, angles["test"], y_test)
    reference.update({"model": spec.name, "kind": "logistic", "seed": seed})
    append_rows(RUNS_PATH, [row, reference])
    # Linear(4 angles → 1 logit) is four weights and a bias. No circuit is trained.
    append_rows(
        LEDGER_PATH,
        [
            {
                "n_parameters": 5,
                "epochs": spec.config.mlp_epochs,
                "executions_per_sample": 0,
                "training_executions": 0,
                "validation_executions": 0,
                "test_executions": 0,
                "circuit_executions": 0,
                "shots": 0,
                "model": spec.name,
                "seed": seed,
                "n_train": len(y_balanced),
                "n_valid": len(y_valid),
                "n_test": len(y_test),
                "noise_locations": 0,
                "best_validation_AUPRC": float(history["validation AUPRC"].max()),
            }
        ],
    )
    print(f"{spec.name} seed {seed}: test AUPRC is {float(row['AUPRC']):.3f}", flush=True)


def execute(spec: GridRun, seeds: list[int], splits, runs: pd.DataFrame) -> None:
    print(spec.summary, flush=True)
    if spec.kind == "vqc":
        done = finished_vqc_cells(runs)
    elif spec.kind == "head":
        done = finished_head_cells(runs, spec.name)
    else:
        raise ValueError(f"{spec.name} has unknown kind {spec.kind}")
    for seed in seeds:
        if (spec.name, seed) in done:
            print(f"skip {spec.name} seed {seed}: already in {RUNS_PATH}", flush=True)
            continue
        if spec.kind == "head":
            run_head(spec, seed, splits)
        else:
            run_circuit(spec, seed, splits)
        done.add((spec.name, seed))


def print_summary() -> None:
    if not RUNS_PATH.exists():
        return
    saved = pd.read_csv(RUNS_PATH)
    summary = summarise(saved)
    print("\nTest AUPRC, mean ± std over finished seeds")
    print(summary.round(3).to_string(index=False))
    heads = saved[saved["kind"] == "head"]
    if not heads.empty:
        grouped = heads.groupby("model")["AUPRC"].agg(["mean", "std", "count"])
        grouped = grouped.rename(columns={"mean": "AUPRC mean", "std": "AUPRC std", "count": "seeds"})
        print("\nClassification head, clean test AUPRC")
        print(grouped.round(3).to_string())


def run_sweep(key: str, seeds: list[int], trials: int) -> None:
    """Dispatch a sweep registered next to the training experiments."""
    if key == "two_phase_ga_adam_chained_hparams":
        from scripts.sweep.chain_ga_hparams import sweep_hparams

        sweep_hparams(trials=trials, seeds=seeds)
        return
    if key == "checkpoint_noise":
        from scripts.sweep.checkpoint_noise import sweep_checkpoints

        sweep_checkpoints(seeds=seeds)
        return
    raise KeyError(key)


def main() -> None:
    factory_for_help = default_factory()
    parser = argparse.ArgumentParser(description="Append Task 2 experiments to the results CSV.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*factory_for_help.keys, *SWEEP_KEYS],
        default=["head", "muon"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--trials",
        type=int,
        default=4,
        help="Hyperparameter trials for two_phase_ga_adam_chained_hparams, including the current defaults.",
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        default="adam",
        help="Ansatz optimizer for --experiments lightcone. adamw uses AdamW with weight decay 0.01.",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        if not RUNS_PATH.exists():
            raise SystemExit(f"No results at {RUNS_PATH}")
        print_summary()
        return
    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    factory = default_factory(optimizer=args.optimizer)
    splits = None
    runs = pd.read_csv(RUNS_PATH) if RUNS_PATH.exists() else pd.DataFrame()
    for key in args.experiments:
        if key in SWEEP_KEYS:
            run_sweep(key, args.seeds, args.trials)
            continue
        if splits is None:
            splits = fixed_split(load_transactions())
        execute(factory.build(key), args.seeds, splits, runs)
        if RUNS_PATH.exists():
            runs = pd.read_csv(RUNS_PATH)
    if any(key not in SWEEP_KEYS for key in args.experiments):
        print_summary()


if __name__ == "__main__":
    main()
