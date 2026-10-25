"""Test metrics, the depolarizing-noise protocol, and the results CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from paths import RESULTS

SEEDS = (0, 1, 2, 3, 4)
NOISE_LEVELS = (0.001, 0.01, 0.05)
HEADLINE_LEVELS = (0.01, 0.05)
# Dense stress grid used by the checkpoint noise sweep: clean, then 0.001 → 0.5.
EXTENDED_NOISE_LEVELS = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
)
SHOTS_PER_EXECUTION = 1024
RUNS_PATH = RESULTS / "task2_runs.csv"
LEDGER_PATH = RESULTS / "task2_ledger.csv"


def noise_grid(start: float = 0.001, stop: float = 0.5, step: float | None = None) -> tuple[float, ...]:
    """``0.0`` plus noise probabilities from ``start`` to ``stop``.

    With ``step is None`` the fixed :data:`EXTENDED_NOISE_LEVELS` grid is used.
    """
    if step is None:
        return EXTENDED_NOISE_LEVELS
    if step <= 0:
        raise ValueError("step must be positive.")
    levels = [0.0]
    value = start
    while value <= stop + 1e-12:
        levels.append(round(float(value), 10))
        value += step
    if levels[-1] < stop - 1e-12:
        levels.append(float(stop))
    return tuple(dict.fromkeys(levels))


def choose_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 on the validation scores. Never uses the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true, scores, threshold) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def evaluate_under_noise(
    model,
    weights,
    angles_valid,
    angles_test,
    y_valid,
    y_test,
    levels: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Threshold from validation at the same noise level as the test scores.

    ``levels`` defaults to clean execution plus ``NOISE_LEVELS``. Spearman rho
    always compares that level with the clean test scores.
    """
    if levels is None:
        levels = (0.0, *NOISE_LEVELS)
    clean_valid = model.predict(angles_valid, weights)
    clean_test = model.predict(angles_test, weights)
    rows = []
    for probability in levels:
        if probability == 0.0:
            valid_scores, test_scores = clean_valid, clean_test
        else:
            valid_scores = model.predict_noisy(angles_valid, probability, weights)
            test_scores = model.predict_noisy(angles_test, probability, weights)
        threshold = choose_threshold(y_valid, valid_scores)
        rows.append(
            {
                "p": probability,
                **evaluate(y_test, test_scores, threshold),
                "Spearman rho": float(spearmanr(clean_test, test_scores).statistic),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def cost_ledger(n_train: int, n_valid: int, n_test: int, n_parameters: int, epochs: int) -> dict:
    """One circuit execution plus two shifted executions per parameter."""
    executions_per_sample = 1 + 2 * n_parameters
    training = epochs * n_train * executions_per_sample
    validation = epochs * n_valid
    test = n_test
    executions = training + validation + test
    return {
        "n_parameters": n_parameters,
        "epochs": epochs,
        "executions_per_sample": executions_per_sample,
        "training_executions": training,
        "validation_executions": validation,
        "test_executions": test,
        "circuit_executions": executions,
        "shots": executions * SHOTS_PER_EXECUTION,
    }


def logistic_row(angles_train, y_train, angles_valid, y_valid, angles_test, y_test) -> dict:
    reference = LogisticRegression(max_iter=1000).fit(angles_train, y_train)
    valid_scores = reference.predict_proba(angles_valid)[:, 1]
    test_scores = reference.predict_proba(angles_test)[:, 1]
    threshold = choose_threshold(y_valid, valid_scores)
    return {"p": 0.0, **evaluate(y_test, test_scores, threshold), "Spearman rho": np.nan, "threshold": threshold}


def finished_vqc_cells(runs: pd.DataFrame) -> set[tuple[str, int]]:
    if runs.empty or "kind" not in runs.columns:
        return set()
    required = {0.0, *NOISE_LEVELS}
    done = set()
    for (model, seed), group in runs[runs["kind"] == "vqc"].groupby(["model", "seed"]):
        if required <= set(group["p"].astype(float)):
            done.add((str(model), int(seed)))
    return done


def finished_head_cells(runs: pd.DataFrame, name: str) -> set[tuple[str, int]]:
    if runs.empty or "kind" not in runs.columns:
        return set()
    subset = runs[(runs["model"] == name) & (runs["kind"] == "head")]
    return {(name, int(seed)) for seed in subset["seed"].unique()}


def append_rows(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    quantum = runs[runs["kind"] == "vqc"].copy()
    grouped = quantum.groupby(["model", "p"])["AUPRC"].agg(["mean", "std", "count"])
    grouped = grouped.rename(columns={"mean": "AUPRC mean", "std": "AUPRC std", "count": "seeds"})
    return grouped.reset_index()
