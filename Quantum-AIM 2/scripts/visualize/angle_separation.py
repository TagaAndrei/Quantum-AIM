"""Compare fraud and legitimate separation in the four angles of both encoders.

The baseline angles come from PCA. The two-phase angles come from the classical
frontend only: this script does not train the quantum circuit. Both encoders are
fit on the same balanced training rows, then drawn on identical axes in [0, π].

Run from the project root::

    python -m scripts.visualize.angle_separation
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from data.creditcard import load_transactions, prepare_splits
from models.baseline import AngleEncoder
from models.two_phase import TwoPhaseConfig, TwoPhaseVQC
from paths import RESULTS

LEGIT_COLOR, FRAUD_COLOR = "#2a78d6", "#eb6834"
OUTPUT_PATH = RESULTS / "angle_separation.png"
SEED = 42


def cluster_stats(angles: np.ndarray, labels: np.ndarray) -> dict[str, float | np.ndarray]:
    legitimate = angles[labels == 0]
    fraud = angles[labels == 1]
    legit_center = legitimate.mean(axis=0)
    fraud_center = fraud.mean(axis=0)
    return {
        "center gap": float(np.linalg.norm(fraud_center - legit_center)),
        "legitimate radius": float(np.linalg.norm(legitimate - legit_center, axis=1).mean()),
        "fraud radius": float(np.linalg.norm(fraud - fraud_center, axis=1).mean()),
        "per angle gap": np.abs(fraud_center - legit_center),
    }


def scatter_angles(ax, angles: np.ndarray, labels: np.ndarray, x_col: int, y_col: int, title: str) -> None:
    for label, name, color in [(0, "legitimate", LEGIT_COLOR), (1, "fraud", FRAUD_COLOR)]:
        mask = labels == label
        ax.scatter(
            angles[mask, x_col],
            angles[mask, y_col],
            s=14,
            alpha=0.65,
            c=color,
            label=name,
            linewidths=0,
        )
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"angle {x_col}")
    ax.set_ylabel(f"angle {y_col}")
    ax.set_title(title)


def validation_view(angles: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every fraud and an equal number of legitimate rows so the plot stays readable."""
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(labels == 1)
    legitimate = rng.choice(np.flatnonzero(labels == 0), size=fraud.size, replace=False)
    keep = np.concatenate([fraud, legitimate])
    return angles[keep], labels[keep]


def print_stats(name: str, split: str, angles: np.ndarray, labels: np.ndarray) -> np.ndarray:
    stats = cluster_stats(angles, labels)
    per_angle = stats["per angle gap"]
    assert isinstance(per_angle, np.ndarray)
    print(
        f"{name:18s} {split:12s}  center gap {stats['center gap']:.3f}"
        f"   legitimate radius {stats['legitimate radius']:.3f}"
        f"   fraud radius {stats['fraud radius']:.3f}"
        f"   per-angle gaps {np.array2string(per_angle, precision=3)}"
    )
    return per_angle


def _style_axes() -> None:
    plt.rcParams.update(
        {"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}
    )


def plot_best_sweep(
    config: TwoPhaseConfig,
    x_train,
    y_train,
    x_valid,
    y_valid,
    seed: int,
    output_path: Path,
) -> None:
    """Fit one frontend configuration and save its train and validation angles."""
    model = TwoPhaseVQC(config)
    model.fit_frontend(x_train, y_train, x_valid, y_valid, seed=seed, verbose=False)
    learned_train = model.encode(x_train)
    learned_valid = model.encode(x_valid)
    train_gap = print_stats("sweep frontend", "train", learned_train, y_train)
    print_stats("sweep frontend", "validation", learned_valid, y_valid)
    valid_angles, valid_labels = validation_view(learned_valid, y_valid, seed)

    _style_axes()
    fig = plt.figure(figsize=(8, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.35)
    scatter_angles(fig.add_subplot(grid[0]), learned_train, y_train, 0, 2, "Best frontend, training")
    scatter_angles(
        fig.add_subplot(grid[1]),
        valid_angles,
        valid_labels,
        0,
        2,
        "Best frontend, validation sample",
    )
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.98))

    gap_axis = fig.add_subplot(grid[2])
    positions = np.arange(4)
    gap_axis.bar(positions, train_gap, color="#e39a78")
    gap_axis.set_xticks(positions, [f"angle {i}" for i in range(4)])
    gap_axis.set_ylabel("|fraud mean − legitimate mean|")
    gap_axis.set_title("Training-set separation along each angle")
    gap_axis.set_ylim(0, np.pi)

    fig.suptitle(
        f"hidden {config.hidden_size}  ·  lr {config.mlp_learning_rate:g}"
        f"  ·  separation {config.separation_weight:g}  ·  axes in [0, π]",
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {output_path.resolve()}")


def main() -> None:
    transactions = load_transactions()
    x_train, y_train, x_valid, y_valid, _, _ = prepare_splits(transactions, seed=SEED)

    baseline = AngleEncoder(n_components=4, seed=SEED).fit(x_train)
    baseline_train = baseline.transform(x_train)
    baseline_valid = baseline.transform(x_valid)

    frontend = TwoPhaseVQC()
    frontend.fit_frontend(x_train, y_train, x_valid, y_valid, seed=SEED)
    learned_train = frontend.encode(x_train)
    learned_valid = frontend.encode(x_valid)

    print()
    base_gap = print_stats("baseline PCA", "train", baseline_train, y_train)
    learned_gap = print_stats("two-phase MLP", "train", learned_train, y_train)
    print_stats("baseline PCA", "validation", baseline_valid, y_valid)
    print_stats("two-phase MLP", "validation", learned_valid, y_valid)

    valid_base, valid_base_y = validation_view(baseline_valid, y_valid, SEED)
    valid_learned, valid_learned_y = validation_view(learned_valid, y_valid, SEED)

    _style_axes()
    fig = plt.figure(figsize=(11, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.32, wspace=0.28)

    panels = [
        (grid[0, 0], baseline_train, y_train, "Baseline PCA, training"),
        (grid[0, 1], learned_train, y_train, "Two-phase MLP, training"),
        (grid[1, 0], valid_base, valid_base_y, "Baseline PCA, validation sample"),
        (grid[1, 1], valid_learned, valid_learned_y, "Two-phase MLP, validation sample"),
    ]
    for slot, angles, labels, title in panels:
        scatter_angles(fig.add_subplot(slot), angles, labels, 0, 2, title)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.98))

    gap_axis = fig.add_subplot(grid[2, :])
    positions = np.arange(4)
    width = 0.36
    gap_axis.bar(positions - width / 2, base_gap, width=width, color="#7aa2d4", label="baseline PCA")
    gap_axis.bar(positions + width / 2, learned_gap, width=width, color="#e39a78", label="two-phase MLP")
    gap_axis.set_xticks(positions, [f"angle {i}" for i in range(4)])
    gap_axis.set_ylabel("|fraud mean − legitimate mean|")
    gap_axis.set_title("Training-set separation along each angle")
    gap_axis.legend(frameon=False)
    gap_axis.set_ylim(0, np.pi)

    fig.suptitle("Angle separation on the shared split  ·  both axes run from 0 to π", y=0.995)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
