# Quantum-AIM — Fraud on a Quantum Budget (Task 2)

Variational quantum classifiers for credit-card fraud detection under simulated depolarizing noise. The project reuses the Task 1 protocol (same data split, same noise model, AUPRC as the primary metric) and adds two-phase angle learning, chain circuits, genetic initialisation, Muon / AdamW, and light-cone kernels.

## Setup

```bash
# From the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install the wheel that matches your driver from [pytorch.org](https://pytorch.org) (example for CUDA 13.0):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Put the ULB credit-card CSV at:

```text
data/creditcard.csv
```

(Kaggle: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).)

Always run commands from the **project root** so imports resolve (`paths.py`, `models`, `scripts`, …).

## Project layout

```text
data/                 creditcard.csv loader and the fixed 60/20/20 split
models/               VQC configs, circuits, baseline, two-phase, light-cone
models/trained/       saved model.pth per experiment / seed
optimizers/           Adam, AdamW, Muon factories
evolutionary/         genetic search for circuit weights
scripts/runs/         seed grid, evaluation, checkpoints
scripts/sweep/        frontend / chain-GA hparams / checkpoint noise sweeps
scripts/visualize/    angle-separation plots
results/              task2_runs.csv, task2_ledger.csv, sweep CSVs
paths.py              DATA_CSV, RESULTS, TRAINED_MODELS
task1_guided_vqc_fraud_detection.ipynb
```

## How experiments are run

All training experiments go through one factory:

```bash
python -m scripts.runs.run_grid --experiments <name> [<name> ...]
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--experiments …` | One or more experiment keys (see below) |
| `--seeds 0 1 2 3 4` | Which seeds to train (default: all five) |
| `--optimizer adam\|adamw\|muon` | Ansatz optimiser for `lightcone` only |
| `--trials N` | Hparam-sweep trials for `two_phase_ga_adam_chained_hparams` |
| `--summary-only` | Print AUPRC summary from `results/task2_runs.csv` |

Results are **appended**. Finished `(model, seed)` pairs that already have all noise rows are skipped. Hyperparameter tags are part of the model name, so a new setting does not overwrite an old one.

### Training experiments

| Key | What it trains |
|-----|----------------|
| `baseline` | PCA angles + full ZZ map + 3 ansatz reps, Adam |
| `baseline_chain` | PCA + chain ZZ map + 1 ansatz rep, Adam |
| `baseline_ga` | PCA + full circuit, genetic init then Adam |
| `baseline_chain_ga` | PCA + chain circuit, genetic then Adam |
| `two_phase` | Learned angles + full circuit, Adam |
| `two_phase_chain` | Learned angles + chain circuit, Adam |
| `two_phase_ga` | Learned angles + GA then Adam (full circuit) |
| `two_phase_chain_ga` | Learned angles + GA then Adam (chain) |
| `two_phase_ga_adam_chained` | Alias of `two_phase_chain_ga` with a short name |
| `head` | Classical head only (no circuit, no noise) |
| `muon` | Same frontend as two-phase; Muon on the ansatz |
| `lightcone` | Two-phase frontend + light-cone kernel VQC |

Examples:

```bash
python -m scripts.runs.run_grid --experiments baseline
python -m scripts.runs.run_grid --experiments two_phase head muon
python -m scripts.runs.run_grid --experiments lightcone --optimizer adamw
python -m scripts.runs.run_grid --experiments baseline_chain two_phase_chain
python -m scripts.runs.run_grid --experiments baseline_ga two_phase_ga
python -m scripts.runs.run_grid --experiments two_phase_ga_adam_chained --seeds 0 1 2 3 4
```

Each finished VQC seed writes:

- rows in `results/task2_runs.csv` (clean + `p ∈ {0.001, 0.01, 0.05}`, plus a logistic reference on the same angles)
- a row in `results/task2_ledger.csv` (parameter-shift execution count)
- `models/trained/<experiment>/seed<k>/model.pth`

### Sweep experiments (also via `run_grid`)

| Key | Script | Output |
|-----|--------|--------|
| `two_phase_ga_adam_chained_hparams` | `scripts/sweep/chain_ga_hparams.py` | `results/two_phase_ga_adam_chained_hparams.csv`, `…_cost.csv`, checkpoints |
| `checkpoint_noise` | `scripts/sweep/checkpoint_noise.py` | `results/checkpoint_noise_metrics.csv`, `results/checkpoint_cost.csv` |

```bash
# Hyperparameter sweep for the chained GA+Adam model
python -m scripts.runs.run_grid --experiments two_phase_ga_adam_chained_hparams --seeds 0 --trials 4

# Same script directly
python -m scripts.sweep.chain_ga_hparams --trials 4 --seeds 0

# Dense noise + cost over every saved model.pth (p = 0, then 0.001 … 0.5)
python -m scripts.runs.run_grid --experiments checkpoint_noise
python -m scripts.sweep.checkpoint_noise --noise-step 0.01
python -m scripts.sweep.checkpoint_noise --names two_phase_ga_adam_chained_pop12_gen6_mut0.1_elite2_ep20_lr0.01_wd0.0001_bs64_sep0.5_spd0.001_m0.5
```

GA defaults (before Adam fine-tune): population 12, generations 6, mutation std 0.1, elites 2. VQC Adam uses learning rate 0.05 for 30 epochs (`VQCConfig`).

Other utilities:

```bash
# Frontend-only random search (no circuit)
python -m scripts.sweep.frontend --trials 50

# PCA vs learned angles plot
python -m scripts.visualize.angle_separation
```

## What to import (library use)

Run from the project root, or ensure the root is on `PYTHONPATH`.

### Data

```python
from data.creditcard import load_transactions, fixed_split, undersample, prepare_splits

frame = load_transactions()
x_train, y_train, x_valid, y_valid, x_test, y_test = fixed_split(frame)
x_bal, y_bal = undersample(x_train, y_train, seed=0)
```

### Models

```python
from models.config import VQCConfig
from models.baseline import AngleEncoder, BaselineVQC
from models.two_phase import TwoPhaseConfig, TwoPhaseVQC, ClassificationHead
from models.lightcone import LightConeVQC
from models.circuit import ZZFeatureMap, EfficientSU2Ansatz
```

### Optimisers and genetic search

```python
from optimizers import make_vqc_optimizer
from evolutionary import GeneticConfig, evolve_weights, genetic_tag, forward_executions

optimizer_fn = make_vqc_optimizer("adam", learning_rate=0.05)  # or "adamw", "muon"
```

### Evaluation and checkpoints

```python
from scripts.runs.evaluation import evaluate_under_noise, cost_ledger, choose_threshold
from scripts.runs.checkpoints import save_two_phase, load_checkpoint, discover_checkpoints
from paths import RESULTS, TRAINED_MODELS
```

### Building a grid experiment in code

```python
from scripts.runs.run_grid import default_factory, execute
from data.creditcard import fixed_split, load_transactions
import pandas as pd
from scripts.runs.evaluation import RUNS_PATH

factory = default_factory()
spec = factory.build("two_phase_ga_adam_chained")
splits = fixed_split(load_transactions())
runs = pd.read_csv(RUNS_PATH) if RUNS_PATH.exists() else pd.DataFrame()
execute(spec, seeds=[0, 1], splits=splits, runs=runs)
```

Register another experiment with `factory.add("my_key", builder)` where `builder(factory) -> GridRun`.

## Evaluation protocol (short)

- Split seed `42`, stratified 60% / 20% / 20%. Training is balanced per model seed (590 rows).
- Train without noise. Test at `p = 0, 0.001, 0.01, 0.05` (Task 2 headline: **0.01** and **0.05**).
- Threshold = F1-maximising cutoff on **validation at the same `p`**. Never on test.
- Primary metric: **test AUPRC**. Also AUROC, precision, recall, F1, MCC, Spearman ρ (noisy vs clean scores).
- Cost ledger: parameter-shift style `1 + 2P` executions per training row × epochs, plus validation/test passes (1024 shots booked per execution). GA searches add forward counts via `forward_executions`.

## Circuit defaults

| Setting | Full circuit | Chain circuit |
|---------|--------------|---------------|
| Qubits | 4 | 4 |
| Feature map | ZZ, all-to-all, 2 reps | ZZ, nearest-neighbour, 2 reps |
| Ansatz | EfficientSU2, 3 reps → 32 params | EfficientSU2, 1 rep → 16 params |
| Readout | ⟨Z₀⟩ → P(fraud) = (1 − ⟨Z⟩) / 2 | same |
| Noise locations | 126 | 68 |

Two-phase frontend (default tag `ep20_lr0.01_wd0.0001_bs64_sep0.5_spd0.001_m0.5`): 30 → 8 → 4 angles in `[0, π]`, AdamW, separation + spread losses, then the circuit is trained on frozen angles.

## Results files

| File | Contents |
|------|----------|
| `results/task2_runs.csv` | Per-seed metrics for every grid experiment |
| `results/task2_ledger.csv` | Executions / shots / best validation AUPRC |
| `results/two_phase_ga_adam_chained_hparams.csv` | Hparam-sweep metrics |
| `results/two_phase_ga_adam_chained_hparams_cost.csv` | Hparam-sweep costs |
| `results/checkpoint_noise_metrics.csv` | Dense noise grid on saved checkpoints |
| `results/checkpoint_cost.csv` | Full cost for each `model.pth` |
| `experiment_report.txt` | Narrative write-up of runs and methods |

## Notebook

`task1_guided_vqc_fraud_detection.ipynb` is the guided Task 1 baseline. Task 2 work lives in the packages above; keep the same split and `evaluate_under_noise` protocol when you compare numbers.
