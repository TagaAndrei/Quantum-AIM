"""Locations that do not depend on the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "creditcard.csv"
RESULTS = ROOT / "results"
TRAINED_MODELS = ROOT / "models" / "trained"
