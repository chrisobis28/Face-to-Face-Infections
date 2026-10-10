"""Stream 1: the shared result cache.

Every stream writes simulation outcomes in ONE row format so figures can be made across
streams without asking anyone. One parquet file per experiment under results/.

Required columns (the contract agreed at kick-off):
    dataset   str    hospital | primary_school | high_school
    version   str    real | time_shuffled | episode_time_shuffled | link_shuffled | aggregated | surrogate
    beta      float  per-window transmission probability
    gamma     float  per-window recovery probability
    seed      int    seed node id (or -1 when not seed-specific)
    run       int    realisation index
    outcome   str    name of the measured quantity (final_size, peak, t10, ...)
    value     float  the measurement
Extra columns are allowed (e.g. strategy, budget, metric) and are kept.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .datasets import ROOT

RESULTS_DIR = ROOT / "results"
REQUIRED = ["dataset", "version", "beta", "gamma", "seed", "run", "outcome", "value"]
VERSIONS = {
    "real", "time_shuffled", "episode_time_shuffled", "link_shuffled",
    "aggregated", "surrogate",
}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"result rows are missing required columns: {missing}")
    bad = set(df["version"].unique()) - VERSIONS
    if bad:
        raise ValueError(f"unknown network version(s) {sorted(bad)}; allowed: {sorted(VERSIONS)}")
    return df


def result_path(experiment: str) -> Path:
    return RESULTS_DIR / f"{experiment}.parquet"


def save_result(df: pd.DataFrame, experiment: str, append: bool = False) -> Path:
    """Write (or append to) results/<experiment>.parquet after validating the schema."""
    validate(df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(experiment)
    if append and path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_result(experiment: str) -> pd.DataFrame:
    return pd.read_parquet(result_path(experiment))


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Non-simulation tables (metric tables, centralities, z-scores) go to results/<name>.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path)
    return path
