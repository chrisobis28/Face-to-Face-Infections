"""Temporal contact statistics used by Stream 3.

The functions in this module operate on the shared contact-table contract.  Memory
curves are computed within calendar days so an overnight closure is not mistaken
for social-network memory.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .datasets import WINDOW_S
from .views import to_snapshots


def contact_episodes(contacts: pd.DataFrame, window_s: int = WINDOW_S) -> pd.DataFrame:
    """Merge consecutive windows of a pair into contact episodes.

    Returns ``i, j, day, start_t, end_t, windows, duration_s``.  Episodes never
    cross calendar-day boundaries.
    """
    required = {"t", "i", "j", "day"}
    missing = required - set(contacts.columns)
    if missing:
        raise ValueError(f"contact table is missing columns: {sorted(missing)}")
    if contacts.empty:
        return pd.DataFrame(
            columns=["i", "j", "day", "start_t", "end_t", "windows", "duration_s"]
        )

    ordered = contacts[["t", "i", "j", "day"]].sort_values(["i", "j", "day", "t"]).copy()
    same_series = (
        ordered["i"].eq(ordered["i"].shift())
        & ordered["j"].eq(ordered["j"].shift())
        & ordered["day"].eq(ordered["day"].shift())
        & ordered["t"].sub(ordered["t"].shift()).eq(window_s)
    )
    ordered["episode"] = (~same_series).cumsum()
    episodes = (
        ordered.groupby("episode", sort=False)
        .agg(i=("i", "first"), j=("j", "first"), day=("day", "first"),
             start_t=("t", "min"), end_t=("t", "max"), windows=("t", "size"))
        .reset_index(drop=True)
    )
    episodes["duration_s"] = episodes["windows"] * window_s
    return episodes


def inter_contact_times(episodes: pd.DataFrame, window_s: int = WINDOW_S) -> np.ndarray:
    """Inactive seconds between successive episodes of the same pair and day."""
    if episodes.empty:
        return np.array([], dtype=np.int64)
    ordered = episodes.sort_values(["i", "j", "day", "start_t"]).copy()
    same_pair_day = (
        ordered["i"].eq(ordered["i"].shift())
        & ordered["j"].eq(ordered["j"].shift())
        & ordered["day"].eq(ordered["day"].shift())
    )
    gaps = ordered["start_t"] - ordered["end_t"].shift() - window_s
    return gaps[same_pair_day].astype(np.int64).to_numpy()


def burstiness(values: np.ndarray) -> float:
    """Return B=(sigma-mu)/(sigma+mu), or NaN when it is undefined."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return (std - mean) / (std + mean) if std + mean else float("nan")


def activity_potential(contacts: pd.DataFrame, node_ids=None) -> pd.DataFrame:
    """Per-node share of all undirected contact involvements (sums to one).

    SocioPatterns contacts have no initiator, so this is the defensible undirected
    analogue of the activity potential used by activity-driven models.
    """
    endpoints = pd.concat([contacts["i"], contacts["j"]], ignore_index=True)
    counts = endpoints.value_counts(sort=False)
    if node_ids is None:
        node_ids = sorted(counts.index)
    out = pd.DataFrame(index=pd.Index(node_ids, name="node"))
    out["contact_involvements"] = counts.reindex(out.index, fill_value=0).astype(int)
    total = int(out["contact_involvements"].sum())
    out["activity_potential"] = out["contact_involvements"] / total if total else 0.0
    return out


def _daily_edge_sets(contacts: pd.DataFrame, bin_s: int) -> list[list[set[tuple[int, int]]]]:
    """Dense within-day sequences of active edge sets, including empty bins."""
    sequences = []
    for _, day_contacts in contacts.groupby("day", sort=True):
        snap = to_snapshots(day_contacts, bin_s)
        first = int(snap["step"].min())
        last = int(snap["step"].max())
        by_step: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for row in snap.itertuples(index=False):
            by_step[int(row.step)].add((int(row.i), int(row.j)))
        sequences.append([by_step[step] for step in range(first, last + 1)])
    return sequences


def memory_curve(
    contacts: pd.DataFrame,
    max_lag: int = 12,
    bin_s: int = 300,
    node_ids=None,
) -> pd.DataFrame:
    """Mean snapshot Jaccard and pooled binary link autocorrelation by lag.

    Autocorrelation pools the 2x2 activity counts over all possible undirected
    node pairs and all within-day snapshot pairs at a given lag.  Comparisons in
    which both snapshots are empty are omitted from both measures.
    """
    sequences = _daily_edge_sets(contacts, bin_s)
    if node_ids is None:
        node_ids = np.unique(np.r_[contacts["i"].to_numpy(), contacts["j"].to_numpy()])
    n_links = len(node_ids) * (len(node_ids) - 1) // 2
    rows = []
    for lag in range(1, max_lag + 1):
        jaccards = []
        n11 = n10 = n01 = n00 = 0
        comparisons = 0
        for seq in sequences:
            for left, right in zip(seq[:-lag], seq[lag:]):
                union = left | right
                if not union:
                    continue
                intersection = len(left & right)
                jaccards.append(intersection / len(union))
                n11 += intersection
                n10 += len(left - right)
                n01 += len(right - left)
                n00 += n_links - len(union)
                comparisons += 1
        denom = np.sqrt(
            float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        )
        autocorr = (n11 * n00 - n10 * n01) / denom if denom else float("nan")
        rows.append(
            {
                "lag": lag,
                "lag_minutes": lag * bin_s / 60,
                "jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
                "autocorrelation": float(autocorr),
                "comparisons": comparisons,
            }
        )
    return pd.DataFrame(rows)


def temporal_summary(contacts: pd.DataFrame, node_ids=None, bin_s: int = 300, max_lag: int = 12) -> dict:
    """Scalar summary used for real-vs-null comparisons and z-scores."""
    episodes = contact_episodes(contacts)
    gaps = inter_contact_times(episodes)
    memory = memory_curve(contacts, max_lag=max_lag, bin_s=bin_s, node_ids=node_ids)
    durations = episodes["duration_s"].to_numpy(dtype=float)
    return {
        "episodes": len(episodes),
        "mean_duration_s": float(durations.mean()) if len(durations) else float("nan"),
        "median_duration_s": float(np.median(durations)) if len(durations) else float("nan"),
        "mean_intercontact_s": float(gaps.mean()) if len(gaps) else float("nan"),
        "intercontact_burstiness": burstiness(gaps),
        "jaccard_lag1": float(memory.loc[memory["lag"] == 1, "jaccard"].iloc[0]),
        "autocorrelation_lag1": float(memory.loc[memory["lag"] == 1, "autocorrelation"].iloc[0]),
        "jaccard_auc": float(np.trapezoid(memory["jaccard"], memory["lag_minutes"])),
        "autocorrelation_auc": float(np.trapezoid(memory["autocorrelation"], memory["lag_minutes"])),
    }
