"""Stream 1: the three views of a contact table.

1. Snapshot sequence  -> `to_snapshots`: contacts binned into steps of `bin_s` seconds.
2. Aggregated graph   -> `aggregate`: one weighted undirected graph, weight = contact seconds.
3. Per-day splits     -> `split_days`: calibration days vs validation days, never overlapping.

All functions take the contact table produced by `f2f.datasets.load_contacts` and never
modify it in place.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .datasets import WINDOW_S


def to_snapshots(contacts: pd.DataFrame, bin_s: int = WINDOW_S) -> pd.DataFrame:
    """Bin contacts into discrete steps.

    Returns a DataFrame with columns (step, i, j), one row per pair active in that step,
    sorted by step. For bin_s = 20 this is the raw sequence; for larger bins a pair that
    was in contact at least once within the bin counts as active in that step.
    """
    if bin_s % WINDOW_S:
        raise ValueError(f"bin_s must be a multiple of {WINDOW_S}")
    step = (contacts["t"].to_numpy() // bin_s).astype(np.int64)
    snap = pd.DataFrame({"step": step, "i": contacts["i"].to_numpy(), "j": contacts["j"].to_numpy()})
    snap = snap.drop_duplicates().sort_values(["step", "i", "j"]).reset_index(drop=True)
    snap.attrs["bin_s"] = bin_s
    snap.attrs["n_steps"] = int(step.max()) + 1
    return snap


def snapshot_graphs(snapshots: pd.DataFrame):
    """Yield (step, nx.Graph) for every step that has at least one contact."""
    for step, g in snapshots.groupby("step", sort=True):
        G = nx.Graph()
        G.add_edges_from(zip(g["i"], g["j"]))
        yield int(step), G


def aggregate(contacts: pd.DataFrame, nodes: pd.DataFrame | None = None, window_s: int = WINDOW_S) -> nx.Graph:
    """Aggregated weighted graph.

    Edge attributes:
        count   number of 20-s windows the pair was in contact
        weight  total contact time in seconds (= count * window_s)
    Node attribute `group` is copied from `nodes` if given. Isolated nodes that never
    appear in a contact are still added (from `nodes`) so N matches the roster.
    """
    pairs = contacts.groupby(["i", "j"]).size().rename("count").reset_index()
    G = nx.Graph()
    if nodes is not None:
        G.add_nodes_from(nodes.index.tolist())
        nx.set_node_attributes(G, nodes["group"].to_dict(), "group")
    G.add_weighted_edges_from(
        zip(pairs["i"], pairs["j"], (pairs["count"] * window_s).astype(float)), weight="weight"
    )
    nx.set_edge_attributes(G, {(a, b): int(c) for a, b, c in zip(pairs["i"], pairs["j"], pairs["count"])}, "count")
    return G


def day_table(contacts: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: contacts, active nodes, first and last contact time."""
    rows = []
    for day, g in contacts.groupby("day", sort=True):
        active = np.unique(np.concatenate([g["i"].to_numpy(), g["j"].to_numpy()]))
        rows.append(
            {
                "day": int(day),
                "contacts": len(g),
                "active_nodes": len(active),
                "t_start": int(g["t"].min()),
                "t_end": int(g["t"].max()),
                "abs_start": g["abs_time"].min() if "abs_time" in g else None,
                "abs_end": g["abs_time"].max() if "abs_time" in g else None,
            }
        )
    return pd.DataFrame(rows).set_index("day")


def split_days(contacts: pd.DataFrame, calibration_days: list[int] | int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (calibration, validation) by calendar day.

    `calibration_days` is either an explicit list of day indices or an int k meaning
    "the first k days". Validation is every remaining day. The two never overlap.
    """
    days = sorted(contacts["day"].unique())
    if isinstance(calibration_days, int):
        calibration_days = days[:calibration_days]
    cal_set = set(int(d) for d in calibration_days)
    if not cal_set.issubset(set(int(d) for d in days)):
        raise ValueError(f"calibration days {sorted(cal_set)} not all present in {days}")
    if cal_set == set(int(d) for d in days):
        raise ValueError("calibration days cover every day; nothing left for validation")
    cal = contacts[contacts["day"].isin(cal_set)].reset_index(drop=True)
    val = contacts[~contacts["day"].isin(cal_set)].reset_index(drop=True)
    return cal, val


def restrict_to_days(contacts: pd.DataFrame, days: list[int]) -> pd.DataFrame:
    return contacts[contacts["day"].isin(set(days))].reset_index(drop=True)
