"""Temporal null models with explicit, testable invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _replace_pairs(contacts: pd.DataFrame, pairs: np.ndarray) -> pd.DataFrame:
    out = contacts.copy()
    out[["i", "j"]] = np.sort(pairs, axis=1)
    return out.sort_values(["t", "i", "j"]).reset_index(drop=True)


def time_shuffle(
    contacts: pd.DataFrame,
    seed: int = 0,
    swaps_per_contact: float = 3.0,
) -> pd.DataFrame:
    """Randomise pair-time assignment while keeping times and pair counts exactly.

    Pair labels are swapped between event rows.  Moves that create a duplicate
    pair at one timestamp are rejected, so every output row remains one observed
    20-second pair-window.
    """
    rng = np.random.default_rng(seed)
    times = contacts["t"].to_numpy(copy=True)
    pairs = contacts[["i", "j"]].to_numpy(copy=True)
    occupied = {(int(t), int(i), int(j)) for t, (i, j) in zip(times, pairs)}
    n = len(contacts)
    for _ in range(int(swaps_per_contact * n)):
        a, b = rng.integers(0, n, size=2)
        if a == b or times[a] == times[b] or np.array_equal(pairs[a], pairs[b]):
            continue
        pa = tuple(int(x) for x in pairs[a])
        pb = tuple(int(x) for x in pairs[b])
        new_a = (int(times[a]), *pb)
        new_b = (int(times[b]), *pa)
        old_a = (int(times[a]), *pa)
        old_b = (int(times[b]), *pb)
        if (new_a in occupied and new_a not in {old_a, old_b}) or (
            new_b in occupied and new_b not in {old_a, old_b}
        ):
            continue
        occupied.remove(old_a)
        occupied.remove(old_b)
        occupied.add(new_a)
        occupied.add(new_b)
        pairs[[a, b]] = pairs[[b, a]]
    return _replace_pairs(contacts, pairs)


def link_shuffle(
    contacts: pd.DataFrame,
    seed: int = 0,
    swaps_per_contact: float = 3.0,
) -> pd.DataFrame:
    """Randomise partners while keeping timestamps and node activity exactly.

    Activity-preserving endpoint swaps act on temporal event rows. Moves producing
    self-contacts or duplicate pair-windows are rejected. Each node's total event
    count is preserved, but the aggregate topology is deliberately *not* preserved.
    """
    rng = np.random.default_rng(seed)
    times = contacts["t"].to_numpy(copy=True)
    pairs = contacts[["i", "j"]].to_numpy(copy=True)
    occupied = {(int(t), int(i), int(j)) for t, (i, j) in zip(times, pairs)}
    n = len(contacts)
    for _ in range(int(swaps_per_contact * n)):
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        edge_a = pairs[a].copy()
        edge_b = pairs[b].copy()
        if rng.random() < 0.5:
            edge_a = edge_a[::-1]
        if rng.random() < 0.5:
            edge_b = edge_b[::-1]
        proposed_a = tuple(sorted((int(edge_a[0]), int(edge_b[1]))))
        proposed_b = tuple(sorted((int(edge_b[0]), int(edge_a[1]))))
        if proposed_a[0] == proposed_a[1] or proposed_b[0] == proposed_b[1]:
            continue
        old_a = (int(times[a]), *tuple(int(x) for x in pairs[a]))
        old_b = (int(times[b]), *tuple(int(x) for x in pairs[b]))
        new_a = (int(times[a]), *proposed_a)
        new_b = (int(times[b]), *proposed_b)
        if new_a == new_b or (
            new_a in occupied and new_a not in {old_a, old_b}
        ) or (new_b in occupied and new_b not in {old_a, old_b}):
            continue
        occupied.remove(old_a)
        occupied.remove(old_b)
        occupied.add(new_a)
        occupied.add(new_b)
        pairs[a] = proposed_a
        pairs[b] = proposed_b
    return _replace_pairs(contacts, pairs)


def assert_time_shuffle_invariants(original: pd.DataFrame, shuffled: pd.DataFrame) -> None:
    """Raise when a time-shuffled table violates its scientific definition."""
    pd.testing.assert_series_equal(
        original.groupby("t").size().sort_index(),
        shuffled.groupby("t").size().sort_index(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        original.groupby(["i", "j"]).size().sort_index(),
        shuffled.groupby(["i", "j"]).size().sort_index(),
        check_names=False,
    )
    if shuffled.duplicated(["t", "i", "j"]).any():
        raise AssertionError("time shuffle created duplicate pair-windows")


def assert_link_shuffle_invariants(original: pd.DataFrame, shuffled: pd.DataFrame) -> None:
    """Raise when a link-shuffled table violates its scientific definition."""
    pd.testing.assert_series_equal(
        original.groupby("t").size().sort_index(),
        shuffled.groupby("t").size().sort_index(),
        check_names=False,
    )
    before = pd.concat([original["i"], original["j"]]).value_counts().sort_index()
    after = pd.concat([shuffled["i"], shuffled["j"]]).value_counts().sort_index()
    pd.testing.assert_series_equal(before, after, check_names=False)
    if (shuffled["i"] >= shuffled["j"]).any():
        raise AssertionError("link shuffle created a self-loop or unsorted pair")
    if shuffled.duplicated(["t", "i", "j"]).any():
        raise AssertionError("link shuffle created duplicate pair-windows")
