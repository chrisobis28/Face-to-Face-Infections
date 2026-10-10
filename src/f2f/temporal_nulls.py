"""Temporal null models with explicit, testable invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import WINDOW_S
from .temporal_metrics import contact_episodes


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


def episode_time_shuffle(
    contacts: pd.DataFrame,
    seed: int = 0,
    max_attempts: int = 1000,
    max_restarts: int = 20,
) -> pd.DataFrame:
    """Move intact contact episodes to other active periods in the recording.

    Every episode keeps its pair and duration, so the aggregate weighted graph and
    duration distribution are exact invariants. Candidate placements may be on a
    different day, but every 20-second window must fall at a timestamp where the
    original dataset had some activity. Episodes of the same pair may not overlap
    or become adjacent, since either case would change the episode distribution.

    Unlike :func:`time_shuffle`, this null does not preserve the number of contacts
    at each timestamp. It isolates ordering effects without mechanically breaking
    long meetings into individual windows.
    """
    if contacts.empty:
        return contacts.copy()
    rng = np.random.default_rng(seed)
    episodes = contact_episodes(contacts)

    active_times = np.sort(contacts["t"].unique().astype(np.int64))
    time_day = contacts.groupby("t")["day"].first().astype(int).to_dict()
    active_set = set(int(t) for t in active_times)

    # Number of consecutive globally active 20-second slots starting at each t.
    forward_run: dict[int, int] = {}
    for t in active_times[::-1]:
        t = int(t)
        next_t = t + WINDOW_S
        if next_t in active_set and time_day[next_t] == time_day[t]:
            forward_run[t] = 1 + forward_run[next_t]
        else:
            forward_run[t] = 1
    candidate_cache = {
        int(w): np.array([t for t in active_times if forward_run[int(t)] >= int(w)], dtype=np.int64)
        for w in episodes["windows"].unique()
    }

    placements: list[tuple[int, int, int, int, int]] = []
    for (i, j), pair_episodes in episodes.groupby(["i", "j"], sort=False):
        specs = list(pair_episodes[["start_t", "windows"]].itertuples(index=False, name=None))
        placed = None
        for _ in range(max_restarts):
            occupied: set[int] = set()
            attempt: dict[int, int] = {}
            tie_break = rng.random(len(specs))
            order = sorted(range(len(specs)), key=lambda idx: (-int(specs[idx][1]), tie_break[idx]))
            failed = False
            for idx in order:
                windows = int(specs[idx][1])
                candidates = candidate_cache[windows]
                chosen = None
                for _ in range(max_attempts):
                    start = int(candidates[rng.integers(0, len(candidates))])
                    end = start + (windows - 1) * WINDOW_S
                    slots = range(start, end + WINDOW_S, WINDOW_S)
                    if start - WINDOW_S in occupied or end + WINDOW_S in occupied:
                        continue
                    if any(t in occupied for t in slots):
                        continue
                    chosen = start
                    occupied.update(slots)
                    attempt[idx] = start
                    break
                if chosen is None:
                    failed = True
                    break
            if not failed:
                placed = attempt
                break
        if placed is None:
            raise RuntimeError(
                f"could not place all intact episodes for pair {(int(i), int(j))}; "
                "increase max_attempts or max_restarts"
            )
        for idx, (_, windows) in enumerate(specs):
            start = placed[idx]
            placements.append((int(i), int(j), start, int(windows), int(time_day[start])))

    rows = []
    for i, j, start, windows, day in placements:
        rows.extend(
            (start + offset * WINDOW_S, i, j, day)
            for offset in range(windows)
        )
    shuffled = pd.DataFrame(rows, columns=["t", "i", "j", "day"])
    if "abs_time" in contacts:
        origin = contacts["abs_time"].iloc[0] - pd.to_timedelta(int(contacts["t"].iloc[0]), unit="s")
        shuffled["abs_time"] = origin + pd.to_timedelta(shuffled["t"], unit="s")
    return shuffled.reindex(columns=contacts.columns).sort_values(["t", "i", "j"]).reset_index(drop=True)


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


def assert_episode_time_shuffle_invariants(original: pd.DataFrame, shuffled: pd.DataFrame) -> None:
    """Raise when an episode-preserving shuffle violates its definition."""
    pd.testing.assert_series_equal(
        original.groupby(["i", "j"]).size().sort_index(),
        shuffled.groupby(["i", "j"]).size().sort_index(),
        check_names=False,
    )
    original_episodes = contact_episodes(original)
    shuffled_episodes = contact_episodes(shuffled)
    before = (
        original_episodes.groupby(["i", "j"])["duration_s"]
        .apply(lambda values: tuple(sorted(values)))
        .sort_index()
    )
    after = (
        shuffled_episodes.groupby(["i", "j"])["duration_s"]
        .apply(lambda values: tuple(sorted(values)))
        .sort_index()
    )
    pd.testing.assert_series_equal(before, after, check_names=False)
    if not set(shuffled["t"]).issubset(set(original["t"])):
        raise AssertionError("episode shuffle placed contacts outside observed active timestamps")
    if (shuffled["i"] >= shuffled["j"]).any():
        raise AssertionError("episode shuffle created a self-loop or unsorted pair")
    if shuffled.duplicated(["t", "i", "j"]).any():
        raise AssertionError("episode shuffle created duplicate pair-windows")
