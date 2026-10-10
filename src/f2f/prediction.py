"""Self-driven link-activity prediction for Stream 3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .views import to_snapshots


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, treating equal scores as one threshold."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked_y = y_true[order]
    ranked_scores = scores[order]
    ends = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), len(scores) - 1]
    true_positives = np.cumsum(ranked_y)[ends]
    precision = true_positives / (ends + 1)
    recall = true_positives / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def self_driven_scores(history: np.ndarray, tau: float) -> np.ndarray:
    """Exponentially weighted link activity, most recent row last.

    This implements w_l(t+1)=sum_k exp[-tau(t-k)] x_l(k).  ``tau=0`` is
    the unweighted historical-frequency baseline.
    """
    if history.ndim != 2:
        raise ValueError("history must be a (time, link) binary matrix")
    ages = np.arange(len(history) - 1, -1, -1, dtype=float)
    return np.exp(-tau * ages) @ history


def _active_sequence(contacts: pd.DataFrame, bin_s: int):
    snap = to_snapshots(contacts, bin_s)
    edges = sorted(set(zip(snap["i"].astype(int), snap["j"].astype(int))))
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    steps = np.sort(snap["step"].unique())
    step_index = {int(step): idx for idx, step in enumerate(steps)}
    activity = np.zeros((len(steps), len(edges)), dtype=np.uint8)
    for row in snap.itertuples(index=False):
        activity[step_index[int(row.step)], edge_index[(int(row.i), int(row.j))]] = 1
    day_by_step = contacts.assign(step=contacts["t"] // bin_s).groupby("step")["day"].first()
    days = np.array([int(day_by_step.loc[step]) for step in steps])
    return edges, steps, days, activity


def evaluate_self_driven(
    contacts: pd.DataFrame,
    bin_s: int = 300,
    lookback: int = 24,
    taus=(0.01, 0.03, 0.1, 0.3, 1.0),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune on day 0, then rank links for each following calendar day.

    At each day boundary, candidate links are only those seen before that day.  The
    target is whether each candidate appears at least once during the next day.
    ``candidate_coverage`` reports the share of all target-day links that had been
    seen previously, so genuinely new links are not silently ignored.  Empty global
    bins are omitted from the memory history, matching the self-driven formulation.

    Returns ``(evaluation_rows, predicted_contacts, metadata)``.  The prediction
    table contains every candidate link and its rank, allowing Stream 6 to choose
    its own intervention budget.
    """
    edges, steps, days, activity = _active_sequence(contacts, bin_s)
    calibration_idx = np.flatnonzero(days == days.min())
    validation_idx = np.flatnonzero(days > days.min())
    if len(validation_idx) == 0:
        raise ValueError("prediction requires at least two days and calibration links")

    # Tune tau by predicting the final third of day 0 from its earlier snapshots.
    cut_position = max(1, 2 * len(calibration_idx) // 3)
    tune_target = calibration_idx[cut_position:]
    tune_end = int(tune_target[0])
    tune_candidates = np.flatnonzero(activity[:tune_end].any(axis=0))
    tune_truth = activity[np.ix_(tune_target, tune_candidates)].any(axis=0)
    history_start = max(0, tune_end - lookback)
    tune_rows = []
    for tau in taus:
        scores = self_driven_scores(activity[history_start:tune_end, tune_candidates], tau)
        tune_rows.append((float(tau), average_precision(tune_truth, scores)))
    best_tau, calibration_ap = max(tune_rows, key=lambda item: item[1])

    evaluation = []
    predictions = []
    for day in np.unique(days[validation_idx]):
        target_idx = np.flatnonzero(days == day)
        target_start = int(target_idx[0])
        candidate_idx = np.flatnonzero(activity[:target_start].any(axis=0))
        start = max(0, target_start - lookback)
        history = activity[start:target_start, candidate_idx]
        truth = activity[np.ix_(target_idx, candidate_idx)].any(axis=0)
        target_total = int(activity[target_idx].any(axis=0).sum())
        coverage = float(truth.sum() / target_total) if target_total else float("nan")
        methods = {
            "self_driven": self_driven_scores(history, best_tau),
            "frequency_baseline": self_driven_scores(history, 0.0),
        }
        for method, scores in methods.items():
            evaluation.append(
                {
                    "day": int(day),
                    "method": method,
                    "average_precision": average_precision(truth, scores),
                    "candidate_coverage": coverage,
                    "positives": int(truth.sum()),
                    "candidates": len(candidate_idx),
                }
            )
        scores = methods["self_driven"]
        order = np.lexsort((candidate_idx, -scores))
        for rank, local_idx in enumerate(order, start=1):
            edge = edges[candidate_idx[local_idx]]
            predictions.append(
                {
                    "day": int(day),
                    "i": edge[0],
                    "j": edge[1],
                    "score": float(scores[local_idx]),
                    "rank": rank,
                }
            )
    metadata = {
        "best_tau": best_tau,
        "calibration_ap": calibration_ap,
        "lookback": lookback,
        "bin_s": bin_s,
        "calibration_links": int(activity[calibration_idx].any(axis=0).sum()),
    }
    return pd.DataFrame(evaluation), pd.DataFrame(predictions), metadata
