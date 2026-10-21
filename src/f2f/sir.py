"""Batched discrete-time SIR on the contact table, with immunisation (streams 4 and 6).

Implements the simulator reserved in PLAN.md, `simulate(contacts, seeds, beta, gamma,
immunised, runs, rng_seed)`, with the same dynamics as
`scripts/spreading.simulate_discrete_temporal`: one step is one 20-s window, every S-I
contact in a window transmits with probability `beta`, every infected node recovers with
probability `gamma` per window. Three differences, all for speed or for longer outbreaks:

- all runs advance together as one (runs x N) array;
- state is stored as times: node v in run r is infected in window inf[r, v] and infectious
  in windows inf < w <= rec. Recovery delays are drawn up front (geometric with parameter
  gamma), which is exactly the per-window coin flip, so windows without contacts are
  skipped with no approximation;
- the recording is repeated periodically when the outbreak outlasts it, as in
  Zhang, Zhao & Wang (2022) and Starnini et al. (2013).

Every window draws the same amount of randomness whatever the state, so two calls with
the same `rng_seed` share random numbers (common random numbers): differences between
immunisation strategies are not run-to-run noise.

`simulate_aggregated` runs the same process on the time-aggregated network: link (i, j)
carries its c_ij contacts spread evenly over the recording, so the only thing removed is
the timing of contacts (same beta, gamma and recovery draws).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import WINDOW_S

NEVER = np.iinfo(np.int64).max // 4  # infection time of a node that is never infected
IMMUNE = -2  # inf = rec = IMMUNE: never susceptible, never infectious
WINDOWS_PER_HOUR = 3600 // WINDOW_S


@dataclass
class Epidemic:
    inf: np.ndarray  # (runs, N) infection window, NEVER if never infected, IMMUNE if immunised
    rec: np.ndarray  # (runs, N) last infectious window
    period: int  # windows in one pass of the recording
    still_active: float  # fraction of runs with infectious nodes when the simulation stopped

    @property
    def n(self) -> int:
        return self.inf.shape[1]

    def final_size(self) -> np.ndarray:
        """Fraction of the population ever infected (seeds included), per run."""
        return ((self.inf != NEVER) & (self.inf != IMMUNE)).sum(axis=1) / self.n

    def prevalence(self, step: int = WINDOWS_PER_HOUR) -> tuple[np.ndarray, np.ndarray]:
        """(grid in windows, (runs, len(grid)) fraction infectious) on a regular grid."""
        horizon = int(self.rec.max()) + 1
        grid = np.arange(0, horizon + step, step)
        out = np.empty((self.inf.shape[0], len(grid)))
        infected = (self.inf != NEVER) & (self.inf != IMMUNE)
        for r in range(self.inf.shape[0]):
            started = np.sort(self.inf[r][infected[r]])
            ended = np.sort(self.rec[r][infected[r]])
            out[r] = np.searchsorted(started, grid, "left") - np.searchsorted(ended, grid, "left")
        return grid, out / self.n

    def time_to(self, fraction: float, step: int = WINDOWS_PER_HOUR) -> np.ndarray:
        """Hours until `fraction` of the population has been infected (NaN if never)."""
        t = np.sort(np.where(self.inf > IMMUNE, self.inf, NEVER), axis=1)
        k = int(np.ceil(fraction * self.n)) - 1
        hit = t[:, k].astype(float)
        hit[hit >= NEVER] = np.nan
        return (hit + 1) / step

    def summary(self) -> dict[str, np.ndarray]:
        grid, prev = self.prevalence()
        return {
            "final_size": self.final_size(),
            "peak": prev.max(axis=1),
            "peak_time_h": grid[prev.argmax(axis=1)] / WINDOWS_PER_HOUR,
            "t10_h": self.time_to(0.10),
            "t50_h": self.time_to(0.50),
        }


def _start(n: int, seeds, immunised, runs: int, gamma: float, rng_seed: int):
    """Initial times and pre-drawn recovery delays. `seeds` is an int k (k random seeds per
    run, drawn among non-immunised nodes), an array of node indices used in every run, or a
    (runs, N) boolean mask. `immunised` is None, an index array (same in every run) or a
    (runs, N) boolean mask."""
    rng = np.random.default_rng(rng_seed)
    delay = rng.geometric(gamma, size=(runs, n)).astype(np.int64)
    order = np.argsort(rng.random((runs, n)), axis=1)  # per-run seed candidates, CRN-friendly
    imm = np.zeros((runs, n), dtype=bool)
    if immunised is not None:
        immunised = np.asarray(immunised)
        if immunised.dtype == bool:
            imm[:] = immunised
        else:
            imm[:, immunised] = True
    inf = np.full((runs, n), NEVER, dtype=np.int64)
    rec = np.full((runs, n), -3, dtype=np.int64)
    for r in range(runs):
        if np.isscalar(seeds):
            cand = order[r][~imm[r][order[r]]]
            s = cand[: int(seeds)]
        else:
            s = np.asarray(seeds)
            s = np.flatnonzero(s[r]) if s.ndim == 2 else s
            if imm[r, s].any():
                raise ValueError("a seed node is immunised")
        inf[r, s] = -1  # infectious from window 0, like the original engine
        rec[r, s] = -1 + delay[r, s]
    inf[imm] = IMMUNE
    rec[imm] = IMMUNE
    return rng, delay, inf, rec


def windows_of(contacts: pd.DataFrame, node_index: pd.Index):
    """Contact arrays grouped by window: (a, b, window of each block, block bounds, period)."""
    w = ((contacts["t"] - contacts["t"].min()) // WINDOW_S).to_numpy()
    order = np.argsort(w, kind="stable")
    w = w[order]
    a = node_index.get_indexer(contacts["i"].to_numpy()[order])
    b = node_index.get_indexer(contacts["j"].to_numpy()[order])
    if (a < 0).any() or (b < 0).any():
        raise ValueError("contact endpoints missing from the node table")
    uw, start = np.unique(w, return_index=True)
    bounds = np.r_[start, len(w)]
    return a, b, uw.astype(np.int64), bounds, int(w.max()) + 1


def simulate(
    contacts: pd.DataFrame,
    nodes: pd.DataFrame,
    seeds=1,
    beta: float = 0.1,
    gamma: float = 0.0005,
    immunised=None,
    runs: int = 100,
    rng_seed: int = 0,
    max_loops: int = 30,
    prepared=None,
) -> Epidemic:
    """SIR on the temporal network. Node order is `nodes.index`; pass `prepared` (the output
    of `windows_of`) to skip regrouping the contacts on repeated calls."""
    a, b, uw, bounds, period = prepared if prepared is not None else windows_of(contacts, nodes.index)
    rng, delay, inf, rec = _start(len(nodes), seeds, immunised, runs, gamma, rng_seed)
    for loop in range(max_loops):
        offset = loop * period
        for k in range(len(uw)):
            w = uw[k] + offset
            if k % 256 == 0 and not (rec >= w).any():
                return Epidemic(inf, rec, period, 0.0)
            s, e = bounds[k], bounds[k + 1]
            pa, pb = a[s:e], b[s:e]
            ia, ra, ib, rb = inf[:, pa], rec[:, pa], inf[:, pb], rec[:, pb]
            u = rng.random((runs, 2 * (e - s)))
            hit_b = (ia < w) & (w <= ra) & (ib == NEVER) & (u[:, : e - s] < beta)
            hit_a = (ib < w) & (w <= rb) & (ia == NEVER) & (u[:, e - s :] < beta)
            if hit_a.any() or hit_b.any():
                rb_, cb_ = np.nonzero(hit_b)
                ra_, ca_ = np.nonzero(hit_a)
                rows = np.r_[rb_, ra_]
                cols = np.r_[pb[cb_], pa[ca_]]
                inf[rows, cols] = w
                rec[rows, cols] = w + delay[rows, cols]
    end = max_loops * period
    return Epidemic(inf, rec, period, float((rec >= end).any(axis=1).mean()))


def contact_matrix(contacts: pd.DataFrame, node_index: pd.Index) -> np.ndarray:
    """Dense symmetric matrix of contact counts (number of 20-s windows per pair)."""
    n = len(node_index)
    c = np.zeros((n, n))
    i = node_index.get_indexer(contacts["i"].to_numpy())
    j = node_index.get_indexer(contacts["j"].to_numpy())
    np.add.at(c, (i, j), 1.0)
    return c + c.T


def simulate_aggregated(
    contacts: pd.DataFrame,
    nodes: pd.DataFrame,
    seeds=1,
    beta: float = 0.1,
    gamma: float = 0.0005,
    immunised=None,
    runs: int = 100,
    rng_seed: int = 0,
    max_loops: int = 30,
    step: int = WINDOWS_PER_HOUR,
) -> Epidemic:
    """SIR on the aggregated weighted network, in steps of `step` windows. During a step a
    susceptible i is infected with probability 1 - (1 - beta)^x, x = sum_j c_ij / period *
    (windows j was infectious in the step): the expected number of contacts with infectious
    nodes when contacts are spread evenly in time. Infection lands at the end of the step."""
    c = contact_matrix(contacts, nodes.index)
    period = int((contacts["t"].max() - contacts["t"].min()) // WINDOW_S) + 1
    rng, delay, inf, rec = _start(len(nodes), seeds, immunised, runs, gamma, rng_seed)
    log1mb = np.log1p(-min(beta, 1 - 1e-12)) / period  # beta = 1 would give -inf * 0
    for k in range(int(np.ceil(max_loops * period / step))):
        s0, s1 = k * step, (k + 1) * step
        if not (rec >= s0).any():
            return Epidemic(inf, rec, period, 0.0)
        lo = np.maximum(inf + 1, s0)
        hi = np.minimum(rec, s1 - 1)
        exposure = np.clip(hi - lo + 1, 0, None).astype(float)  # infectious windows in the step
        p = -np.expm1(log1mb * (exposure @ c))
        new = (inf == NEVER) & (rng.random(inf.shape) < p)
        inf[new] = s1 - 1
        rec[new] = s1 - 1 + delay[new]
    end = max_loops * period
    return Epidemic(inf, rec, period, float((rec >= end).any(axis=1).mean()))
