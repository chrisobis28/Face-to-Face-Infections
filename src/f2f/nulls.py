"""Stream 2: static null models and z-scores (lectures 1 and 2).

configuration_model(G)           random graph with the same degree sequence (stub matching,
                                 then self-loops and multi-links removed, as in the lecture)
degree_preserving_rewiring(G)    double-edge swaps: exactly the same degrees, random wiring
rewire_to_assortativity(G, rho)  degree-preserving swaps accepted only when they move the
                                 assortativity toward a target (lecture 1)
null_ensemble(G, n, kind)        metrics over n realisations
zscores(real, null_df)           (real - mean_null) / std_null with an empirical p-value

Null graphs are unweighted; the group attribute is kept on the nodes so that
`modularity_groups` on a null graph answers "is the class/ward structure stronger than
chance given the degrees?".
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .static_metrics import NULL_METRICS, network_metrics


def _unweighted_with_groups(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges())
    return H


def configuration_model(G: nx.Graph, seed: int | None = None) -> nx.Graph:
    """Configuration model: give each node d_i stubs, pair stubs at random, drop loops and duplicates."""
    nodes = list(G.nodes())
    degs = [d for _, d in G.degree()]
    M = nx.configuration_model(degs, seed=seed)  # multigraph with loops
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    H.add_edges_from((u, v) for u, v in M.edges() if u != v)
    mapping = {k: nodes[k] for k in range(len(nodes))}
    H = nx.relabel_nodes(H, mapping)
    for n, data in G.nodes(data=True):
        H.nodes[n].update(data)
    return H


def degree_preserving_rewiring(G: nx.Graph, swaps_per_link: float = 10, seed: int | None = None) -> nx.Graph:
    """Random degree-preserving rewiring (double-edge swaps). Degrees are preserved exactly."""
    H = _unweighted_with_groups(G)
    L = H.number_of_edges()
    if L < 2:
        return H
    nswap = int(swaps_per_link * L)
    nx.double_edge_swap(H, nswap=nswap, max_tries=nswap * 100, seed=seed)
    return H


def _assortativity_numerator(H: nx.Graph) -> float:
    return float(sum(H.degree(u) * H.degree(v) for u, v in H.edges()))


def rewire_to_assortativity(
    G: nx.Graph, target_rho: float, max_steps: int = 20000, seed: int | None = None, tol: float = 1e-3
) -> nx.Graph:
    """Degree-preserving rewiring that drives the assortativity rho_D toward `target_rho`.

    With degrees fixed, rho_D is an increasing function of sum_{links} d_i d_j, so a swap
    (a-b, c-d) -> (a-c, b-d) is accepted when it moves that sum in the right direction
    (lecture 1, "only two terms change"). Stops when |rho - target| < tol or max_steps.
    """
    rng = np.random.default_rng(seed)
    H = _unweighted_with_groups(G)
    deg = dict(H.degree())
    edges = [tuple(e) for e in H.edges()]
    S = _assortativity_numerator(H)
    rho = nx.degree_assortativity_coefficient(H)
    direction = 1 if target_rho > rho else -1
    for _ in range(max_steps):
        if abs(rho - target_rho) < tol:
            break
        k1, k2 = rng.integers(0, len(edges), size=2)
        if k1 == k2:
            continue
        a, b = edges[k1]
        c, d = edges[k2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or H.has_edge(a, c) or H.has_edge(b, d):
            continue
        delta = deg[a] * deg[c] + deg[b] * deg[d] - deg[a] * deg[b] - deg[c] * deg[d]
        if delta * direction <= 0:
            continue
        H.remove_edge(a, b)
        H.remove_edge(c, d)
        H.add_edge(a, c)
        H.add_edge(b, d)
        edges[k1], edges[k2] = (a, c), (b, d)
        S += delta
        rho = nx.degree_assortativity_coefficient(H)
    return H


NULL_KINDS = {
    "configuration": configuration_model,
    "rewired": degree_preserving_rewiring,
}


def null_ensemble(
    G: nx.Graph, n: int = 100, kind: str = "rewired", seed: int = 0, metrics: list[str] | None = None
) -> pd.DataFrame:
    """Compute `network_metrics` on n null realisations. Returns one row per realisation."""
    if kind not in NULL_KINDS:
        raise ValueError(f"kind must be one of {sorted(NULL_KINDS)}")
    metrics = metrics or NULL_METRICS
    make = NULL_KINDS[kind]
    rows = []
    for r in range(n):
        H = make(G, seed=seed + r)
        m = network_metrics(H, seed=seed + r)
        rows.append({k: m[k] for k in metrics} | {"realisation": r, "kind": kind})
    return pd.DataFrame(rows).set_index("realisation")


def zscores(real: dict, null_df: pd.DataFrame) -> pd.DataFrame:
    """Compare the real metrics with the null ensemble.

    Columns: real, null_mean, null_std, null_q025, null_q975, z, p_two_sided (empirical,
    with +1 smoothing so it is never exactly 0), rel_diff = (real - null_mean) / |null_mean|.

    Read z as "is this explained by the degrees alone?" (|z| > 3: no) and rel_diff as
    "by how much does it matter?". A tiny null spread can give z in the hundreds for a
    difference of one percent, so never rank metrics by z.
    """
    rows = []
    n = len(null_df)
    for k in null_df.columns:
        if k in ("kind",) or k not in real:
            continue
        x = null_df[k].to_numpy(dtype=float)
        mu, sd = x.mean(), x.std(ddof=1) if n > 1 else float("nan")
        z = (real[k] - mu) / sd if sd and sd > 0 else float("inf") if real[k] != mu else 0.0
        more_extreme = np.sum(np.abs(x - mu) >= abs(real[k] - mu))
        p = (more_extreme + 1) / (n + 1)
        rows.append(
            {
                "metric": k,
                "real": real[k],
                "null_mean": mu,
                "null_std": sd,
                "null_q025": np.quantile(x, 0.025),
                "null_q975": np.quantile(x, 0.975),
                "z": z,
                "p_two_sided": p,
                "rel_diff": (real[k] - mu) / abs(mu) if mu != 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
