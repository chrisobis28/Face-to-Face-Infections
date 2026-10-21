"""Stream 6: node immunisation strategies (RQ4) and the network left after immunisation.

Every strategy turns the contact data into the set of f*N nodes to immunise at day 0.
Deterministic strategies rank nodes by a score (top-k, ties broken by strength); the
local strategies (random, acquaintance, weight) draw a new set in every run.

| strategy         | score / rule                                         | source                         |
|------------------|------------------------------------------------------|--------------------------------|
| random           | uniform                                              | Pastor-Satorras & Vespignani 2002 |
| acquaintance     | random partner of a random node                      | Cohen, Havlin & ben-Avraham 2003 |
| weight           | most frequent partner of a random node               | Lee et al. 2012                |
| degree           | distinct partners                                    | Pastor-Satorras & Vespignani 2002 |
| strength         | contacts (= activity potential, same ranking)        | Liu et al. 2014                |
| betweenness      | aggregated, unweighted                               | Holme et al. 2002              |
| eigenvector      | aggregated, weighted                                 | Van Mieghem et al. 2009 (1/λ1) |
| linkers          | share of links outside own group closest to 1/2      | Génois et al. 2015             |
| temporal_reach   | time-respecting out-component within a day           | Holme 2005                     |
| transmissibility | sum_j 1 - (1 - beta)^(c_ij D / T), D = 1/gamma       | Newman 2002 (proposed here)    |
| oracle           | simulated drop in final size when only i is immunised | Starnini et al. 2013 (τ-protocol) |

Link weights are contact counts (number of 20-s windows), as asked in the midterm feedback.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nxc

from . import static_metrics, temporal_metrics, views
from .datasets import WINDOW_S

RANDOMISED = ["random", "acquaintance", "weight"]
RANKED = ["degree", "strength", "betweenness", "eigenvector", "linkers", "temporal_reach", "transmissibility"]
STRATEGIES = RANDOMISED + RANKED + ["oracle"]


def temporal_reach(contacts: pd.DataFrame, node_index: pd.Index) -> pd.Series:
    """Mean over days of the fraction of nodes reachable from i through time-respecting
    paths that day (one hop per 20-s window, as in the SIR engine). Holme (2005)."""
    n = len(node_index)
    total = np.zeros(n)
    days = sorted(contacts["day"].unique())
    for _, day in contacts.sort_values("t", kind="stable").groupby("day"):
        reach = np.eye(n, dtype=bool)  # reach[v, src]: src reaches v
        a = node_index.get_indexer(day["i"].to_numpy())
        b = node_index.get_indexer(day["j"].to_numpy())
        t = day["t"].to_numpy()
        bounds = np.r_[0, np.flatnonzero(np.diff(t)) + 1, len(t)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            pa, pb = a[s:e], b[s:e]
            va, vb = reach[pa], reach[pb]  # copies: one hop per window
            np.logical_or.at(reach, pb, va)
            np.logical_or.at(reach, pa, vb)
        total += reach.sum(axis=0) - 1
    return pd.Series(total / (len(days) * (n - 1)), index=node_index)


def node_temporal_properties(contacts: pd.DataFrame, node_index: pd.Index) -> pd.DataFrame:
    """Per node: number of meetings (contact episodes), mean meeting length in seconds, and
    burstiness of the times between the starts of its meetings within a day (stream 3 code)."""
    ep = temporal_metrics.contact_episodes(contacts)
    both = pd.concat([ep.rename(columns={"i": "node"}), ep.rename(columns={"j": "node"})], ignore_index=True)
    both = both.sort_values(["node", "day", "start_t"])
    both["gap"] = both.groupby(["node", "day"])["start_t"].diff()
    by_node = both.groupby("node")
    out = pd.DataFrame(index=node_index)
    out["meetings"] = by_node.size()
    out["mean_meeting_s"] = by_node["duration_s"].mean()
    out["burstiness"] = by_node["gap"].apply(lambda g: temporal_metrics.burstiness(g.dropna().to_numpy()))
    return out


def node_scores(contacts: pd.DataFrame, nodes: pd.DataFrame, beta: float, gamma: float) -> pd.DataFrame:
    """One row per node, one column per ranked strategy (higher = immunise first)."""
    contacts = contacts.sort_values("t", kind="stable")
    G = views.aggregate(contacts, nodes)
    cent = static_metrics.centralities(G).reindex(nodes.index)
    count = {(u, v): d["count"] for u, v, d in G.edges(data=True)}
    period = (contacts["t"].max() - contacts["t"].min()) // WINDOW_S + 1
    group = nodes["group"]

    out = pd.DataFrame(index=nodes.index)
    out["degree"] = cent["degree"]
    out["strength"] = cent["strength_s"] / WINDOW_S  # contacts; same ranking as activity potential
    out["betweenness"] = cent["betweenness"]
    out["eigenvector"] = cent["eigenvector_w"]
    ext = pd.Series({v: np.mean([group[u] != group[v] for u in G[v]]) if G.degree(v) else 0.0 for v in G})
    out["linkers"] = -(ext.reindex(nodes.index) - 0.5).abs()
    out["temporal_reach"] = temporal_reach(contacts, nodes.index)
    x = np.array(list(count.values()), dtype=float) / (gamma * period)  # expected contacts while infectious
    te = pd.Series(-np.expm1(np.log1p(-beta) * x), index=pd.MultiIndex.from_tuples(count.keys()))
    out["transmissibility"] = (te.groupby(level=0).sum().add(te.groupby(level=1).sum(), fill_value=0)
                               .reindex(nodes.index, fill_value=0.0))
    return out.fillna(0.0)


def ranking(scores: pd.Series, strength: pd.Series) -> np.ndarray:
    """Positions (into the node order) from best to worst; ties broken by strength."""
    return np.lexsort((-strength.to_numpy(), -scores.to_numpy()))


def immunisation_masks(
    strategy: str,
    k: int,
    scores: pd.DataFrame,
    contacts_matrix: np.ndarray,
    runs: int,
    rng_seed: int = 0,
) -> np.ndarray:
    """(runs, N) boolean mask of immunised nodes. `scores` must contain `strategy` for the
    ranked strategies and the oracle; `contacts_matrix` is `sir.contact_matrix`."""
    n = len(scores)
    mask = np.zeros((runs, n), dtype=bool)
    if k == 0:
        return mask
    if strategy not in RANDOMISED:
        mask[:, ranking(scores[strategy], scores["strength"])[:k]] = True
        return mask
    rng = np.random.default_rng(rng_seed)
    for r in range(runs):
        if strategy == "random":
            mask[r, rng.choice(n, k, replace=False)] = True
            continue
        chosen = mask[r]
        while chosen.sum() < k:
            u = rng.integers(n)
            free = np.flatnonzero((contacts_matrix[u] > 0) & ~chosen)
            if len(free) == 0:
                continue
            if strategy == "acquaintance":
                chosen[rng.choice(free)] = True
            else:  # weight: the most frequent partner not yet immunised
                chosen[free[np.argmax(contacts_matrix[u, free])]] = True
    return mask


def pruned_metrics(contacts_matrix: np.ndarray, removed: np.ndarray, seed: int = 0) -> dict:
    """Properties of the aggregated network after deleting the immunised nodes (Zhang, Zhao &
    Wang 2022, Fig. 2): largest eigenvalue of the weighted matrix W*, Louvain modularity of
    W*, relative size of its largest component, and the fraction of contacts removed."""
    keep = ~removed
    w = contacts_matrix[np.ix_(keep, keep)]
    G = nx.from_numpy_array(w)
    lcc = max((len(c) for c in nx.connected_components(G)), default=0)
    parts = nxc.louvain_communities(G, weight="weight", seed=seed)
    return {
        "lambda1": float(np.linalg.eigvalsh(w)[-1]) if keep.any() else 0.0,
        "modularity": float(nxc.modularity(G, parts, weight="weight")) if G.number_of_edges() else 0.0,
        "lcc": lcc / len(removed),
        "contacts_removed": 1 - w.sum() / contacts_matrix.sum(),
    }
