"""Stream 2: static characterisation of the aggregated contact graph (lecture 1).

Network metrics (one number per graph):   `network_metrics(G)`
Degree-distribution fits (lecture 1):     `fit_degree_distribution(degrees)`
Centralities (one row per node):          `centralities(G)`

Conventions
- Topological metrics use the unweighted graph (a link exists if the pair ever met).
- `strength` is the weighted degree: total contact seconds of a node.
- Path-based metrics (hopcount, closeness, betweenness) are computed on the largest
  connected component, unweighted, so they match the lecture definitions. Weighted
  betweenness/closeness are also given, using distance = 1 / contact seconds.
- Spectral radius lambda1 is the largest adjacency eigenvalue of the unweighted graph:
  the SIS epidemic threshold is tau_c = 1 / lambda1.
- Algebraic connectivity is the second-smallest Laplacian eigenvalue.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nxc


def _lcc(G: nx.Graph) -> nx.Graph:
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def spectral_radius(G: nx.Graph) -> float:
    A = nx.to_numpy_array(G, weight=None)
    return float(np.linalg.eigvalsh(A)[-1])


def algebraic_connectivity(G: nx.Graph) -> float:
    L = nx.laplacian_matrix(G, weight=None).toarray().astype(float)
    mu = np.linalg.eigvalsh(L)
    return float(mu[1]) if len(mu) > 1 else 0.0


def modularity_of_groups(G: nx.Graph, attr: str = "group") -> float:
    groups = nx.get_node_attributes(G, attr)
    if not groups:
        return float("nan")
    parts: dict[str, set] = {}
    for n, g in groups.items():
        parts.setdefault(g, set()).add(n)
    return float(nxc.modularity(G, list(parts.values()), weight=None))


def louvain_modularity(G: nx.Graph, seed: int = 0) -> tuple[float, int]:
    parts = nxc.louvain_communities(G, weight=None, seed=seed)
    return float(nxc.modularity(G, parts, weight=None)), len(parts)


def network_metrics(G: nx.Graph, group_attr: str = "group", seed: int = 0) -> dict:
    """Lecture-1 metrics of one graph. Keys are stable; scripts rely on them."""
    N, L = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], dtype=float)
    strength = np.array([s for _, s in G.degree(weight="weight")], dtype=float)
    H = _lcc(G)
    q_louv, n_comm = louvain_modularity(G, seed=seed)
    out = {
        "N": N,
        "L": L,
        "density": 2 * L / (N * (N - 1)) if N > 1 else float("nan"),
        "mean_degree": deg.mean(),
        "var_degree": deg.var(),
        "degree_dispersion_var_over_mean": deg.var() / deg.mean() if deg.mean() > 0 else float("nan"),
        "max_degree": deg.max(),
        "mean_strength_s": strength.mean(),
        "clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "giant_component_frac": H.number_of_nodes() / N,
        "mean_hopcount": nx.average_shortest_path_length(H) if H.number_of_nodes() > 1 else float("nan"),
        "diameter": nx.diameter(H) if H.number_of_nodes() > 1 else 0,
        "assortativity": nx.degree_assortativity_coefficient(G) if L > 0 else float("nan"),
        "modularity_groups": modularity_of_groups(G, group_attr),
        "modularity_louvain": q_louv,
        "n_louvain_communities": n_comm,
        "spectral_radius": spectral_radius(G),
        "algebraic_connectivity": algebraic_connectivity(H),
    }
    out["sis_threshold_1_over_lambda1"] = 1.0 / out["spectral_radius"] if out["spectral_radius"] > 0 else float("nan")
    return out


# Metrics that make sense on an unweighted degree-preserving null graph.
# L and mean_degree are included on purpose: the configuration model drops self-loops and
# duplicate stubs, so on dense graphs it loses links. A z-score on L shows that loss, and
# is the reason the rewired null (exact degrees) is the default for z-scores.
NULL_METRICS = [
    "L",
    "mean_degree",
    "clustering",
    "transitivity",
    "mean_hopcount",
    "diameter",
    "assortativity",
    "modularity_groups",
    "modularity_louvain",
    "spectral_radius",
    "algebraic_connectivity",
]


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line y = a x + b; returns (a, b, R^2)."""
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def fit_degree_distribution(degrees, min_count: int = 1) -> dict:
    """Lecture-1 curve fitting of Pr[D = k].

    Power law:   log Pr = -tau * log k + c   (linear in log-log)
    Exponential: log Pr = -lam * k + c       (linear in log-linear)
    Fits are done both on the empirical pmf (k with at least `min_count` nodes) and on
    the complementary CDF Pr[D >= k]. With a few hundred nodes the pmf is far too noisy
    (most degrees occur once), so `best` is decided on the CCDF fits. A pmf power law
    with exponent tau has a CCDF slope of 1 - tau, which is why the CCDF exponent is
    reported as `1 - slope`.
    """
    deg = np.asarray(degrees, dtype=int)
    deg = deg[deg > 0]
    ks_all, counts_all = np.unique(deg, return_counts=True)
    p_all = counts_all / counts_all.sum()
    out: dict = {
        "n_distinct_degrees": int(len(ks_all)),
        "mean_degree": float(deg.mean()),
        # ER random graphs have Var[D] = E[D] (Poisson): dispersion 1. Heavy tails give >> 1,
        # a narrow bell (like these contact networks) gives a value near or below 1.
        "degree_dispersion_var_over_mean": float(deg.var() / deg.mean()) if deg.mean() > 0 else float("nan"),
    }
    if len(ks_all) < 3:
        return {**out, "best": "undetermined"}
    # pmf fit on degrees seen at least min_count times (otherwise log p is dominated by singletons)
    keep = counts_all >= min_count
    ks, p = ks_all[keep], p_all[keep]
    if len(ks) >= 3:
        tau, _, r2_pl = _linfit(np.log(ks), np.log(p))
        lam, _, r2_exp = _linfit(ks.astype(float), np.log(p))
        out.update(
            {
                "pmf_powerlaw_exponent": -tau,
                "pmf_powerlaw_r2": r2_pl,
                "pmf_exponential_rate": -lam,
                "pmf_exponential_r2": r2_exp,
            }
        )
    # CCDF Pr[D >= k] on ALL degrees (no truncation: the tail is exactly what we want to see)
    ks = ks_all
    ccdf = 1 - np.cumsum(p_all) + p_all
    mask = ccdf > 0
    tau_c, _, r2_pl_c = _linfit(np.log(ks[mask]), np.log(ccdf[mask]))
    lam_c, _, r2_exp_c = _linfit(ks[mask].astype(float), np.log(ccdf[mask]))
    out.update(
        {
            "ccdf_powerlaw_exponent": 1 - tau_c,  # pmf exponent = ccdf slope + 1
            "ccdf_powerlaw_r2": r2_pl_c,
            "ccdf_exponential_rate": -lam_c,
            "ccdf_exponential_r2": r2_exp_c,
        }
    )
    out["best"] = "power_law" if r2_pl_c > r2_exp_c else "exponential"
    out["best_r2"] = max(r2_pl_c, r2_exp_c)
    return out


def centralities(G: nx.Graph) -> pd.DataFrame:
    """One row per node: degree, strength, betweenness, closeness, eigenvector, k-shell.

    Unweighted versions follow the lecture definitions. Weighted betweenness and
    closeness treat 1 / contact-seconds as the length of a link, so pairs that met
    longer are closer.
    """
    nodes = list(G.nodes())
    H = _lcc(G)
    dist = {(u, v): 1.0 / d["weight"] for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, dist, "dist")

    df = pd.DataFrame(index=pd.Index(nodes, name="node"))
    df["group"] = pd.Series(nx.get_node_attributes(G, "group"))
    df["degree"] = pd.Series(dict(G.degree()))
    df["strength_s"] = pd.Series(dict(G.degree(weight="weight")))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df["betweenness_w"] = pd.Series(nx.betweenness_centrality(G, normalized=True, weight="dist"))
    df["closeness"] = pd.Series(nx.closeness_centrality(G))
    df["closeness_w"] = pd.Series(nx.closeness_centrality(G, distance="dist"))
    try:
        df["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(H, weight=None))
        df["eigenvector_w"] = pd.Series(nx.eigenvector_centrality_numpy(H, weight="weight"))
    except Exception:
        df["eigenvector"] = np.nan
        df["eigenvector_w"] = np.nan
    G0 = G.copy()
    G0.remove_edges_from(nx.selfloop_edges(G0))
    df["kshell"] = pd.Series(nx.core_number(G0))
    for col in ["eigenvector", "eigenvector_w"]:
        df[col] = df[col].fillna(0.0)
    return df


def centrality_correlations(cent: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    cols = [c for c in cent.columns if c != "group"]
    return cent[cols].corr(method=method)
