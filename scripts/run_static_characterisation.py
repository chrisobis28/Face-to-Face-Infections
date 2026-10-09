"""Stream 2: static characterisation of the aggregated graphs with null models (RQ1, static half).

For every dataset:
  results/static_metrics_<ds>.csv       lecture-1 metrics of the real aggregated graph
  results/degree_fit_<ds>.csv           power-law vs exponential fits of Pr[D = k]
  results/centralities_<ds>.csv         one row per node (used by streams 4 and 6)
  results/null_<kind>_<ds>.csv          metrics of each null realisation
  results/zscores_<kind>_<ds>.csv       real vs null, with z and empirical p
  figures/degree_distribution_<ds>.png
Across datasets:
  results/static_metrics_all.csv, results/zscores_all.csv
  figures/rq1_static_vs_null.png        real metric vs null-model band, one panel per metric

Usage:
  python scripts/run_static_characterisation.py --datasets hospital --n-null 20   # quick
  python scripts/run_static_characterisation.py --n-null 100                      # full
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f2f import cache, datasets, nulls, static_metrics as sm, views  # noqa: E402

FIG_DIR = datasets.ROOT / "figures"
PLOT_METRICS = [
    "L",
    "clustering",
    "mean_hopcount",
    "assortativity",
    "modularity_groups",
    "modularity_louvain",
    "spectral_radius",
    "algebraic_connectivity",
]


def degree_figure(key: str, G, fit: dict) -> None:
    deg = np.array([d for _, d in G.degree()])
    ks, counts = np.unique(deg[deg > 0], return_counts=True)
    p = counts / counts.sum()
    ccdf = 1 - np.cumsum(p) + p
    # Fitted lines on the CCDF (see fit_degree_distribution: pmf is too noisy at this N)
    a_pl, b_pl = np.polyfit(np.log(ks), np.log(ccdf), 1)
    a_ex, b_ex = np.polyfit(ks.astype(float), np.log(ccdf), 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].loglog(ks, ccdf, "s", ms=4, label="Pr[D >= k] (data)")
    axes[0].loglog(ks, np.exp(b_pl) * ks ** a_pl, "-", color="C3",
                   label=f"power law, tau={fit['ccdf_powerlaw_exponent']:.2f}, R2={fit['ccdf_powerlaw_r2']:.2f}")
    axes[0].set_xlabel("degree k")
    axes[0].set_ylabel("Pr[D >= k]")
    axes[0].set_title(f"{key}: log-log scale", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].semilogy(ks, ccdf, "s", ms=4, label="Pr[D >= k] (data)")
    axes[1].semilogy(ks, np.exp(b_ex + a_ex * ks), "-", color="C3",
                     label=f"exponential, rate={fit['ccdf_exponential_rate']:.3f}, R2={fit['ccdf_exponential_r2']:.2f}")
    axes[1].semilogy(ks, p, "o", ms=3, alpha=0.4, color="grey", label="Pr[D = k] (noisy at this N)")
    axes[1].set_xlabel("degree k")
    axes[1].set_title(f"log-linear scale  (best: {fit['best']})", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"degree_distribution_{key}.png", dpi=150)
    plt.close(fig)


def summary_figure(z_all: pd.DataFrame, kind: str) -> None:
    dsets = list(z_all["dataset"].unique())
    ncol = 4
    nrow = int(np.ceil(len(PLOT_METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(PLOT_METRICS):]:
        ax.set_visible(False)
    for ax, metric in zip(axes, PLOT_METRICS):
        sub = z_all[(z_all["metric"] == metric) & (z_all["kind"] == kind)].set_index("dataset").reindex(dsets)
        x = np.arange(len(dsets))
        ax.errorbar(
            x, sub["null_mean"], yerr=[sub["null_mean"] - sub["null_q025"], sub["null_q975"] - sub["null_mean"]],
            fmt="s", color="grey", capsize=4, label="null 95% band",
        )
        ax.plot(x, sub["real"], "o", color="C3", label="real")
        for xi, (_, r) in zip(x, sub.iterrows()):
            sig = "*" if abs(r["z"]) > 3 else ""
            # metrics whose null sits near zero (modularity, assortativity): show the absolute gap
            if abs(r["null_mean"]) < 0.05:
                label = f"Δ={r['real'] - r['null_mean']:+.2f}{sig}"
            else:
                label = f"{r['rel_diff']:+.0%}{sig}"
            ax.annotate(label, (xi, r["real"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(dsets, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"Real aggregated network vs degree-preserving null ({kind}). "
        "Label: difference to the null mean (relative, or absolute Δ when the null is near 0); * = beyond chance (|z| > 3)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"rq1_static_vs_null_{kind}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS))
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--kinds", nargs="+", default=["rewired", "configuration"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_metrics, all_z = [], []
    for key in args.datasets:
        t0 = time.time()
        contacts, nodes = datasets.load_contacts(key)
        G = views.aggregate(contacts, nodes)
        real = sm.network_metrics(G, seed=args.seed)
        fit = sm.fit_degree_distribution([d for _, d in G.degree()])
        cent = sm.centralities(G)
        cache.save_table(pd.Series(real, name=key).to_frame(), f"static_metrics_{key}")
        cache.save_table(pd.Series(fit, name=key).to_frame(), f"degree_fit_{key}")
        cache.save_table(cent, f"centralities_{key}")
        degree_figure(key, G, fit)
        all_metrics.append(pd.Series(real, name=key))
        print(f"[{key}] N={real['N']} L={real['L']} <k>={real['mean_degree']:.1f} C={real['clustering']:.3f} "
              f"E[H]={real['mean_hopcount']:.2f} rho={real['assortativity']:.3f} Q_groups={real['modularity_groups']:.3f} "
              f"Q_louvain={real['modularity_louvain']:.3f} lambda1={real['spectral_radius']:.1f} "
              f"a(G)={real['algebraic_connectivity']:.2f} Var/E[D]={real['degree_dispersion_var_over_mean']:.2f} "
              f"degree fit: {fit['best']} (CCDF R2={fit['best_r2']:.2f})  ({time.time()-t0:.1f}s)")
        for kind in args.kinds:
            t1 = time.time()
            null = nulls.null_ensemble(G, n=args.n_null, kind=kind, seed=args.seed)
            z = nulls.zscores(real, null)
            cache.save_table(null, f"null_{kind}_{key}")
            cache.save_table(z, f"zscores_{kind}_{key}")
            all_z.append(z.reset_index().assign(dataset=key, kind=kind))
            print(f"   null={kind} n={args.n_null}: "
                  + ", ".join(f"{m} {z.loc[m,'rel_diff']:+.0%} (z={z.loc[m,'z']:.0f})" for m in PLOT_METRICS)
                  + f"  ({time.time()-t1:.1f}s)")

    cache.save_table(pd.concat(all_metrics, axis=1), "static_metrics_all")
    z_all = pd.concat(all_z, ignore_index=True)
    cache.save_table(z_all, "zscores_all")
    for kind in args.kinds:
        summary_figure(z_all, kind)
    print("wrote results/ and figures/")


if __name__ == "__main__":
    main()
