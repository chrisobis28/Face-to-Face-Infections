"""Figures and tables for stream 6 from the cached results of run_immunization.py.

    python scripts/plot_immunization.py

Writes figures/immunization/*.png and results/immunization_efficiency.csv, and prints
the markdown tables used in IMMUNIZATION_RESULTS.md.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from f2f import cache, datasets, immunization as im, sir  # noqa: E402

FIG = datasets.ROOT / "figures" / "immunization"
plt.style.use(datasets.ROOT / "figures" / "f2f.mplstyle")

DATASETS = ["hospital", "primary_school", "high_school"]
DS_LABEL = {"hospital": "Hospital", "primary_school": "Primary school", "high_school": "High school"}
DS_COLOR = {"hospital": "#2a78d6", "primary_school": "#eb6834", "high_school": "#1baf7a"}
SETTINGS = {"low": "β=0.03, γ=5e-4 (11 h)", "team": "β=0.1, γ=5e-4 (11 h)", "zhang": "β=0.01, γ=2.4e-5 (10 d)"}
VERSION_LABEL = {"real": "real", "time_shuffled": "time-shuffled", "aggregated": "aggregated"}
MAIN, F_MAIN = "team", 0.10
INK, MUTED = "#0b0b0b", "#8a8984"

# Colour follows the strategy in every figure (validated palette, fixed order); local and
# reference strategies are neutral so at most seven hues are on screen.
STYLE = {
    "degree": dict(color="#2a78d6", ls="-"),
    "strength": dict(color="#eb6834", ls="-"),
    "betweenness": dict(color="#1baf7a", ls="-"),
    "linkers": dict(color="#eda100", ls="-"),
    "temporal_reach": dict(color="#e87ba4", ls="-"),
    "eigenvector": dict(color="#008300", ls="-"),
    "transmissibility": dict(color="#6250d6", ls="-", lw=2.6),
    "acquaintance": dict(color="#b5b4ae", ls="-."),
    "weight": dict(color="#b5b4ae", ls=":"),
    "random": dict(color="#52514e", ls="--"),
    "oracle": dict(color=INK, ls=":", lw=1.6),
}
LABEL = {"transmissibility": "transmissibility (proposed)", "temporal_reach": "temporal reach",
         "oracle": "oracle (simulated)", "weight": "weight (Lee)", "acquaintance": "acquaintance"}
ORDER = list(STYLE)


def lbl(s: str) -> str:
    return LABEL.get(s, s)


def efficiency(df: pd.DataFrame, outcome: str = "final_size", n_boot: int = 1000) -> pd.DataFrame:
    """Per cell: mean outcome, reduction 1 - mean(V)/mean(none) with a paired bootstrap 95% CI
    (runs share random numbers, so run r of a strategy is paired with run r of `none`)."""
    keys = ["setting", "dataset", "version"]
    x = df[df["outcome"] == outcome]
    base = {k: g.sort_values("run")["value"].to_numpy() for k, g in x[x["strategy"] == "none"].groupby(keys)}
    rng = np.random.default_rng(0)
    rows = []
    for (s, d, v, st, f), g in x[x["strategy"] != "none"].groupby(keys + ["strategy", "budget"]):
        b, val = base[(s, d, v)], g.sort_values("run")["value"].to_numpy()
        idx = rng.integers(0, len(b), (n_boot, len(b)))
        boot = 1 - val[idx].mean(1) / b[idx].mean(1)
        rows.append({"setting": s, "dataset": d, "version": v, "strategy": st, "budget": f,
                     "mean": val.mean(), "baseline": b.mean(), "reduction": 1 - val.mean() / b.mean(),
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)})
    out = pd.DataFrame(rows)
    best = out[out["strategy"] != "oracle"].groupby(keys + ["budget"])["reduction"].max().rename("best")
    out = out.merge(best, left_on=keys + ["budget"], right_index=True)
    out["regret"] = (out["best"] - out["reduction"]) * 100  # pp short of the best strategy in the cell
    return out


def regret_table(eff: pd.DataFrame) -> pd.DataFrame:
    """Minimax regret (Savage): per strategy, the largest and the mean shortfall (pp) from the
    best strategy over all cells (setting x dataset x version), per budget. The general strategy
    has the smallest largest shortfall. Unlike ratios, saturated cells (no strategy beats
    random) add only small regrets instead of noise."""
    e = eff[eff["strategy"] != "oracle"]
    t = e.groupby(["strategy", "budget"])["regret"].agg(["max", "mean"]).unstack("budget")
    t.columns = [f"{s} f={b:.0%}" for s, b in t.columns]
    return t.sort_values("max f=10%")


def fig_budget_curves(eff: pd.DataFrame) -> None:
    """Starnini et al. (2013) Figs 6-7: final-size ratio i_V = FS(V)/FS(none) vs fraction f."""
    real = eff[eff["version"] == "real"]
    settings = [s for s in SETTINGS if s in set(real["setting"])]
    fig, axes = plt.subplots(len(settings), 3, figsize=(10, 2.7 * len(settings) + 0.6), sharex=True, squeeze=False)
    for r, s in enumerate(settings):
        for c, d in enumerate(DATASETS):
            ax = axes[r, c]
            cell = real[(real["setting"] == s) & (real["dataset"] == d)]
            for st in ORDER:
                g = cell[cell["strategy"] == st].sort_values("budget")
                if g.empty:
                    continue
                kw = {"lw": 1.5, **STYLE[st]}
                ax.plot(np.r_[0, g["budget"]] * 100, np.r_[1, 1 - g["reduction"]], **kw)
            base = cell["baseline"].iloc[0] if len(cell) else np.nan
            ax.set_ylim(0, 1.1)
            ax.axhline(1, color=MUTED, lw=0.8)
            ax.text(0.98, 0.995, f"no immunisation: {base:.0%} infected", transform=ax.transAxes,
                    ha="right", va="top", color="#52514e", fontsize=7.5)
            if r == 0:
                ax.set_title(DS_LABEL[d])
            if c == 0:
                ax.set_ylabel(f"{SETTINGS[s]}\nfinal size / no immunisation")
            if r == len(settings) - 1:
                ax.set_xlabel("immunised nodes f (%)")
    handles = [Line2D([0], [0], **{"lw": 1.5, **STYLE[s]}) for s in ORDER]
    fig.legend(handles, [lbl(s) for s in ORDER], loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Outbreak size after immunising a fraction f of nodes at day 0 (real temporal network, 100 runs)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(FIG / "budget_curves.png")
    plt.close(fig)


def fig_strategy_table(eff: pd.DataFrame) -> None:
    """Zhang et al. (2022) Tables 2-5 as a heat map: reduction (%) at f = 10 % per setting x version x
    dataset, best per column in bold; colour = shortfall from the best strategy (regret, pp).
    Last two columns: largest and mean regret, which select the general strategy."""
    e = eff[eff["budget"] == F_MAIN].copy()
    e["col"] = e["setting"] + "|" + e["version"] + "|" + e["dataset"]
    cols = [f"{s}|{v}|{d}" for v in VERSION_LABEL for s in SETTINGS for d in DATASETS]
    cols = [c for c in cols if c in set(e["col"])]
    reg = e.pivot_table(index="strategy", columns="col", values="regret")[cols]
    red = e.pivot_table(index="strategy", columns="col", values="reduction")[cols] * 100
    summary = reg.drop(index="oracle", errors="ignore").agg(["max", "mean"], axis=1)
    order = list(summary["max"].sort_values().index) + (["oracle"] if "oracle" in reg.index else [])
    reg, red = reg.loc[order], red.loc[order]
    reg[["largest regret", "mean regret"]] = summary.reindex(order).to_numpy()
    data = reg.to_numpy(dtype=float)
    ncol = len(cols)

    fig, ax = plt.subplots(figsize=(0.42 * data.shape[1] + 2.8, 0.36 * data.shape[0] + 2.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    img = ax.imshow(np.clip(data, 0, 30), cmap=cmap, vmin=0, vmax=30, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = red.iat[i, j] if j < ncol else data[i, j]
            if np.isnan(v):
                continue
            bold = j < ncol and data[i, j] == 0
            ax.text(j, i, f"{v:.0f}" if j < ncol else f"{v:.1f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if bold else "normal", color="white" if data[i, j] > 18 else INK)
    ax.set_yticks(range(len(order)), [lbl(s) for s in order])
    ax.set_xticks(range(data.shape[1]),
                  [DS_LABEL[c.split("|")[2]].split()[0][:4] + "." for c in cols] + ["largest", "mean"],
                  rotation=90)
    for j in range(0, ncol, 3):
        s_, v_, _ = cols[j].split("|")
        ax.text(j + 1, -0.8, f"{v_.replace('_', '-')}\n{s_}", ha="center", va="bottom", fontsize=6.5)
        if j:
            ax.axvline(j - 0.5, color="white", lw=1.5 if j % 9 else 3)
    ax.axvline(ncol - 0.5, color="white", lw=3)
    ax.grid(False)
    ax.set_title("Final-size reduction (%) at f = 10 % (bold = best in column); colour = shortfall from the best "
                 "strategy (pp)\nlast two columns: largest and mean shortfall over all 27 columns (minimax regret)\n\n",
                 fontsize=9)
    fig.colorbar(img, ax=ax, fraction=0.02, pad=0.01, label="shortfall from best (pp, capped at 30)")
    fig.tight_layout()
    fig.savefig(FIG / "strategy_table.png")
    plt.close(fig)


def fig_curves() -> None:
    """Rizzo et al. (2016) Fig. 8: mean prevalence over time, no intervention vs strategies at f = 10 %."""
    cur = pd.read_parquet(cache.result_path("immunization_curves"))
    show = ["none", "random", "degree", "strength", "betweenness", "linkers", "transmissibility"]
    settings = [s for s in ["team", "zhang"] if s in set(cur["setting"])]
    fig, axes = plt.subplots(len(settings), 3, figsize=(10, 2.6 * len(settings) + 0.6), squeeze=False)
    for r, s in enumerate(settings):
        for c, d in enumerate(DATASETS):
            ax = axes[r, c]
            for st in show:
                g = cur[(cur["setting"] == s) & (cur["dataset"] == d) & (cur["strategy"] == st)]
                kw = dict(color=INK, ls="-", lw=1.8) if st == "none" else {"lw": 1.5, **STYLE[st]}
                ax.plot(g["t_h"] / 24, g["prevalence"] * 100, **kw)
            ax.set_xlim(0, None)
            if r == 0:
                ax.set_title(DS_LABEL[d])
            if c == 0:
                ax.set_ylabel(f"{SETTINGS[s]}\ninfectious (% of nodes)")
            if r == len(settings) - 1:
                ax.set_xlabel("time (days, recording looped)")
    handles = [Line2D([0], [0], color=INK, lw=1.8)] + [Line2D([0], [0], **{"lw": 1.5, **STYLE[s]}) for s in show[1:]]
    fig.legend(handles, ["no immunisation"] + [lbl(s) for s in show[1:]], loc="lower center", ncol=7, frameon=False)
    fig.suptitle("Mean prevalence with 10 % of nodes immunised (real temporal network, 100 runs)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG / "prevalence_curves.png")
    plt.close(fig)


def fig_pruned(eff: pd.DataFrame) -> pd.DataFrame:
    """Zhang et al. (2022) Fig. 2: outcome vs properties of the aggregated network left after immunisation."""
    pr = pd.read_csv(cache.result_path("immunization_pruned").with_suffix(".csv"), index_col=0)
    pr = pr[(pr["setting"] == MAIN) & (pr["budget"] == F_MAIN)]
    full = {}
    for d in DATASETS:
        c, n = datasets.load_contacts(d)
        m = sir.contact_matrix(c, n.index)
        full[d] = im.pruned_metrics(m, np.zeros(len(n), bool))
    pr = pr.assign(lambda1_rel=[r.lambda1 / full[r.dataset]["lambda1"] for r in pr.itertuples()],
                   modularity_gain=[r.modularity - full[r.dataset]["modularity"] for r in pr.itertuples()])
    e = eff[(eff["setting"] == MAIN) & (eff["version"] == "real") & (eff["budget"] == F_MAIN)]
    pr = pr.merge(e[["dataset", "strategy", "reduction"]], on=["dataset", "strategy"])
    props = [("lambda1_rel", "λ1(W*) / λ1(W)"), ("modularity_gain", "modularity(W*) − modularity(W)"),
             ("contacts_removed", "contacts removed (fraction)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    rho = {}
    for ax, (p, name) in zip(axes, props):
        for d in DATASETS:
            g = pr[(pr["dataset"] == d) & (pr["strategy"] != "oracle")]
            ax.scatter(g[p], g["reduction"] * 100, s=34, color=DS_COLOR[d], edgecolor="white", lw=0.8, zorder=3)
            t = g[g["strategy"] == "transmissibility"]
            ax.scatter(t[p], t["reduction"] * 100, s=90, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
            rho[(d, p)] = g[[p, "reduction"]].corr("spearman").iat[0, 1]
        ax.set_xlabel(name + "\nρ: " + ", ".join(f"{DS_LABEL[d].split()[0][:4]}. {rho[(d, p)]:+.2f}" for d in DATASETS))
    axes[0].set_ylabel("final-size reduction at f = 10 % (%)")
    handles = [Line2D([0], [0], marker="o", ls="", color=DS_COLOR[d], markeredgecolor="white") for d in DATASETS]
    handles.append(Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=9))
    fig.legend(handles, [DS_LABEL[d] for d in DATASETS] + ["transmissibility (proposed)"],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Why strategies work: the aggregated network left after immunising 10 % (one dot per strategy;"
                 " ρ = Spearman)\nThe largest component always keeps every remaining node that has contacts, so it is not shown.", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(FIG / "pruned_network.png")
    plt.close(fig)
    return pd.DataFrame(rho, index=["spearman"]).T


PROPS = ["degree", "strength", "betweenness", "eigenvector", "closeness", "kshell", "transmissibility",
         "linkers", "temporal_reach", "meetings", "mean_meeting_s", "burstiness"]
PROP_LABEL = {"kshell": "k-shell", "linkers": "linker-ness", "temporal_reach": "temporal reach (T)",
              "meetings": "meetings (T)", "mean_meeting_s": "mean meeting length (T)",
              "burstiness": "burstiness (T)", "transmissibility": "transmissibility"}


def split_half(d: str) -> dict:
    """Reliability of the per-node roles: Spearman between the estimates from the first and the
    second half of the runs (a value near 1 means the node ranking is not simulation noise)."""
    nd = pd.read_parquet(cache.result_path("immunization_nodes"))
    nd = nd[nd["dataset"] == d]
    seed = nd[(nd["role"] == "seed") & (nd["outcome"] == "final_size")]
    half = seed["run"] < seed["run"].max() / 2
    s = seed.groupby([half, "node"])["value"].mean().unstack(0)
    blk = nd[nd["outcome"] == "others_infected"]
    bhalf = blk["run"] < blk["run"].max() / 2
    m = blk.groupby([bhalf, "role", "node"])["value"].mean()
    b = {h: m[h]["baseline"] - m[h]["blocked"] for h in (True, False)}
    return {"split_half_seed": s[True].corr(s[False], method="spearman"),
            "split_half_blocker": b[True].corr(b[False], method="spearman")}


def fig_seed_vs_blocker() -> pd.DataFrame:
    """Holme (2017) / Radicchi & Castellano (2017): a node's value as seed vs as immunisation target."""
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45)
    corr, stats = {}, {}
    role_marker = {"NUR": "o", "MED": "s", "ADM": "^", "PAT": "D"}
    role_color = {"NUR": "#2a78d6", "MED": "#eb6834", "ADM": "#1baf7a", "PAT": "#8a8984"}
    for c, d in enumerate(DATASETS):
        t = pd.read_csv(cache.result_path(f"immunization_node_table_{d}").with_suffix(".csv"), index_col=0)
        ax = fig.add_subplot(gs[0, c])
        if d == "hospital":
            for role, g in t.groupby("group"):
                ax.scatter(g["seed_influence"] * 100, g["blocking_importance"] * 100, s=22, edgecolor="white", lw=0.5, label=role)
            ax.legend(loc="upper left", fontsize=7, frameon=False, handletextpad=0.1)
        else:
            ax.scatter(t["seed_influence"] * 100, t["blocking_importance"] * 100, s=14, color=DS_COLOR[d],
                       edgecolor="white", lw=0.4)
        k = max(1, round(0.1 * len(t)))
        top_s = set(t["seed_influence"].nlargest(k).index)
        top_b = set(t["blocking_importance"].nlargest(k).index)
        r = t[["seed_influence", "blocking_importance"]].corr("spearman").iat[0, 1]
        stats[d] = {"spearman": r, "top10_overlap": len(top_s & top_b) / k, **split_half(d)}
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.text(0.98, 0.04, f"Spearman {r:+.2f}\ntop-10 % overlap {len(top_s & top_b)}/{k}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="#52514e")
        ax.set_title(DS_LABEL[d])
        ax.set_xlabel("final size %")
        if c == 0:
            ax.set_ylabel("infections prevented %")
        for role, col in [("seed_influence", "seed"), ("blocking_importance", "target")]:
            corr[(DS_LABEL[d], col)] = t[PROPS].corrwith(t[role], method="spearman")
    m = pd.DataFrame(corr)
    ax = fig.add_subplot(gs[1, :])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    img = ax.imshow(m.T.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(m.shape[1]):
        for j in range(m.shape[0]):
            v = m.iat[j, i]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(PROPS)), [PROP_LABEL.get(p, p) for p in PROPS], rotation=30, ha="right")
    ax.set_yticks(range(m.shape[1]), [f"{a}: as {b}" for a, b in m.columns])
    ax.grid(False)
    fig.colorbar(img, ax=ax, fraction=0.02, pad=0.01, label="Spearman ρ")
    # ax.set_title("Which node property predicts each role? (T = temporal property; main setting; 100 runs per node as seed, 400 as immunised)")
    fig.suptitle("Good seeds vs good immunisation targets", y=0.99)
    fig.savefig(FIG / "seed_vs_blocker.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats).T


def fig_seeds_and_sets(eff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 vs 5 random seeds, and each strategy's immunisation set reused as the seed set."""
    e5 = efficiency(pd.read_parquet(cache.result_path("immunization_seeds5")).assign(setting=MAIN), n_boot=200)
    e1 = eff[(eff["setting"] == MAIN) & (eff["version"] == "real")]
    ss = pd.read_parquet(cache.result_path("immunization_seedsets"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    sets = {}
    strategies = [s for s in ORDER if s in set(e1["strategy"])]
    for c, d in enumerate(DATASETS):
        ax = axes[0, c]
        for i, st in enumerate(strategies):
            a = e1[(e1["dataset"] == d) & (e1["strategy"] == st) & (e1["budget"] == F_MAIN)]["reduction"].iat[0] * 100
            b = e5[(e5["dataset"] == d) & (e5["strategy"] == st) & (e5["budget"] == F_MAIN)]["reduction"].iat[0] * 100
            ax.plot([a, b], [i, i], color=MUTED, lw=1, zorder=1)
            ax.scatter(a, i, s=36, color=STYLE[st]["color"], zorder=3, edgecolor="white")
            ax.scatter(b, i, s=36, facecolor="white", edgecolor=STYLE[st]["color"], lw=1.4, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(strategies)), [lbl(s) for s in strategies] if c == 0 else [])
        ax.invert_yaxis()
        ax.set_title(DS_LABEL[d])
        ax.set_xlabel("final-size reduction at f = 10 % (%)")

        ax = axes[1, c]
        x = ss[(ss["dataset"] == d)]
        fs = x[x["outcome"] == "final_size"].groupby("strategy")["value"].mean() * 100
        t50 = x[x["outcome"] == "t50_h"].groupby("strategy")["value"]
        red = e1[(e1["dataset"] == d) & (e1["budget"] == F_MAIN)].set_index("strategy")["reduction"] * 100
        r_fs = red.reindex(fs.index).corr(fs, method="spearman")
        sets[d] = {"spearman(block, seed final size)": r_fs,
                   "spearman(block, -t50)": red.reindex(fs.index).corr(-t50.median(), method="spearman")}
        for st in fs.index:
            sets[d][f"{st}: FS% / t50 h / reached50"] = (f"{fs[st]:.0f} / {t50.get_group(st).median():.0f} / "
                                                         f"{t50.get_group(st).notna().mean():.2f}")
            ax.scatter(red[st], fs[st], s=40, color=STYLE[st]["color"], edgecolor="white", zorder=3)
        for st in ["random", "strength", "betweenness", "linkers", "transmissibility", "oracle", "temporal_reach"]:
            if st in fs.index:
                ax.annotate(lbl(st).split(" (")[0], (red[st], fs[st]), xytext=(4, 3), textcoords="offset points",
                            fontsize=7, color="#52514e")
        ax.text(0.02, 0.98, f"Spearman {r_fs:+.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#52514e")
        ax.set_xlabel("reduction, top 10 % immunised (%)")
        if c == 0:
            ax.set_ylabel("final size (%) when the ranking's\ntop 5 nodes are the 5 seeds")
    handles = [Line2D([0], [0], marker="o", ls="", color=INK), Line2D([0], [0], marker="o", ls="",
               markerfacecolor="white", markeredgecolor=INK)]
    fig.legend(handles, ["1 random seed", "5 random seeds"], loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("")
    fig.suptitle("Top: immunisation with 1 vs 5 random seeds.  Bottom: do rankings that find good immunisation"
                 " targets also find good seeds?")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "seeds_and_sets.png")
    plt.close(fig)
    return e5, pd.DataFrame(sets).T


def md(df: pd.DataFrame, fmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                                    for v in r]) + " |")
    return "\n".join(lines)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    eff = efficiency(pd.read_parquet(cache.result_path("immunization_grid")))
    cache.save_table(eff, "immunization_efficiency")
    # fig_budget_curves(eff)
    # fig_strategy_table(eff)
    # fig_curves()
    # rho = fig_pruned(eff)
    stats = fig_seed_vs_blocker()
    # e5, sets = fig_seeds_and_sets(eff)

    # print("\n## minimax regret: shortfall (pp) from the best strategy over all 27 cells\n")
    # print(md(regret_table(eff).rename_axis("strategy"), fmt="{:.1f}"))
    # main_tab = eff[(eff["setting"] == MAIN) & (eff["version"] == "real") & (eff["budget"] == F_MAIN)]
    # main_tab = main_tab.assign(cell=lambda x: [f"{r*100:.0f} [{lo*100:.0f}, {hi*100:.0f}]" for r, lo, hi in
    #                                           zip(x["reduction"], x["ci_lo"], x["ci_hi"])])
    # print("\n## main setting, real network, f = 10 %: reduction % [95% CI]\n")
    # print(md(main_tab.pivot(index="strategy", columns="dataset", values="cell")[DATASETS].loc[
    #     [s for s in ORDER if s in set(main_tab["strategy"])]], fmt="{}"))
    # for name, outcome in [("peak", "peak"), ("peak time (h)", "peak_time_h")]:
    #     g = pd.read_parquet(cache.result_path("immunization_grid"))
    #     g = g[(g["setting"] == MAIN) & (g["version"] == "real") & (g["outcome"] == outcome)
    #           & (g["budget"].isin([0.0, F_MAIN]))].groupby(["strategy", "dataset"])["value"].mean().unstack()[DATASETS]
    #     print(f"\n## main setting, real network, mean {name} (none vs f = 10 %)\n")
    #     print(md(g.loc[["none"] + [s for s in ORDER if s in g.index]], fmt="{:.3f}" if outcome == "peak" else "{:.1f}"))
    # print("\n## pruned network: Spearman(property, reduction) across strategies\n")
    # print(md(rho.unstack(0)))
    # print("\n## seed vs blocker per node\n")
    # print(md(stats))
    # print("\n## 5 seeds: reduction at f = 10 %\n")
    # t5 = e5[e5["budget"] == F_MAIN].pivot(index="strategy", columns="dataset", values="reduction")[DATASETS]
    # print(md(t5.loc[[s for s in ORDER if s in t5.index]] * 100, fmt="{:.0f}"))
    # print("\n## immunisation sets as seed sets\n")
    # print(md(sets.T, fmt="{:+.2f}"))


if __name__ == "__main__":
    main()
