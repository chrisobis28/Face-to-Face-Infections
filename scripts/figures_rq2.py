"""RQ2 figures: what predicts a node's role as seed and as blocker (immunisation target)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

UP = "results/data_s"
OUT = "figures/immunization"
DATASETS = ["hospital", "primary_school", "high_school"]
LABEL = {"hospital": "Hospital", "primary_school": "Primary school", "high_school": "High school"}
COLOR = {"hospital": "#2a78d6", "primary_school": "#eb6834", "high_school": "#1baf7a"}
PROPS = ["degree", "strength", "betweenness", "eigenvector", "closeness", "kshell", "transmissibility",
         "linkers", "temporal_reach", "meetings", "mean_meeting_s", "burstiness"]
PLAB = {"kshell": "k-shell", "linkers": "linker-ness", "temporal_reach": "temporal reach (T)",
        "meetings": "meetings (T)", "mean_meeting_s": "mean meeting\nlength (T)", "burstiness": "burstiness (T)",
        "transmissibility": "transmissi-\nbility"}
TOP_HEAT, TOP_OVERLAP = 0.25, 0.10
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

tabs = {d: pd.read_csv(f"{UP}/immunization_node_table_{d}.csv", index_col=0) for d in DATASETS}


def topk(t, col, frac):
    k = max(1, round(frac * len(t)))
    return t[col].nlargest(k).index, k


# ---------- 1. heatmaps on the top 25% ----------
def heatmap(role, fname):
    m, ns = {}, {}
    for d, t in tabs.items():
        idx, k = topk(t, role, TOP_HEAT)
        sub = t.loc[idx]
        m[LABEL[d]] = sub[PROPS].corrwith(sub[role], method="spearman")
        ns[LABEL[d]] = k
    m = pd.DataFrame(m).T
    fig, ax = plt.subplots(figsize=(11, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    img = ax.imshow(m.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(PROPS)), [PLAB.get(p, p) for p in PROPS], rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]), [f"{r}  (n={ns[r]})" for r in m.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(img, ax=ax, fraction=0.025, pad=0.01, label="Spearman ρ")
    # ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close(fig)
    return m


h_seed = heatmap("seed_influence", "fig1a_heatmap_seed_top25.png")
h_blk = heatmap("blocking_importance", "fig1b_heatmap_blocker_top25.png")


# ---------- 2. scatter plots, all nodes ----------
def scatter(prop, fname):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    rows = [("seed_influence", "seed influence"),
            ("blocking_importance", "blocking importance")]
    rho = {}
    for r, (role, ylab) in enumerate(rows):
        for c, d in enumerate(DATASETS):
            ax = axes[r, c]
            t = tabs[d]
            x, y = t[prop].to_numpy(float), t[role].to_numpy(float) * 100
            ax.scatter(x, y, s=18, color=COLOR[d], edgecolor="white", lw=0.4, alpha=0.9)
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, a * xs + b, color=INK, lw=1.6)
            rho[(role, d)] = pd.Series(x).corr(pd.Series(y), method="spearman")
            if role == "blocking_importance":
                ax.axhline(0, color=MUTED, lw=0.8, ls=":")
            ttl = f"Spearman {rho[(role, d)]:+.2f}"
            ax.set_title(f"{LABEL[d]}\n{ttl}" if r == 0 else ttl, fontsize=11 if r == 0 else 10)
            if r == 1:
                ax.set_xlabel("strength (contact windows)" if prop == "strength" else "degree (distinct contacts)")
            if c == 0:
                ax.set_ylabel(ylab, fontsize=10)
    # fig.suptitle(f"{'Strength' if prop == 'strength' else 'Degree'} vs. role in an outbreak "
    #              f"(one dot per person, line = linear trend)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close(fig)
    return rho


r_str = scatter("strength", "fig2a_scatter_strength.png")
r_deg = scatter("degree", "fig2b_scatter_degree.png")


# ---------- 3. overlap of top-10% seeds and top-10% blockers ----------
rows = []
for d, t in tabs.items():
    N = len(t)
    s_idx, k = topk(t, "seed_influence", TOP_OVERLAP)
    b_idx, _ = topk(t, "blocking_importance", TOP_OVERLAP)
    obs = len(set(s_idx) & set(b_idx))
    rows.append({"dataset": d, "N": N, "k": k, "overlap": obs, "expected_chance": k * k / N,
                 "p_value": hypergeom.sf(obs - 1, N, k, k)})
ov = pd.DataFrame(rows).set_index("dataset")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(DATASETS))
w = 0.36
obs_pct = (ov["overlap"] / ov["k"] * 100).to_numpy()
exp_pct = (ov["expected_chance"] / ov["k"] * 100).to_numpy()
ax.bar(x - w / 2, obs_pct, w, color=[COLOR[d] for d in DATASETS], label="observed")
ax.bar(x + w / 2, exp_pct, w, color="#d6d5d0", label="expected by chance")
for i, d in enumerate(DATASETS):
    ax.text(i - w / 2, obs_pct[i] + 1, f"{ov.loc[d, 'overlap']}/{ov.loc[d, 'k']}", ha="center", fontsize=10)
    ax.text(i + w / 2, exp_pct[i] + 1, f"{ov.loc[d, 'expected_chance']:.1f}/{ov.loc[d, 'k']}", ha="center",
            fontsize=10, color="#52514e")
ax.set_xticks(x, [f"{LABEL[d]}\n(N={ov.loc[d, 'N']})" for d in DATASETS])
ax.set_ylabel("% of top-10% seeds that are\nalso top-10% blockers")
ax.set_ylim(0, max(obs_pct.max(), exp_pct.max()) * 1.25 + 5)
# ax.set_title("Do good seeds also make good blockers?", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_overlap_seed_blocker.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 200)
print("HEATMAP seed (top25%)\n", h_seed.round(2).to_string())
print("\nHEATMAP blocker (top25%)\n", h_blk.round(2).to_string())
print("\nSCATTER spearman strength\n", pd.Series(r_str).round(2).to_string())
print("\nSCATTER spearman degree\n", pd.Series(r_deg).round(2).to_string())
print("\nOVERLAP\n", ov.round(3).to_string())
# context: all-node correlation between the two roles and strength-degree correlation
for d, t in tabs.items():
    print(d, "rho(seed,blocker) all nodes:", round(t["seed_influence"].corr(t["blocking_importance"], method="spearman"), 2),
          "| rho(degree,strength):", round(t["degree"].corr(t["strength"], method="spearman"), 2))