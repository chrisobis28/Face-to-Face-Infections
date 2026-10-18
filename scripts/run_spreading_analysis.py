"""
Outputs and plots for the spreading analysis. 
SIS/SIR (discrete for now) on aggregate and temporal networks, for each network type. 
if beta · λ₁ / gamma > 1, then an outbreak occurs 
"""
import argparse
import sys
import time
from pathlib import Path
from spreading import * 
from itertools import product
from f2f import datasets, views  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



FIG_DIR = datasets.ROOT / "figures" / "spreading"
PROCESSED_DIR = datasets.ROOT / "data" / "spreading"
MODELS = ["SIR"]
HS = "high_school"
PS = "primary_school"
ZKH = "hospital"
DATASET_KEYS = [HS, PS, ZKH]
VARIANTS = ["aggr", "temporal"]
PARAMS = {
    "aggr":     dict(beta=0.05, gamma=0.02, n_seeds=5, timesteps=500),
    "temporal": dict(beta=0.1,  gamma=0.0005, n_seeds=5),
}
# N_RUNS = 20 #ToDO  
COLORS = {"S": "tab:blue", "I": "tab:red", "R": "tab:green"}
DATASET_LABELS = {HS: "High school", PS: "Primary school", ZKH: "Hospital"}
VARIANT_LABELS = {"aggr": "Aggregated network", "temporal": "Temporal network"}
SECONDS_PER_DAY = 86400
METHODS = ["discrete", "ode"]
WINDOW_S = 20 


def result_path(model: str, key: str, variant: str, method: str = "discrete"):
    return PROCESSED_DIR / f"{model}_{key}_{method}_{variant}.csv"

def compute_k_mean(variant: str, data: dict) -> tuple[float, int]:
    """returns (k_mean, N) to pass to the ode function"""
    if variant == "aggr":
        G = data["graph"]
        N = G.number_of_nodes()
        # same normalization in discrete(): A / A.max()
        # we always use the weighted graph, so return a weighted version of the average 
        A = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
        A = A / A.max()
        return float(A.sum(axis=1).mean()), N
    elif variant == "temporal":
        contacts, N = data["contacts"], len(data["nodes"])
        w = (contacts["t"] - contacts["t"].min()) // WINDOW_S
        n_windows = int(w.max()) + 1
        # every row is a chance to infect in both directions, so * 2. 
        return 2 * len(contacts) / (N * n_windows), N
    raise ValueError(f"Unknown variant: {variant}")

def run_one_discrete(variant: str, model: str, data: dict, **params) -> pd.DataFrame:
    if variant == "aggr":
        res = discrete(graph = data["graph"], model=model, **params)
    elif variant == "temporal":
        res = simulate_discrete_temporal(contacts=data["contacts"], nodes=data["nodes"], model=model, **params)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return res

def run_one_ode(variant: str, model: str, data: dict, **params) -> pd.DataFrame:
    k_mean, N = compute_k_mean(variant, data)
    if variant == "aggr":
        t_max = params["timesteps"]
    else:  # temporal: t_max = number of windows 
        c = data["contacts"]
        t_max = int((c["t"].max() - c["t"].min()) // WINDOW_S) + 1
    res = simulate_ode(model=model, beta=params["beta"], gamma=params["gamma"],
                       k_mean=k_mean, i0=params["n_seeds"] / N,
                       t_max=t_max, n_points=t_max)
    if variant == "temporal":   # windows -> seconds, so _prepare() can convert to days 
        res.index = pd.Index(res.index * WINDOW_S, name="t_s")
    print(f"   k_mean={k_mean:.4g}, R0={params['beta'] * k_mean / params['gamma']:.3g}")
    return res

def load_data(key: str) -> dict:
    contacts, nodes = datasets.load_contacts(key)
    return {"contacts": contacts, "nodes": nodes, "graph": views.aggregate(contacts, nodes)}

def run_all(keys=DATASET_KEYS, models=MODELS, variants=VARIANTS, overwrite=True):
    """all combinations of dataset x methods x variant. writes all to csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in keys:
        data = load_data(key)  
        for model, variant, method in product(models, variants, METHODS):
            path = result_path(model, key, variant, method)
            if path.exists() and not overwrite:
                print(f"skip {path.name}")
                continue
            print(f"run  {path.name}")
            if method == "discrete": 
                res = run_one_discrete(variant, model, data, **PARAMS[variant])
            elif method == "ode": 
                res = run_one_ode(variant, model, data, **PARAMS[variant])
            else:
                raise ValueError("Not a valid calculation method, choose ode or discrete")
            res.to_csv(path)
            results[(key, model, variant, method)] = res
    return results

def _prepare(res: pd.DataFrame) -> pd.DataFrame:
    """prepares result dfs for plotting"""
    df = res.reset_index()                       # index 't' or 't_s' 
    time_col = "t_s" if "t_s" in df.columns else "t"
    df = df.rename(columns={time_col: "time"})
    if time_col == "t_s":
        df["time"] = df["time"] / SECONDS_PER_DAY    # temporal: seconds to days 
    if "run" not in df.columns:
        df["run"] = 0
    comps = ["S", "I", "R"]
    df[comps] = df[comps].div(df[comps].sum(axis=1), axis=0)
    return df

def _plot_panel(ax, df: pd.DataFrame, comps, ode = None) -> None:
    multiple_runs = df["run"].nunique() > 1
    for c in comps:
        if multiple_runs:                                   # separate runs thin lines 
            for _, g in df.groupby("run"):
                ax.plot(g["time"], g[c], color=COLORS[c], alpha=0.12, lw=0.8)
        mean = df.groupby("time")[c].mean()                 # gemiddelde dik
        ax.plot(mean.index, mean.values, color=COLORS[c], lw=2, label=c)
        if ode is not None: 
            ax.plot(ode["time"], ode[c], color = COLORS[c], lw = 2, ls="--")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

def plot_results(results: dict, models=MODELS, keys=DATASET_KEYS, variants=VARIANTS, save=True):
    """per model 1 figure: rows = datasets, columns = variants."""
    figs = {}
    for model in models:
        comps = ["S", "I", "R"] if model == "SIR" else ["S", "I"]
        fig, axes = plt.subplots(len(keys), len(variants),
                                figsize=(5.5 * len(variants), 3.6 * len(keys)),
                                sharey=True, squeeze=False)
        for r, key in enumerate(keys):
            for c, variant in enumerate(variants):
                ax = axes[r, c]
                res = results.get((key, model, variant, "discrete"))
                ode = results.get((key, model, variant, "ode"))

                if res is None:
                    ax.text(0.5, 0.5, "geen data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                _plot_panel(ax, _prepare(res), comps, ode=_prepare(ode) if ode is not None else None)
                parameters = PARAMS[variant]
                txt = f"β = {parameters['beta']:g}, γ = {parameters['gamma']:g}, n_seeds = {parameters['n_seeds']:g}"
                if variant == "temporal":   # 1 timestep = 20 s
                    txt += f"\n(infectiontime ≈ {20 / parameters['gamma'] / 3600:.1f} u)"
                ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
                if r == 0:
                    ax.set_title(VARIANT_LABELS.get(variant, variant))
                if c == 0:
                    ax.set_ylabel(f"{DATASET_LABELS.get(key, key)}\nfraction nodes")
                if r == len(keys) - 1:
                    ax.set_xlabel("time (days)" if variant == "temporal" else "timestep")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="k", ls="--", lw=2)); labels.append("ODE (mean-field)")
        fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.suptitle(f"{model}-model", fontsize=14)
        fig.tight_layout(rect=(0, 0, 0.97, 0.97))
        if save:
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIG_DIR / f"{model}_overview.png", dpi=200, bbox_inches="tight")
        figs[model] = fig
    return figs
    
if __name__ == "__main__":
    results = run_all()
    plot_results(results)