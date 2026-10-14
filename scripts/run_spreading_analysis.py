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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



FIG_DIR = datasets.ROOT / "figures" / "spreading"
PROCESSED_DIR = datasets.ROOT / "data" / "spreading"
MODELS = ["SIR", "SIS"]
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

def result_path(model: str, key: str, variant: str):
    return PROCESSED_DIR / f"{model}_{key}_discrete_{variant}.csv"

def run_one(variant: str, model: str, data: dict, **params) -> pd.DataFrame:
    if variant == "aggr":
        res = discrete(graph = data["graph"], model=model, **params)
    elif variant == "temporal":
        res = simulate_temporal(contacts=data["contacts"], nodes=data["nodes"], model=model, **params)
    else:
        raise ValueError(f"Onbekende variant: {variant}")
    return res

def load_data(key: str) -> dict:
    contacts, nodes = datasets.load_contacts(key)
    return {"contacts": contacts, "nodes": nodes, "graph": views.aggregate(contacts, nodes)}

def run_all(keys=DATASET_KEYS, models=MODELS, variants=VARIANTS, overwrite=True):
    """all combinations of dataset x model x variant. writes all to csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in keys:
        data = load_data(key)  
        for model, variant in product(models, variants):
            path = result_path(model, key, variant)
            if path.exists() and not overwrite:
                print(f"skip {path.name}")
                continue
            print(f"run  {path.name}")
            res = run_one(variant, model, data, **PARAMS[variant])
            res.to_csv(path)
            results[(key, model, variant)] = res
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

def _plot_panel(ax, df: pd.DataFrame, comps) -> None:
    multiple_runs = df["run"].nunique() > 1
    for c in comps:
        if multiple_runs:                                   # separate runs thin lines 
            for _, g in df.groupby("run"):
                ax.plot(g["time"], g[c], color=COLORS[c], alpha=0.12, lw=0.8)
        mean = df.groupby("time")[c].mean()                 # gemiddelde dik
        ax.plot(mean.index, mean.values, color=COLORS[c], lw=2, label=c)
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
                res = results.get((key, model, variant))
                if res is None:
                    ax.text(0.5, 0.5, "geen data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                _plot_panel(ax, _prepare(res), comps)
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