"""
Stream 5: Validation on held-out days: SIR outbreak and
Stream 5: Same SIR on real, time-shuffled, aggregated and surrogate; final size vs beta (RQ3 figure)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from f2f import cache, sir
from f2f.datasets import ROOT, WINDOW_S, load_contacts
from f2f.surrogate_generator import (generate_surrogate, group_contacts_by_day)
from f2f.temporal_metrics import activity_potential
from f2f.temporal_nulls import time_shuffle
from f2f.views import split_days
from tqdm import tqdm

def prepare_heldout_networks(dataset_key, repeats=5):
    """Load fitted parameters and generate held-out surrogate sequences"""

    calibration_path = ROOT / f"results/calibration_{dataset_key}.csv"
    summary = pd.read_csv(calibration_path)

    # Get best calibration loss, best m and eta values
    best = summary.loc[summary["loss"].idxmin()]
    best_m = int(best["m"])
    best_eta = float(best["eta"])


    # Reproduce the split 
    contacts, nodes = load_contacts(dataset_key)
    available_days = np.sort(contacts["day"].unique())
    n_calibration_days = min(3, len(available_days) - 1)

    # Get calibration and validation splits
    calibration, validation = split_days(contacts, n_calibration_days)
    calibration_days = np.sort(calibration["day"].unique())
    validation_days = np.sort(validation["day"].unique())


    # Activities estimated from calibration contacts only.
    activity = activity_potential(calibration, nodes.index)["activity_potential"]

    # Create a schedule by using validation split
    validation_schedule = group_contacts_by_day(validation)

    origin = (contacts["abs_time"].iloc[0] - pd.Timedelta(seconds=int(contacts["t"].iloc[0])))

    network_seeds = [100 + repeat for repeat in range(repeats)]
    
    surrogates = []

    # Generate independent sequences to estimate average behaviour and variability.
    for seed in network_seeds:
        generated = generate_surrogate(
            nodes=nodes,
            activity=activity,
            schedule=validation_schedule,
            origin=origin,
            m=best_m,
            eta=best_eta,
            seed=(seed),
        )

        surrogates.append(generated)

    metadata = {
        "dataset": dataset_key,
        "m": best_m,
        "eta": best_eta,
        "calibration_days": calibration_days.tolist(),
        "validation_days": validation_days.tolist(),
        "calibration_file": str(calibration_path),
        "network_seeds": network_seeds
    }

    return validation, nodes, surrogates, metadata

def prepare_temporal_contacts(contacts, nodes, start, end):
    """Prepare temporal contacts using a shared observation period."""

    if end < start or (end - start) % WINDOW_S != 0:
        raise ValueError("Invalid observation-period boundaries")
    
    ordered_contacts = contacts.sort_values(["t", "i", "j"])
    times = ordered_contacts["t"].to_numpy()

    # Some checks
    if not nodes.index.is_unique:
        raise ValueError("Node IDs must be unique")

    if (not np.isfinite(times).all() or ((times <start) | (times>end)).any() or ((times - start) % WINDOW_S != 0).any()):
        raise ValueError("Contacts are outside the observation period")

    windows_per_contact = (times - start) // WINDOW_S

    a = nodes.index.get_indexer(ordered_contacts["i"])
    b = nodes.index.get_indexer(ordered_contacts["j"])

    if (a<0).any() or (b<0).any():
        raise ValueError("Contact endpoints missing from nodes")

    period = (end - start) // WINDOW_S + 1

    windows = np.unique(np.r_[0, windows_per_contact, period-1])
    bounds = np.r_[np.searchsorted(windows_per_contact, windows), len(windows_per_contact)]

    return a, b, windows, bounds, period

def experiment(dataset_key, real, nodes, surrogates, betas, gamma=5e-4, runs=100, max_loops=100):
    """Simulate held-out outbreaks on all four network versions"""

    betas = np.asarray(betas, dtype=float)

    start = int(real["t"].min())
    end = int(real["t"].max())

    real_prepared = prepare_temporal_contacts(real, nodes, start, end)

    records = []

    for repeat, surrogate in tqdm(enumerate(surrogates)):

        shuffled = time_shuffle(real, seed=(300+repeat))

        networks = {
            "real": real,
            "time_shuffled": shuffled,
            "surrogate":  surrogate
        }

        prepared = {
            "real": real_prepared,
            "time_shuffled": prepare_temporal_contacts(shuffled, nodes, start, end),
            "surrogate": prepare_temporal_contacts(surrogate, nodes, start, end)
        }


        # Choose one initial infected person per epidemic run
        # Every network version receives the same initial people
        rng = np.random.default_rng(400+repeat)
        positions = rng.integers(len(nodes), size=runs)

        initial = np.zeros((runs, len(nodes)), dtype=bool)
        initial[np.arange(runs), positions] = True

        for beta in tqdm(betas):
            for version in ["real", "time_shuffled", "aggregated", "surrogate"]:

                # Initialize common settings
                settings = {
                    "seeds": initial,
                    "beta": beta,
                    "gamma":  gamma,
                    "runs":  runs,
                    "rng_seed":  400+repeat,
                    "max_loops": max_loops
                }

                # Run SIR simulation
                if version == "aggregated":
                    epidemic = sir.simulate_aggregated(real, nodes, step=1, **settings)
                else:
                    epidemic = sir.simulate(networks[version], nodes, prepared=prepared[version], **settings)
                
                if epidemic.still_active > 0:
                    raise RuntimeError("Increase max_loops")
                
                # SIR outcome
                # Fraction of nodes ever infected in each epidemic run
                final_sizes = epidemic.final_size()

                records.append(pd.DataFrame({
                    "dataset": dataset_key,
                    "version": version,
                    "beta": beta,
                    "gamma": gamma,
                    "seed": nodes.index.to_numpy()[positions],
                    "run": repeat * runs + np.arange(runs),
                    "outcome": "final_size",
                    "value": final_sizes,
                    "network_repeat": repeat,
                    "period_windows": real_prepared[-1],
                    "max_loops": max_loops,
                }))
    
    return cache.validate(pd.concat(records, ignore_index=True))


def summarise_sir(results):
    """Summarise outbreak sizes and calculate differences from real data."""

    summary = (results.groupby(["version", "beta"])["value"]
               .agg(mean="mean")
               .reset_index()
    )

    comparison = summary.pivot(index="beta", columns="version", values="mean")

    # HELD-OUT SIR VALIDATION:
    # Difference between generated and observed mean outbreak sizes.
    comparison["surrogate_minus_real_pp"] = 100 * (comparison["surrogate"] - comparison["real"])

    # RQ3: effects of changing the temporal representation.
    comparison["shuffle_minus_real_pp"] = 100 * (comparison["time_shuffled"] - comparison["real"])

    comparison["aggregated_minus_real_pp"] = 100 * (comparison["aggregated"] - comparison["real"])

    return summary, comparison

def plot_sir_comparison(summary, comparison, dataset_key):
    """ Mean final sizes, outbreak variability, and differences from real """

    labels = {
        "real": "Real",
        "time_shuffled": "Time Shuffled",
        "aggregated": "Aggregated",
        "surrogate": "Surrogate"
    }

    colors = {
        "real": "black",
        "time_shuffled": "navy",
        "aggregated": "tab:orange",
        "surrogate": "tab:green"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    for version, label in labels.items():
        data = summary.loc[summary["version"].eq(version)].sort_values("beta")

        beta_values = data["beta"].to_numpy()
        axes[0].plot(beta_values, data["mean"].to_numpy(), label=label, color=colors[version])
    axes[0].set(
        xlabel="Transmission probability of a disease (beta) per window",
        ylabel="Fraction ever infected",
        title="Mean Final Size vs Beta Values",
        ylim=(0,1)
    )

    columns = {
        "time_shuffled": "shuffle_minus_real_pp",
        "aggregated": "aggregated_minus_real_pp",
        "surrogate": "surrogate_minus_real_pp"
    }

    comparison = comparison.sort_index()
    for version, column in columns.items():
        axes[1].plot(comparison.index.to_numpy(), comparison[column].to_numpy(), label=labels[version], color=colors[version])
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Transmission probability of a disease (beta) per window",
        ylabel="Difference fro real (percentage points)",
        title="Mean final-size differences"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    
    fig.suptitle(f"{dataset_key.replace('_', ' ').title()}: SIR on Held-Out Contacts")

    output = ROOT / "figures/sir_four_way_comparison"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/f"{dataset_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def export_surrogate(real, nodes, surrogates, metadata):
    """ 
    Export validated surrogate contact sequence for Stream 6.
    Stream 6 can run the following command to read them:
    ```python
    nodes = pd.read_csv(folder / "nodes.csv", index_col="node_id")
    contacts = pd.read_csv(folder / f"surrogate_seed_{seed}.csv",parse_dates=["abs_time"])
    ```
    """
    output = ROOT / f"data/surrogate_contact_sequences/{metadata['dataset']}"
    output.mkdir(parents=True, exist_ok=True)

    if len(surrogates) != len(metadata["network_seeds"]):
        raise ValueError("Surrogate and network seed counts differ")
    
    start = int(real["t"].min())
    end = int(real["t"].max())

    files = []
    for contacts, network_seed in zip(surrogates, metadata["network_seeds"]):
        prepare_temporal_contacts(contacts, nodes, start, end)

        contacts.to_csv(output / f"surrogate_seed_{network_seed}.csv", index=False)
        files.append({
            "network_seed": network_seed,
            "file": f"surrogate_seed_{network_seed}.csv",
            "contact_rows": len(contacts)
        })

    nodes.to_csv(output/"nodes.csv", index=True, index_label="node_id")
    manifest = {
        **metadata,
        "observation_start_t": start,
        "observation_end_t": end,
        "window_s": int(WINDOW_S),
        "period_windows": (end - start) // WINDOW_S + 1,
        "schedule_conditioning": "Observed held-out daily start/end times",
        "replay": "Periodic repetition of the whole held-out interval",
        "nodes_file": "nodes.csv",
        "surrogate_files": files,
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def run_four_way_comparison(dataset_key, betas, gamma=5e-4, runs=100, max_loops=100, repeats=10):
    """ Run the four way comparison and save results """
    
    real, nodes, surrogates, metadata = prepare_heldout_networks(dataset_key, repeats)

    results = experiment(dataset_key=dataset_key, real=real, nodes=nodes,surrogates=surrogates,betas=betas,gamma=gamma,
                         runs=runs, max_loops=max_loops)
    
    summary, comparison = summarise_sir(results)

    # Surrogate vs real SIR comparison for Stream 5 task
    error_pp = comparison["surrogate_minus_real_pp"].to_numpy()
    report = {
        "mae_pp": np.abs(error_pp).mean(),
        "rmse_pp": np.sqrt(np.mean(error_pp**2)),
        "max_abs_error_pp": np.abs(error_pp).max()
    }

    metadata.update({
        "betas": np.asarray(betas, dtype=float).tolist(),
        "gamma": float(gamma),
        "runs_per_network_repeat": runs,
        "max_loops": max_loops,
        "epidemic_seeds": [400 + repeat for repeat in range(repeats)],
        "shuffle_seeds": [300 + repeat for repeat in range(repeats)],
        "final_size_denominator": len(nodes),
        "aggregated_step_windows": 1,
        "sir_surrogate_validation": report
    })

    output = ROOT / f"results/surrogate_sir/{dataset_key}"
    output.mkdir(parents=True, exist_ok=True)

    cache.validate(results)    
    results.to_csv(output/"sir_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparison.to_csv(output/"comparison.csv")

    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_sir_comparison(summary=summary, comparison=comparison, dataset_key=dataset_key)
    export_surrogate(real,nodes,surrogates,metadata)

if __name__ == "__main__":
    datasets = ["hospital", "primary_school", "high_school"]
    betas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    for key in datasets:
        run_four_way_comparison(key, betas=betas)