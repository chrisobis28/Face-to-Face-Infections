"""
Stream 5: Validation on held-out days: degree distribution, contact volume
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from f2f.datasets import ROOT, load_contacts
from matplotlib.ticker import FuncFormatter
from f2f.temporal_metrics import activity_potential
from f2f.views import split_days
from f2f.surrogate_generator import (group_contacts_by_day, generate_surrogate, calculate_daily_stats)

def run_validation(dataset_key="primary_school", repeats=10):
    """
    Evaluate a frozen calibration fit on held-out days.
    """
    # Get the calibration results summary
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

    surrogates = []

    # Generate independent sequences to estimate average behaviour and variability.
    for repeat in range(repeats):
        generated = generate_surrogate(
            nodes=nodes,
            activity=activity,
            schedule=validation_schedule,
            origin=origin,
            m=best_m,
            eta=best_eta,
            seed=(100+repeat),
        )

        surrogates.append(generated)
    
    # # Measure the observed network on held-out days
    observed_degree_dist, observed_cv = calculate_daily_stats(validation, nodes, validation_days)

    generated_stats = [calculate_daily_stats(network, nodes, validation_days) for network in surrogates]

    generated_degree_dist = np.stack([stats[0] for stats in generated_stats])
    generated_cv = np.stack([stats[1] for stats in generated_stats])

    mean_generated_degree_dist = generated_degree_dist.mean(axis=0)
    mean_generated_volume = generated_cv.mean(axis=0)

    # Save results in a
    validation_table = pd.DataFrame({
        "day": validation_days,
        "observed_volume": observed_cv,
        "surrogate_mean_volume": mean_generated_volume,
        "surrogate_volume_sd": generated_cv.std(axis=0, ddof=1),
        "relative_volume_error": ((mean_generated_volume - observed_cv) / np.maximum(observed_cv, 1)),
        "degree_cdf_rmse": np.sqrt(np.mean((mean_generated_degree_dist - observed_degree_dist) ** 2, axis=1)),
        "mean_degree_cdf_mse": np.mean((generated_degree_dist - observed_degree_dist[None, :, :]) ** 2, axis=(0, 2)),
        "mean_relative_volume_mse": np.mean(((generated_cv - observed_cv[None, :]) / np.maximum(observed_cv[None, :], 1)) ** 2, axis=0)
    })

    metadata = {
        "dataset": dataset_key,
        "m": best_m,
        "eta": best_eta,
        "calibration_days": calibration_days.tolist(),
        "validation_days": validation_days.tolist(),
        "calibration_file": str(calibration_path),
        "network_seeds": [100 + repeat for repeat in range(repeats)]
    }

    # All numerical inputs needed to redraw the figure.
    plot_data = {
        "days": validation_days,
        "observed_dist": observed_degree_dist,
        "observed_volume": observed_cv,
        "generated_dist": generated_degree_dist,
        "generated_volumes": generated_cv,
    }

    return validation_table, plot_data, metadata, surrogates


def plot_validation(plot_data, dataset_key):
    """Compare degree CDFs and contact volume averaged over held-out days."""

    observed_dist = plot_data["observed_dist"]
    observed_volume = plot_data["observed_volume"]
    generated_dist = plot_data["generated_dist"]
    generated_volumes = plot_data["generated_volumes"]
    days = plot_data["days"]

    # Give every held-out day equal weight.
    observed_mean_cdf = observed_dist.mean(axis=0)
    observed_mean_volume = observed_volume.mean()

    generated_mean_cdf = generated_dist.mean(axis=(0, 1))
    generated_mean_volume = generated_volumes.mean()

    degree_grid = np.arange(len(observed_mean_cdf))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # FIRST PLOT: DEGREE DISTRIBUTION CDF
    ax = axes[0]

    ax.step(
        degree_grid,
        observed_mean_cdf,
        where="post",
        color="firebrick",
        linewidth=2,
        label="Observed",
    )

    ax.step(
        degree_grid,
        generated_mean_cdf,
        where="post",
        color="navy",
        linewidth=2,
        label="Surrogate mean",
    )

    ax.set(
        xlabel="Daily Distinct Partners, k",
        ylabel="Mean Fraction of People with degree ≤ k",
        title="Average Daily Degree Distribution",
        ylim=(0, 1.02),
    )

    ax.grid(alpha=0.2)
    ax.legend()

    # SECOND PLOT: VOLUME 
    ax = axes[1]
    ax.bar(
        ["Observed", "Surrogate"],
        [observed_mean_volume, generated_mean_volume],
        color=["lightgray", "chocolate"],
        width=0.6,
    )

    ax.set(
        ylabel="Mean Daily Contact Volume (20-second pair-windows)",
        title="Average Daily Contact Volume",
    )
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"))

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    day_labels = ", ".join(str(day) for day in days)
    fig.suptitle(
        f"{dataset_key.replace('_', ' ').title()}: "
        f"Held-Out Validation — Days {day_labels}"
    )

    figures_dir = ROOT / "figures/surrogate_validation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"validation_{dataset_key}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    dataset_key = "primary_school"

    validation_table, plot_data, metadata, surrogates = run_validation(dataset_key=dataset_key, repeats=5)

    results_dir = ROOT / "results/surrogate_validation"

    results_dir.mkdir(parents=True, exist_ok=True)

    validation_table.to_csv(results_dir / f"validation_{dataset_key}.csv", index=False)

    metadata_path = results_dir / f"metadata_{dataset_key}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2),encoding="utf-8")

    plot_validation(plot_data, dataset_key)
