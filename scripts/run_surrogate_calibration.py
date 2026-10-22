"""
Stream 5: Calibration by least squares over m and η to the observed degree distribution and daily contact volume on calibration days
          Least-squares surface figure

To get plots, you can run the following command from the root directory:

python -m scripts.run_surrogate_calibration
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from f2f.datasets import ROOT
import numpy as np
import pandas as pd
from tqdm import tqdm
from f2f.datasets import load_contacts, WINDOW_S
from f2f.temporal_metrics import activity_potential
from f2f.views import split_days
from f2f.surrogate_generator import (group_contacts_by_day, generate_surrogate, calculate_daily_stats, compare_networks)


def calibrate(calibration, nodes, activity, schedule, origin, m_values, eta_values, repeats=5):
    """
        This function calibrates the surrogate model to the observed contact data by trying various m and eta values.

        Params:
            calibration: Real contacts used for fitting
            nodes: The population
            activity: Each person’s activity
            schedule: Contains the earliest and lates timestamps for each day
            origin: Absolute day and time corresponding to t=0
            m_values: Candidate number of partners
            eta_values: Candidate activity multipliers
            repeats: Number of generated sequences per parameter combination
    """
    # Get days
    days = schedule.index.to_numpy()

    # Measure the observed network once using calibration days only
    observed_stat = calculate_daily_stats(calibration, nodes, days)

    info = []
    for m in tqdm(m_values):
        for eta in tqdm(eta_values):
            # Skip parameters that would give an activation probability > 1
            if eta * activity.max() > 1:
                continue

            stored_differences = []
            for repeat in range(repeats):
                # Get activity-driven surrogate network
                surrogate_network = generate_surrogate(nodes=nodes, activity=activity, schedule=schedule, 
                                                       origin=origin, m=m, eta=eta, seed=repeat)
                surrogate_stats = calculate_daily_stats(surrogate_network, nodes,days)
                stored_differences.append(compare_networks(observed_stat, surrogate_stats))

            degree_loss, volume_loss = np.mean(stored_differences, axis=0)
            info.append(
                {
                    "m": int(m),
                    "eta": float(eta),
                    "degree_loss": degree_loss,
                    "volume_loss": volume_loss,
                    "loss": degree_loss + volume_loss
                }
            )
    
    summary = pd.DataFrame(info)
    best_diff = summary.loc[summary["loss"].idxmin()]
    return summary, int(best_diff["m"]), float(best_diff["eta"])

def run_calibration(dataset_key="hospital", repeats=5, eta_count=20):
    "Calibrates one dataset and return the loss table and best parameters"

    # Load contacts and population
    contacts, nodes = load_contacts(dataset_key)

    available_days = np.sort(contacts["day"].unique())

    # Since primary school has 2 days, and others have 5 days
    # We use first day for calibration in primary school data
    # and first three days for calibration in other datasets.
    n_calibration_days = min(3, len(available_days) - 1)

    calibration, validation = split_days(contacts, n_calibration_days)

    # Estimate activity by using calibration contacts
    activity = activity_potential(calibration, nodes.index)["activity_potential"]

    # Create a schedule
    schedule = group_contacts_by_day(calibration)

    # Calculate origin
    origin = (contacts["abs_time"].iloc[0] - pd.Timedelta(seconds=int(contacts["t"].iloc[0])))

    # All possible m value
    m_values = np.arange(1, len(nodes))

    # Get the number of windows
    n_windows = int(((schedule["end"] - schedule["start"]) // WINDOW_S + 1).sum())
    observed_per_window = len(calibration) / n_windows

    # Construct a search space for eta values
    eta_limit = 1.0 / activity.max() # probability must be <= 1
    eta_high = min(2.0 * observed_per_window, 0.999 * eta_limit)
    eta_low = min(observed_per_window / (2.0 * m_values.max()), eta_high / 20.0)
    eta_values = np.geomspace(eta_low,eta_high,eta_count)

    summary, best_m, best_eta = calibrate(
        calibration=calibration,
        nodes=nodes,
        activity=activity,
        schedule=schedule,
        origin=origin,
        m_values=m_values,
        eta_values=eta_values,
        repeats=repeats,
    )

    print(f"\nBest parameters for {dataset_key}: m = {best_m}, eta = {best_eta:.4g}")
    return summary, best_m, best_eta


def plot_surface(summary, dataset_key):
    "Plot least-squares surface figure for calibration"

    fig_key = dataset_key.replace("_", " ").capitalize()

    loss_grid = (summary.pivot(index="m", columns="eta", values="loss").sort_index().sort_index(axis=1))

    fig, ax = plt.subplots(figsize=(9, 5))

    mesh = ax.pcolormesh(
        loss_grid.columns.to_numpy(dtype=float),
        loss_grid.index.to_numpy(dtype=float),
        loss_grid.to_numpy(dtype=float),
        shading="nearest",
        cmap="plasma",
    )

    ax.set_xscale("log")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set(
        xlabel="Activity multiplier (eta)",
        ylabel="Partners selected per activation (m)",
        title=f"{fig_key}: Least-Squares Calibration",
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Calibration Loss",
    )

    output_dir = ROOT / "figures/calibration_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_dir / f"calibration_{dataset_key}.png", dpi=300, bbox_inches="tight",)

if __name__ == "__main__":
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    keys = ["primary_school"]
    for key in keys:
        summary, best_m, best_eta = run_calibration(key)
        summary.to_csv(results_dir / f"calibration_{key}.csv", index=False)
        plot_surface(summary, key)


