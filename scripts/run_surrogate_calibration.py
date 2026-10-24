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
from scipy.interpolate import griddata
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
        for eta in tqdm(eta_values[m]):
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

def run_calibration(dataset_key="hospital", repeats=5, eta_count=50):
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

    # For candidate m values, we used the maximum m values in each dataset and evaluate the model with [1,max_m]
    max_m_in_data = 4 if dataset_key== "high_school" or dataset_key == "primary_school" else 7
    m_values = np.arange(1, max_m_in_data+1)

    # Get the number of windows
    n_windows = int(((schedule["end"] - schedule["start"]) // WINDOW_S + 1).sum())
    observed_per_window = len(calibration) / n_windows

    # Construct a search space for eta values
    eta_limit = 1.0 / activity.max() # probability must be <= 1
    search_factor = 5.0
    eta_values_by_m = {}
    for m in m_values:
        eta_target = observed_per_window / (m * activity.sum())
        eta_low = max(min(eta_target/search_factor, eta_limit/search_factor), 1e-8 )
        eta_high = min(eta_target * search_factor, 0.999 * eta_limit)
        if eta_low >= eta_high:
            raise ValueError(f"No valid eta interval for m={m}")
        eta_values_by_m[m] = np.geomspace(eta_low, eta_high, eta_count)

    summary, best_m, best_eta = calibrate(
        calibration=calibration,
        nodes=nodes,
        activity=activity,
        schedule=schedule,
        origin=origin,
        m_values=m_values,
        eta_values=eta_values_by_m,
        repeats=repeats,
    )

    best = summary.loc[summary["loss"].idxmin()]
    print(f"\nBest parameters for {dataset_key}: m = {best_m}, eta = {best_eta:.4g}")
    print(f"degree_loss={best['degree_loss']:.6g}")
    print(f"volume_loss={best['volume_loss']:.6g}")
    print(f"total_loss={best['loss']:.6g}")

    best_eta_grid = eta_values_by_m[best_m]
    if np.isclose(best_eta, best_eta_grid[0]) or np.isclose(best_eta, best_eta_grid[-1]):
        print("Check the eta search boundary before accepting this fit.")

    return summary, best_m, best_eta

def plot_surface_3d(summary, dataset_key):
    """
        Plot a 3D calibration loss surface with the evaluated points
        and the best configuration.
    """

    best = summary.loc[summary["loss"].idxmin()]

    # Original sampled coordinates which are eta and m
    x = np.log10(summary["eta"].to_numpy())
    y = summary["m"].to_numpy()

    # Log-transform the loss (z coordinate)
    z = np.log10(1+summary["loss"].to_numpy())

    # Create a grid
    x_i = np.linspace(x.min(), x.max(), 180)
    y_i = np.linspace(y.min(), y.max(), 180)
    X,Y = np.meshgrid(x_i, y_i)

    # Linear interpolation of loss
    Z = griddata((x,y), z, (X,Y), method="linear")
    Z = np.ma.masked_invalid(Z)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Continous surface of loss
    surface = ax.plot_surface(X,Y,Z,cmap="plasma", edgecolor="none", 
                              antialiased=True, alpha=0.95, rcount=180,
                              ccount=180, zorder=1)

    # Plot evaluated configuration on the surface
    # ax.scatter(x,y,z, color="black", s=5,alpha=0.4, depthshade=False,
    #            label="Evaluated configurations",zorder=2)
    
    best_x = np.log10(best["eta"])
    best_y = best["m"]
    best_z = np.log10(1+best["loss"])

    # Plot the best configuration on the surface
    ax.scatter(best_x, best_y, best_z, marker="*", s=400, color="gold", edgecolors="black",
               linewidths=1.2, depthshade=True, label=f"Best configuration: m:{best['m']}, eta:{best['eta']:.3f}, loss:{best['loss']:.3f}")
    
    ax.set(
        xlabel="Activity Multiplier (eta) on log10 scale",
        ylabel="Partners per activation (m)",
        zlabel="Calibration loss on log10 scale",
        title=f"{dataset_key.replace('_', ' ').capitalize()}: 3D Calibration Loss"
    )

    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.12, label="Calibration Loss")
    ax.legend(loc="upper right")

    output_dir = ROOT / "figures/calibration_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"calibration_3d_{dataset_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    keys = ["hospital", "high_school", "primary_school"]
    for key in keys:
        summary, best_m, best_eta = run_calibration(key)
        summary.to_csv(results_dir / f"calibration_{key}.csv", index=False)
        plot_surface_3d(summary, key)

