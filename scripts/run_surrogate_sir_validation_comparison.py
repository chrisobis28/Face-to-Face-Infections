"""
Stream 5: Validation on held-out days: SIR outbreak and
Stream 5: Same SIR on real, time-shuffled, aggregated and surrogate; final size vs beta (RQ3 figure)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from f2f import cache, sir
from f2f.datasets import ROOT, WINDOW_S, load_contacts
from f2f.surrogate_generator import (generate_surrogate, group_contacts_by_day)
from f2f.temporal_metrics import activity_potential
from f2f.temporal_nulls import time_shuffle
from f2f.views import split_days

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

    windows_per_contact = (times - start) // WINDOW_S

    a = nodes.index.get_indexer(ordered_contacts["i"])
    b = nodes.index.get_indexer(ordered_contacts["j"])

    period = (end - start) // WINDOW_S + 1

    windows = np.unique(np.r_[0, windows_per_contact, period-1])
    bounds = np.r_[np.searchsorted(windows_per_contact, windows), len(windows_per_contact)]

    return a, b, windows, bounds, period

def experiment(dataset_key, real, nodes, surrogates, seeds, betas, gamma=5e-4, runs=100, max_loops=100):
    """Simulate held-out outbreaks on all four network versions"""

    betas = np.asarray(betas, dtype=float)

    start = int(real["t"].min())
    end = int(real["t"].max())

    real_prepared = prepare_temporal_contacts(real, nodes, start, end)

    records = []

    for repeat, surrogate in enumerate(surrogates):

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

        for beta in betas:
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
               .agg(mean="mean", p10=lambda values: values.quantile(0.1), p90=lambda values: values.quantile(0.9))
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