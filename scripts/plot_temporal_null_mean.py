"""Plot real temporal metrics against null means over repeated shuffles.

This is a presentation-oriented alternative to ``rq1_temporal_vs_null.png``.
It reruns the time and link shuffles because the per-run distribution and memory
curves are not retained by the main characterisation pipeline.
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

from f2f import datasets, temporal_metrics as tm, temporal_nulls as tn  # noqa: E402

FIG_DIR = datasets.ROOT / "figures"
plt.style.use(FIG_DIR / "f2f.mplstyle")


def _gaps_minutes(contacts) -> np.ndarray:
    return tm.inter_contact_times(tm.contact_episodes(contacts)) / 60


def _survival(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical Pr[value >= x] evaluated on a shared x-grid."""
    ordered = np.sort(np.asarray(values, dtype=float))
    return (ordered.size - np.searchsorted(ordered, x, side="left")) / ordered.size


def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical survival curve at every observed positive value."""
    values = np.asarray(values)
    x, count = np.unique(values[values > 0], return_counts=True)
    fractions = count / count.sum()
    return x, 1 - np.cumsum(fractions) + fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS))
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4310)
    parser.add_argument("--swaps-per-contact", type=float, default=3.0)
    parser.add_argument("--bin-s", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=12)
    args = parser.parse_args()
    if args.n_null < 1:
        parser.error("--n-null must be at least 1")

    versions = {
        "time shuffled": (tn.time_shuffle, "C0", "--"),
        "link shuffled": (tn.link_shuffle, "C2", ":"),
    }
    figure_data = {}

    for key in args.datasets:
        started = time.time()
        contacts, nodes = datasets.load_contacts(key)
        node_ids = nodes.index.to_numpy()
        real_gaps = _gaps_minutes(contacts)
        real_memory = tm.memory_curve(
            contacts, max_lag=args.max_lag, bin_s=args.bin_s, node_ids=node_ids
        )
        null_data = {}

        for label, (shuffle, color, linestyle) in versions.items():
            gap_runs = []
            memory_runs = []
            for run in range(args.n_null):
                shuffled = shuffle(
                    contacts,
                    seed=args.seed + run,
                    swaps_per_contact=args.swaps_per_contact,
                )
                gap_runs.append(_gaps_minutes(shuffled))
                memory_runs.append(
                    tm.memory_curve(
                        shuffled,
                        max_lag=args.max_lag,
                        bin_s=args.bin_s,
                        node_ids=node_ids,
                    )["autocorrelation"].to_numpy()
                )
            null_data[label] = {
                "gaps": gap_runs,
                "memory": np.asarray(memory_runs),
                "color": color,
                "linestyle": linestyle,
            }
            print(f"[{key}] completed {args.n_null} {label} runs", flush=True)

        all_gaps = [real_gaps]
        for values in null_data.values():
            all_gaps.extend(values["gaps"])
        positive_min = min(float(gaps[gaps > 0].min()) for gaps in all_gaps)
        positive_max = max(float(gaps.max()) for gaps in all_gaps)
        x_grid = np.geomspace(positive_min, positive_max, 220)

        figure_data[key] = {
            "real_gaps": real_gaps,
            "real_memory": real_memory,
            "nulls": null_data,
            "x_grid": x_grid,
        }
        print(f"[{key}] finished in {time.time() - started:.1f}s", flush=True)

    fig, axes = plt.subplots(
        2, len(figure_data), figsize=(4.2 * len(figure_data), 6.2), squeeze=False
    )
    for col, (key, values) in enumerate(figure_data.items()):
        x_grid = values["x_grid"]
        real_gaps = values["real_gaps"]
        real_x, real_y = _ccdf(real_gaps)
        axes[0, col].semilogx(
            real_x,
            real_y,
            color="C3",
            label=f"real (median {np.median(real_gaps):.1f} min)",
        )
        real_memory = values["real_memory"]
        axes[1, col].plot(
            real_memory["lag_minutes"],
            real_memory["autocorrelation"],
            color="C3",
            label="real",
        )

        for label, null in values["nulls"].items():
            survival_runs = np.asarray(
                [_survival(gaps, x_grid) for gaps in null["gaps"]]
            )
            mean_median = float(np.mean([np.median(gaps) for gaps in null["gaps"]]))
            survival_mean = survival_runs.mean(axis=0)
            axes[0, col].semilogx(
                x_grid,
                survival_mean,
                color=null["color"],
                linestyle=null["linestyle"],
                label=f"{label} (median {mean_median:.1f} min)",
            )

            memory_mean = null["memory"].mean(axis=0)
            axes[1, col].plot(
                real_memory["lag_minutes"],
                memory_mean,
                color=null["color"],
                linestyle=null["linestyle"],
                label=label,
            )

        axes[0, col].set_title(key.replace("_", " "))
        axes[0, col].set_xlabel("within-day inter-contact time (min)")
        axes[0, col].set_ylim(0, 1.02)
        axes[0, col].legend(fontsize=8, loc="upper right")
        axes[1, col].set_xlabel("lag (min)")
        if col == 0:
            axes[0, col].set_ylabel("fraction of gaps at least x")
            axes[1, col].set_ylabel("link-activity autocorrelation")
            axes[1, col].legend()

    fig.suptitle(f"Temporal metrics: null means over {args.n_null} randomizations")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    output = FIG_DIR / "rq1_temporal_vs_null_mean.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
