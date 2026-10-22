"""Stream 3: temporal metrics, null models and self-driven prediction.

Usage:
  python scripts/run_temporal_characterisation.py --datasets hospital --n-null 3
  python scripts/run_temporal_characterisation.py --n-null 20
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
import pandas as pd  # noqa: E402

from f2f import cache, datasets, prediction, temporal_metrics as tm, temporal_nulls as tn  # noqa: E402

FIG_DIR = datasets.ROOT / "figures"
PROCESSED_DIR = datasets.ROOT / "data" / "processed"
plt.style.use(FIG_DIR / "f2f.mplstyle")


def _null_zscores(real: dict, null_rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for metric, value in real.items():
        if metric == "episodes" or metric not in null_rows:
            continue
        values = null_rows[metric].dropna().to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            {
                "kind": kind,
                "metric": metric,
                "real": value,
                "null_mean": mean,
                "null_std": std,
                "z": (value - mean) / std if std > 0 else float("nan"),
                "null_q025": float(np.quantile(values, 0.025)),
                "null_q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _ccdf(values: np.ndarray):
    values = np.asarray(values)
    x, count = np.unique(values[values > 0], return_counts=True)
    fractions = count / count.sum()
    return x, 1 - np.cumsum(fractions) + fractions


def temporal_figure(plot_data: dict) -> None:
    datasets_order = list(plot_data)
    fig, axes = plt.subplots(
        2, len(datasets_order), figsize=(4.2 * len(datasets_order), 6.2), squeeze=False
    )
    styles = {
        "real": ("C3", "-"),
        "time shuffled": ("C0", "--"),
        "link shuffled": ("C2", ":"),
    }
    for col, key in enumerate(datasets_order):
        for label, version in plot_data[key].items():
            if label not in styles:
                continue
            color, linestyle = styles[label]
            gaps = tm.inter_contact_times(tm.contact_episodes(version)) / 60
            x, y = _ccdf(gaps)
            median_gap = float(np.median(gaps))
            axes[0, col].semilogx(
                x, y, label=f"{label} (median {median_gap:.1f} min)",
                color=color, linestyle=linestyle,
            )
            memory = tm.memory_curve(version, max_lag=12, bin_s=300)
            axes[1, col].plot(
                memory["lag_minutes"], memory["autocorrelation"],
                label=label, color=color, linestyle=linestyle,
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
    fig.suptitle("Temporal burstiness and memory compared with invariant-preserving null models")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "rq1_temporal_vs_null.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS))
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4310)
    ap.add_argument("--swaps-per-contact", type=float, default=3.0)
    ap.add_argument("--bin-s", type=int, default=300)
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--lookback", type=int, default=24)
    args = ap.parse_args()
    if args.n_null < 2:
        ap.error("--n-null must be at least 2 to estimate a null standard deviation")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_summary, all_zscores, all_prediction = [], [], []
    plot_data = {}
    for key in args.datasets:
        started = time.time()
        contacts, nodes = datasets.load_contacts(key)
        node_ids = nodes.index.to_numpy()
        episodes = tm.contact_episodes(contacts)
        gaps = tm.inter_contact_times(episodes)
        activity = tm.activity_potential(contacts, node_ids).reset_index()
        memory = tm.memory_curve(contacts, args.max_lag, args.bin_s, node_ids)
        real = tm.temporal_summary(contacts, node_ids, args.bin_s, args.max_lag)

        cache.save_table(episodes, f"contact_episodes_{key}")
        cache.save_table(pd.DataFrame({"intercontact_s": gaps}), f"intercontact_times_{key}")
        cache.save_table(activity, f"activity_potential_{key}")
        cache.save_table(memory, f"memory_curve_real_{key}")
        all_summary.append(pd.DataFrame([{"dataset": key, "version": "real", **real}]))

        representatives = {"real": contacts}
        for label, kind, shuffle, check in [
            ("time shuffled", "time_shuffled", tn.time_shuffle, tn.assert_time_shuffle_invariants),
            (
                "episode shuffled", "episode_time_shuffled",
                tn.episode_time_shuffle, tn.assert_episode_time_shuffle_invariants,
            ),
            ("link shuffled", "link_shuffled", tn.link_shuffle, tn.assert_link_shuffle_invariants),
        ]:
            null_rows = []
            for run in range(args.n_null):
                kwargs = {"seed": args.seed + run}
                if kind != "episode_time_shuffled":
                    kwargs["swaps_per_contact"] = args.swaps_per_contact
                shuffled = shuffle(contacts, **kwargs)
                check(contacts, shuffled)
                summary = tm.temporal_summary(shuffled, node_ids, args.bin_s, args.max_lag)
                null_rows.append({"run": run, **summary})
                if run == 0:
                    representatives[label] = shuffled
                    shuffled.to_parquet(PROCESSED_DIR / f"{key}_{kind}.parquet", index=False)
                    cache.save_table(
                        tm.memory_curve(shuffled, args.max_lag, args.bin_s, node_ids),
                        f"memory_curve_{kind}_{key}",
                    )
            null_df = pd.DataFrame(null_rows)
            cache.save_table(null_df, f"temporal_null_{kind}_{key}")
            zscores = _null_zscores(real, null_df, kind).assign(dataset=key)
            cache.save_table(zscores, f"temporal_zscores_{kind}_{key}")
            all_zscores.append(zscores)
            null_mean = null_df.drop(columns=["run"]).mean().to_dict()
            all_summary.append(pd.DataFrame([{"dataset": key, "version": kind, **null_mean}]))

        evaluation, predicted, metadata = prediction.evaluate_self_driven(
            contacts, bin_s=args.bin_s, lookback=args.lookback
        )
        evaluation.insert(0, "dataset", key)
        predicted.insert(0, "dataset", key)
        cache.save_table(evaluation, f"prediction_evaluation_{key}")
        cache.save_table(predicted, f"predicted_contacts_{key}")
        cache.save_table(pd.Series(metadata, name=key).to_frame(), f"prediction_metadata_{key}")
        all_prediction.append(evaluation)
        plot_data[key] = representatives
        sd_ap = evaluation.loc[evaluation["method"] == "self_driven", "average_precision"].mean()
        base_ap = evaluation.loc[evaluation["method"] == "frequency_baseline", "average_precision"].mean()
        print(
            f"[{key}] episodes={real['episodes']:,}, duration={real['mean_duration_s']:.1f}s, "
            f"burstiness={real['intercontact_burstiness']:.3f}, lag-1 r={real['autocorrelation_lag1']:.3f}; "
            f"prediction AP={sd_ap:.3f} vs frequency={base_ap:.3f}, tau={metadata['best_tau']} "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )

    cache.save_table(pd.concat(all_summary, ignore_index=True), "temporal_summary_all")
    cache.save_table(pd.concat(all_zscores, ignore_index=True), "temporal_zscores_all")
    cache.save_table(pd.concat(all_prediction, ignore_index=True), "prediction_evaluation_all")
    temporal_figure(plot_data)
    print("wrote Stream 3 tables to results/, shuffles to data/processed/, and figures/rq1_temporal_vs_null.png")


if __name__ == "__main__":
    main()
