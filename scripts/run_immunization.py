"""Stream 6: immunisation strategies (RQ4) and nodes as seeds vs immunisation targets (RQ2).

Immunise f*N nodes at day 0 (midterm feedback), chosen by each strategy in
`f2f.immunization`, and run SIR with 100 runs per cell on three network versions:
the real contact sequence, its time-shuffled null and the aggregated weighted network.
Node rankings always come from the real data; the version only changes the network the
epidemic runs on. All runs of all strategies share random numbers (same rng_seed).

Stages (results/<name>.parquet, row format of f2f.cache plus extra columns):
  nodes     per node: final size when it is the only seed, and when it is the only node
            immunised (main setting, real network). Gives the oracle ranking.
  grid      every setting x dataset x version x strategy x budget, 1 random seed
            (the zhang setting on the real network only)
  seeds5    main setting, real network, 5 random seeds instead of 1
  seedsets  main setting, real network: each strategy's 10 % set used as the seed set
Tables: results/immunization_pruned.csv, immunization_curves.parquet, immunization_node_table_<ds>.csv

    python scripts/run_immunization.py --quick        # hospital, 20 runs, ~1 min
    python scripts/run_immunization.py                # everything, ~30 min on 8 cores
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f2f import cache, datasets, immunization as im, sir, static_metrics, temporal_nulls, views  # noqa: E402

SETTINGS = {"low": (0.03, 5e-4), "team": (0.1, 5e-4), "zhang": (0.01, 2.44e-5)}
MAIN = "team"
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
VERSIONS = ["real", "time_shuffled", "aggregated"]
DATASETS = ["hospital", "primary_school", "high_school"]
RNG = 4310
PROCESSED = datasets.ROOT / "data" / "processed"
RUNS = 100


@lru_cache(maxsize=None)
def data(key: str) -> dict:
    contacts, nodes = datasets.load_contacts(key)
    path = PROCESSED / f"{key}_time_shuffled.parquet"
    if path.exists():
        shuffled = pd.read_parquet(path)
    else:
        shuffled = temporal_nulls.time_shuffle(contacts, seed=0)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        shuffled.to_parquet(path)
    return {
        "contacts": contacts, "nodes": nodes, "matrix": sir.contact_matrix(contacts, nodes.index),
        "real": sir.windows_of(contacts, nodes.index),
        "time_shuffled": sir.windows_of(shuffled, nodes.index),
    }


@lru_cache(maxsize=None)
def scores(key: str, setting: str) -> pd.DataFrame:
    d = data(key)
    out = im.node_scores(d["contacts"], d["nodes"], *SETTINGS[setting])
    oracle = cache.result_path("immunization_nodes")
    if oracle.exists():
        out["oracle"] = oracle_scores(pd.read_parquet(oracle), key).reindex(out.index)
    return out


def oracle_scores(nodes_df: pd.DataFrame, key: str) -> pd.Series:
    """Drop in mean final size when only node i is immunised (Starnini's tau-protocol)."""
    df = nodes_df[(nodes_df["dataset"] == key) & (nodes_df["role"] == "blocked") & (nodes_df["outcome"] == "final_size")]
    base = df.loc[df["node"] == -1, "value"].mean()
    return base - df[df["node"] >= 0].groupby("node")["value"].mean()


def epidemic(key, version, beta, gamma, seeds=1, immunised=None, runs=RUNS):
    d = data(key)
    if version == "aggregated":
        return sir.simulate_aggregated(d["contacts"], d["nodes"], seeds, beta, gamma, immunised, runs, RNG)
    return sir.simulate(d["contacts"], d["nodes"], seeds, beta, gamma, immunised, runs, RNG, prepared=d[version])


def rows(ep: sir.Epidemic, **meta) -> pd.DataFrame:
    s = pd.DataFrame(ep.summary()).rename_axis("run").reset_index()
    out = s.melt(id_vars="run", var_name="outcome", value_name="value")
    for k, v in meta.items():
        out[k] = v
    return out


# ---- stage workers (top level so they pickle) ----

def node_task(key: str, chunk: list[int], runs: int) -> pd.DataFrame:
    beta, gamma = SETTINGS[MAIN]
    nodes = data(key)["nodes"]
    out = []
    for pos in chunk:
        meta = dict(dataset=key, version="real", beta=beta, gamma=gamma, setting=MAIN)
        if pos >= 0:
            out.append(rows(epidemic(key, "real", beta, gamma, seeds=[pos], runs=runs),
                            seed=int(nodes.index[pos]), role="seed", node=int(nodes.index[pos]), **meta))
        out.append(rows(epidemic(key, "real", beta, gamma, immunised=[pos] if pos >= 0 else None, runs=runs),
                        seed=-1, role="blocked", node=int(nodes.index[pos]) if pos >= 0 else -1, **meta))
    return pd.concat(out)


def grid_task(setting: str, key: str, version: str, strategy: str, n_seeds: int, runs: int):
    beta, gamma = SETTINGS[setting]
    d = data(key)
    sc = scores(key, setting)
    n = len(d["nodes"])
    meta = dict(dataset=key, version=version, beta=beta, gamma=gamma, seed=-1, setting=setting, n_seeds=n_seeds)
    out, pruned, curves = [], [], []
    budgets = [0.0] if strategy == "none" else BUDGETS
    for f in budgets:
        mask = im.immunisation_masks(strategy if f else "random", round(f * n), sc, d["matrix"], runs, RNG)
        ep = epidemic(key, version, beta, gamma, seeds=n_seeds, immunised=mask, runs=runs)
        out.append(rows(ep, strategy=strategy, budget=f, **meta))
        if version == "real" and n_seeds == 1 and f in (0.0, 0.10):
            grid, prev = ep.prevalence()
            curves.append(pd.DataFrame({"t_h": grid / sir.WINDOWS_PER_HOUR, "prevalence": prev.mean(axis=0),
                                        "dataset": key, "setting": setting, "strategy": strategy, "budget": f}))
        if version == "real" and n_seeds == 1 and f > 0:
            draws = 5 if strategy in im.RANDOMISED else 1
            m = pd.DataFrame([im.pruned_metrics(d["matrix"], mask[r]) for r in range(draws)]).mean()
            pruned.append({"dataset": key, "setting": setting, "strategy": strategy, "budget": f, **m.to_dict()})
    return pd.concat(out), pruned, curves


def seedset_task(key: str, strategy: str, runs: int) -> pd.DataFrame:
    beta, gamma = SETTINGS[MAIN]
    d = data(key)
    k = round(0.10 * len(d["nodes"]))
    mask = im.immunisation_masks(strategy, k, scores(key, MAIN), d["matrix"], runs, RNG)
    ep = epidemic(key, "real", beta, gamma, seeds=mask, runs=runs)
    return rows(ep, dataset=key, version="real", beta=beta, gamma=gamma, seed=-1, setting=MAIN,
                strategy=strategy, budget=0.10)


def node_table(key: str) -> pd.DataFrame:
    """Every node property used in RQ2 next to its role as seed and as blocker."""
    d = data(key)
    sc = scores(key, MAIN)
    cent = static_metrics.centralities(views.aggregate(d["contacts"], d["nodes"])).reindex(sc.index)
    props = im.node_temporal_properties(d["contacts"], d["nodes"].index)
    nd = pd.read_parquet(cache.result_path("immunization_nodes"))
    nd = nd[(nd["dataset"] == key) & (nd["outcome"] == "final_size")]
    seed = nd[nd["role"] == "seed"].groupby("node")["value"].mean()
    table = pd.concat([d["nodes"][["group"]], sc, cent[["kshell", "closeness"]], props], axis=1)
    table["seed_influence"] = seed
    table["blocking_importance"] = sc["oracle"]
    return table


def run_parallel(fn, tasks, workers):
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, *t) for t in tasks]
        for i, fut in enumerate(futures, 1):
            yield fut.result()
            if i % max(1, len(futures) // 10) == 0:
                print(f"    {i}/{len(futures)} tasks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stages", nargs="+", default=["nodes", "grid", "seeds5", "seedsets"])
    ap.add_argument("--quick", action="store_true", help="hospital only, 20 runs, main setting")
    a = ap.parse_args()
    if a.quick:
        a.datasets, a.settings, a.runs = ["hospital"], [MAIN], 20
    for key in a.datasets:  # build shuffles once, before the workers need them
        data(key)
    t0 = time.time()

    if "nodes" in a.stages:
        print("stage nodes: every node as single seed and as single immunised node")
        tasks = []
        for key in a.datasets:
            pos = [-1] + list(range(len(data(key)["nodes"])))
            tasks += [(key, pos[i:i + 8], a.runs) for i in range(0, len(pos), 8)]
        cache.save_result(pd.concat(run_parallel(node_task, tasks, a.workers)), "immunization_nodes")
        scores.cache_clear()
        for key in a.datasets:
            cache.save_table(node_table(key), f"immunization_node_table_{key}")

    strategies = ["none"] + im.STRATEGIES
    if "grid" in a.stages:
        print("stage grid: setting x dataset x version x strategy x budget")
        # shortcut: the slow zhang setting (10-day infectious period) runs on the real network only,
        # as a robustness check of the ranking; versions are compared in the low and team settings.
        tasks = [(s, k, v, st, 1, a.runs) for s in a.settings for k in reversed(a.datasets)
                 for v in (["real"] if s == "zhang" else VERSIONS) for st in strategies]
        tasks.sort(key=lambda t: t[0] != "zhang")  # longest tasks first
        res = list(run_parallel(grid_task, tasks, a.workers))
        cache.save_result(pd.concat(r[0] for r in res), "immunization_grid")
        cache.save_table(pd.DataFrame([p for r in res for p in r[1]]), "immunization_pruned")
        pd.concat([c for r in res for c in r[2]]).to_parquet(cache.result_path("immunization_curves"))

    if "seeds5" in a.stages:
        print("stage seeds5: 5 random seeds")
        tasks = [(MAIN, k, "real", st, 5, a.runs) for k in a.datasets for st in strategies]
        cache.save_result(pd.concat(r[0] for r in run_parallel(grid_task, tasks, a.workers)), "immunization_seeds5")

    if "seedsets" in a.stages:
        print("stage seedsets: immunisation sets used as seed sets")
        tasks = [(k, st, a.runs) for k in a.datasets for st in im.STRATEGIES]
        cache.save_result(pd.concat(run_parallel(seedset_task, tasks, a.workers)), "immunization_seedsets")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
