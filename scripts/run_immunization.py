"""Stream 6: immunisation strategies (RQ4) and nodes as seeds vs immunisation targets (RQ2).

Immunise f*N nodes at day 0 (midterm feedback), chosen by each strategy in
`f2f.immunization`, and run SIR with 100 runs per cell on three network versions:
the real contact sequence, its time-shuffled null and the aggregated weighted network.
Node rankings always come from the real data; the version only changes the network the
epidemic runs on. Every scenario of a stage is simulated side by side (`blocks`) with the
same random numbers per run, so strategy differences are not run-to-run noise.

Stages (results/<name>.parquet, row format of f2f.cache plus extra columns):
  nodes     per node, main setting, real network: final size when it is the only seed
            (100 runs), and infections among the other nodes when it is the only node
            immunised vs not (400 runs: single-node effects are small). Gives the oracle.
  grid      every setting x dataset x version x strategy x budget, 1 random seed
  seeds5    main setting, real network, 5 random seeds instead of 1
  seedsets  main setting, real network: each ranking's top 5 nodes used as the 5 seeds
Tables: results/immunization_pruned.csv, immunization_curves.parquet, immunization_node_table_<ds>.csv

    python scripts/run_immunization.py --quick        # hospital, 20 runs, ~1 min
    python scripts/run_immunization.py                # everything, ~45 min on a 4-core laptop
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
BLOCK_RUNS_FACTOR = 4  # runs for single-node immunisation = 4 x RUNS
NODE_CHUNK = 12
SEEDSET_K = 5  # same number of seeds as the seeds5 stage


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
    path = cache.result_path("immunization_nodes")
    if path.exists():
        out["oracle"] = oracle_scores(pd.read_parquet(path), key).reindex(out.index)
    return out


def oracle_scores(nodes_df: pd.DataFrame, key: str) -> pd.Series:
    """Drop in the infections among the other nodes when only node i is immunised (Starnini's
    tau-protocol with final size). Excluding i itself removes the trivial 1/N effect."""
    df = nodes_df[(nodes_df["dataset"] == key) & (nodes_df["outcome"] == "others_infected")]
    m = df.groupby(["role", "node"])["value"].mean()
    return m["baseline"] - m["blocked"]


def epidemic(key, version, beta, gamma, seeds=1, immunised=None, runs=RUNS, blocks=1):
    d = data(key)
    if version == "aggregated":
        return sir.simulate_aggregated(d["contacts"], d["nodes"], seeds, beta, gamma, immunised, runs, RNG,
                                       blocks=blocks)
    return sir.simulate(d["contacts"], d["nodes"], seeds, beta, gamma, immunised, runs, RNG,
                        prepared=d[version], blocks=blocks)


def rows(ep: sir.Epidemic, **meta) -> pd.DataFrame:
    s = pd.DataFrame(ep.summary()).rename_axis("run").reset_index()
    out = s.melt(id_vars="run", var_name="outcome", value_name="value")
    for k, v in meta.items():
        out[k] = v
    return out


def one_hot(n: int, runs: int, positions) -> np.ndarray:
    """(len(positions) * runs, n) mask; block b marks node positions[b] (or nothing if -1)."""
    mask = np.zeros((len(positions) * runs, n), dtype=bool)
    for b, pos in enumerate(positions):
        if pos >= 0:
            mask[b * runs:(b + 1) * runs, pos] = True
    return mask


# ---- stage workers (top level so they pickle) ----

def node_task(key: str, chunk: list[int], runs: int) -> pd.DataFrame:
    beta, gamma = SETTINGS[MAIN]
    ids = data(key)["nodes"].index
    n = len(ids)
    meta = dict(dataset=key, version="real", beta=beta, gamma=gamma, setting=MAIN)
    out = []
    eps = epidemic(key, "real", beta, gamma, seeds=one_hot(n, runs, chunk), runs=runs, blocks=len(chunk))
    for pos, ep in zip(chunk, eps.split(len(chunk))):
        out.append(rows(ep, seed=int(ids[pos]), role="seed", node=int(ids[pos]), **meta))
    r = BLOCK_RUNS_FACTOR * runs
    eps = epidemic(key, "real", beta, gamma, immunised=one_hot(n, r, [-1] + chunk), runs=r,
                   blocks=len(chunk) + 1).split(len(chunk) + 1)
    base = (eps[0].inf != sir.NEVER) & (eps[0].inf != sir.IMMUNE)
    for pos, ep in zip(chunk, eps[1:]):
        for role, value in [("baseline", (base.sum(1) - base[:, pos]) / n), ("blocked", ep.final_size())]:
            out.append(pd.DataFrame({"run": np.arange(r), "outcome": "others_infected", "value": value,
                                     "seed": -1, "role": role, "node": int(ids[pos]), **meta}))
    return pd.concat(out)


def grid_task(setting: str, key: str, version: str, strategies: list[str], n_seeds: int, runs: int):
    beta, gamma = SETTINGS[setting]
    d = data(key)
    sc = scores(key, setting)
    n = len(d["nodes"])
    cells = [(st, f) for st in strategies for f in ([0.0] if st == "none" else BUDGETS)]
    masks = [im.immunisation_masks(st if f else "random", round(f * n), sc, d["matrix"], runs, RNG) for st, f in cells]
    eps = epidemic(key, version, beta, gamma, seeds=n_seeds, immunised=np.vstack(masks), runs=runs,
                   blocks=len(cells)).split(len(cells))
    meta = dict(dataset=key, version=version, beta=beta, gamma=gamma, seed=-1, setting=setting, n_seeds=n_seeds)
    out, pruned, curves = [], [], []
    for (st, f), mask, ep in zip(cells, masks, eps):
        out.append(rows(ep, strategy=st, budget=f, **meta))
        if version == "real" and n_seeds == 1 and f in (0.0, 0.10):
            grid, prev = ep.prevalence()
            curves.append(pd.DataFrame({"t_h": grid / sir.WINDOWS_PER_HOUR, "prevalence": prev.mean(axis=0),
                                        "dataset": key, "setting": setting, "strategy": st, "budget": f}))
        if version == "real" and n_seeds == 1 and f > 0:
            draws = 5 if st in im.RANDOMISED else 1
            m = pd.DataFrame([im.pruned_metrics(d["matrix"], mask[r]) for r in range(draws)]).mean()
            pruned.append({"dataset": key, "setting": setting, "strategy": st, "budget": f, **m.to_dict()})
    return pd.concat(out), pruned, curves


def seedset_task(key: str, runs: int) -> pd.DataFrame:
    beta, gamma = SETTINGS[MAIN]
    d = data(key)
    masks = [im.immunisation_masks(st, SEEDSET_K, scores(key, MAIN), d["matrix"], runs, RNG) for st in im.STRATEGIES]
    eps = epidemic(key, "real", beta, gamma, seeds=np.vstack(masks), runs=runs, blocks=len(masks))
    return pd.concat(rows(ep, dataset=key, version="real", beta=beta, gamma=gamma, seed=-1, setting=MAIN,
                          strategy=st, budget=0.0) for st, ep in zip(im.STRATEGIES, eps.split(len(masks))))


def node_table(key: str) -> pd.DataFrame:
    """Every node property used in RQ2 next to its role as seed and as blocker."""
    d = data(key)
    sc = scores(key, MAIN)
    cent = static_metrics.centralities(views.aggregate(d["contacts"], d["nodes"])).reindex(sc.index)
    props = im.node_temporal_properties(d["contacts"], d["nodes"].index)
    nd = pd.read_parquet(cache.result_path("immunization_nodes"))
    nd = nd[(nd["dataset"] == key) & (nd["role"] == "seed") & (nd["outcome"] == "final_size")]
    table = pd.concat([d["nodes"][["group"]], sc, cent[["kshell", "closeness"]], props], axis=1)
    table["seed_influence"] = nd.groupby("node")["value"].mean()
    table["blocking_importance"] = sc["oracle"]
    return table


def run_parallel(fn, tasks, workers):
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, *t) for t in tasks]
        for i, fut in enumerate(futures, 1):
            yield fut.result()
            print(f"    {i}/{len(futures)} tasks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--stages", nargs="+", default=["nodes", "grid", "seeds5", "seedsets"])
    ap.add_argument("--quick", action="store_true", help="hospital only, 20 runs, main setting")
    a = ap.parse_args()
    if a.quick:
        a.datasets, a.settings, a.runs = ["hospital"], [MAIN], 20
    for key in a.datasets:  # build shuffles and scores once, before the workers fork
        data(key)
    t0 = time.time()

    if "nodes" in a.stages:
        print("stage nodes: every node as single seed and as single immunised node", flush=True)
        tasks = []
        for key in a.datasets:
            pos = list(range(len(data(key)["nodes"])))
            tasks += [(key, pos[i:i + NODE_CHUNK], a.runs) for i in range(0, len(pos), NODE_CHUNK)]
        tasks.sort(key=lambda t: -len(data(t[0])["contacts"]))  # largest datasets first
        cache.save_result(pd.concat(run_parallel(node_task, tasks, a.workers)), "immunization_nodes")
        scores.cache_clear()
        for key in a.datasets:
            cache.save_table(node_table(key), f"immunization_node_table_{key}")
    for key in a.datasets:
        for s in a.settings:
            scores(key, s)

    strategies = ["none"] + im.STRATEGIES
    chunks = [strategies[i:i + 4] for i in range(0, len(strategies), 4)]
    if "grid" in a.stages:
        print("stage grid: setting x dataset x version x strategy x budget", flush=True)
        tasks = [(s, k, v, ch, 1, a.runs) for s in a.settings for k in a.datasets for v in VERSIONS for ch in chunks]
        tasks.sort(key=lambda t: (t[0] != "zhang", t[2] == "aggregated", -len(data(t[1])["contacts"])))
        res = list(run_parallel(grid_task, tasks, a.workers))
        cache.save_result(pd.concat(r[0] for r in res), "immunization_grid")
        cache.save_table(pd.DataFrame([p for r in res for p in r[1]]), "immunization_pruned")
        pd.concat([c for r in res for c in r[2]]).to_parquet(cache.result_path("immunization_curves"))

    if "seeds5" in a.stages:
        print("stage seeds5: 5 random seeds", flush=True)
        tasks = [(MAIN, k, "real", ch, 5, a.runs) for k in a.datasets for ch in chunks]
        cache.save_result(pd.concat(r[0] for r in run_parallel(grid_task, tasks, a.workers)), "immunization_seeds5")

    if "seedsets" in a.stages:
        print("stage seedsets: immunisation sets used as seed sets", flush=True)
        tasks = [(k, a.runs) for k in a.datasets]
        cache.save_result(pd.concat(run_parallel(seedset_task, tasks, a.workers)), "immunization_seedsets")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
