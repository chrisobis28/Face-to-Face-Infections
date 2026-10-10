# Face-to-Face-Infections

Using SocioPatterns face-to-face contact networks (hospital ward, primary school, high school) we ask:
which nodes and which contacts matter most for contagion, and can a cheap intervention be designed
from topology alone?

Final project for *Modelling and Data Analysis in Complex Networks* (DSAIT4310), TU Delft, October 2026.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `import f2f` work everywhere
bash scripts/download_data.sh   # ~1 MB, from the Netzschleuder mirror of SocioPatterns
python scripts/check_data.py    # stream 1 acceptance check (checksums, day table)
```

## Data contract (agreed at kick-off, do not change without telling everyone)

`f2f.datasets.load_contacts(key)` returns `(contacts, nodes)` for `key` in
`hospital`, `primary_school`, `high_school`:

| table | columns |
| --- | --- |
| `contacts` | `t` (int, seconds since recording start, multiple of 20), `i`, `j` (int node ids, `i < j`), `abs_time` (Europe/Paris), `day` (0-based calendar day) |
| `nodes` | index `node`; `group` (ward role or class), plus `gender` where available |

One row = one 20-second window in which `i` and `j` were in proximity.

Network versions are named strings used everywhere: `real`, `time_shuffled`,
`episode_time_shuffled`, `link_shuffled`, `aggregated`, `surrogate`. Simulation results go to
`results/<experiment>.parquet` through
`f2f.cache.save_result` with the row format `dataset, version, beta, gamma, seed, run, outcome, value`.

Progress per stream and per task: see [PLAN.md](PLAN.md).
Final Stream 3 numbers and their interpretation: see [STREAM3_RESULTS.md](STREAM3_RESULTS.md).

## Layout

```
src/f2f/
  datasets.py        stream 1  loader, checksums, dataset registry and citations
  views.py           stream 1  snapshots, aggregated weighted graph, per-day splits
  cache.py           stream 1  result cache row format
  static_metrics.py  stream 2  lecture-1 metrics, degree fits, centralities
  nulls.py           stream 2  configuration model, degree-preserving rewiring, z-scores
  temporal_metrics.py stream 3 contact episodes, burstiness, activity and memory
  temporal_nulls.py  stream 3  invariant-preserving time and link shuffles
  prediction.py      stream 3  self-driven temporal-link prediction
scripts/
  download_data.sh, check_data.py, run_static_characterisation.py,
  run_temporal_characterisation.py
results/             generated tables (git-ignored), figures/ tracked
```

## Datasets 
| key | nodes | contacts | days | group | cite |
| --- | --- | --- | --- | --- | --- |
| hospital | 75 | 32,424 | 5 (Mon 13:00 to Fri 14:00) | NUR / PAT / MED / ADM | Vanhems et al. 2013, PLoS ONE 8(9) e73970 |
| primary_school | 242 | 125,773 | 2 | class (10) + Teachers | Stehlé et al. 2011, PLoS ONE 6(8) e23176 |
| high_school | 329 | 188,508 | 5 | class (9) | Mastrandrea et al. 2015, PLoS ONE 10(9) e0136497 |

Data: https://sociopatterns.org/datasets.html, mirror https://networks.skewed.de/.

## Stream 2: static characterisation (RQ1, static half)

```bash
python scripts/run_static_characterisation.py --datasets hospital --n-null 20   # quick, ~20 s
python scripts/run_static_characterisation.py --n-null 100                      # all, ~5 min
```

Outputs `results/static_metrics_<ds>.csv`, `results/centralities_<ds>.csv` (used by streams 4 and 6),
`results/zscores_<kind>_<ds>.csv` and `figures/rq1_static_vs_null_<kind>.png`.

Two null models are provided. **Use `rewired` (double-edge swaps) for z-scores**: it keeps every
degree exactly. The `configuration` model (stub matching, then self-loops and duplicates removed, as
in lecture 2) loses 10 to 40 % of the links on these dense graphs, which biases clustering and
hopcount; its `L` z-score shows the loss. It is kept for comparison with the lecture.

## Stream 3: temporal characterisation and prediction (RQ1, temporal half)

```bash
python scripts/run_temporal_characterisation.py --datasets hospital --n-null 3  # quick check
python scripts/run_temporal_characterisation.py --n-null 20                    # full run, ~13 min
python -m unittest discover -s tests                                          # acceptance tests
```

The time shuffle preserves every pair's number of 20-second contacts and every timestamp's contact
count, but randomises their ordering. The episode-time shuffle instead moves intact meetings across
active periods, preserving each pair and meeting duration but not exact per-timestamp volume. The
link shuffle preserves every timestamp's contact count and every node's total contact involvement,
but randomises partners. All nulls reject self-loops and duplicate pair-windows. Memory is measured
in 5-minute bins within each calendar day, so nights do not create artificially long gaps. Because
contacts are undirected, activity potential is a node's share of all contact involvements, not a
fictitious share of "initiated" contacts.

The self-driven predictor follows [Zou, Ceria and Wang
(2023)](https://doi.org/10.1007/s41109-023-00597-w): recent binary link activity is weighted by
exponential memory. We adapt its next-snapshot scores to rank contacts for each later day. The decay
parameter is selected on day 0 and compared with an unweighted frequency baseline over the same
history window. Only links observed before the predicted day are candidates. Results include
candidate coverage so unseen future links are not silently ignored.
