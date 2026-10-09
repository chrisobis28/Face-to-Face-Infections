# Implementation plan and progress

Six streams, one per teammate. Four start on day one with no dependency; the loader (stream 1)
and the simulation engine (stream 4) are the two sequential bottlenecks, and the streams that wait
on them build their own pieces in the meantime. Everyone works on the
hospital first, then the primary school, then the high school.

Legend: ✅ done in this MR · ⬜ open. Tick items as PRs land.

Shared contracts (agreed on day one, see README):
- ✅ Contact table: `t, i, j, abs_time, day`; node table with `group`
- ✅ Network version names: `real`, `time_shuffled`, `link_shuffled`, `aggregated`, `surrogate`
- ✅ Result cache row: `dataset, version, beta, gamma, seed, run, outcome, value` (`f2f.cache`)
- ⬜ Simulator signature: `simulate(contacts, seeds, beta, gamma, immunised, removed_contacts, runs, rng_seed)` (stream 4 to implement)
- ⬜ Figure style file under `figures/`

---

## Stream 1: Data and infrastructure

Needs: nothing. Delivers to: everyone.

- ✅ Repo structure: `src/f2f` package, `scripts/`, `data/`, `results/`, `figures/`, `pyproject.toml`, `requirements.txt`, `.gitignore`
- ✅ Download script for the three CSV archives (`scripts/download_data.sh`)
- ✅ Loader into the agreed contact table for hospital, primary school, high school (`f2f.datasets.load_contacts`)
- ✅ Checksums against published node and contact counts; raises on mismatch
- ✅ Dataset registry with time origin, group column, citation and URL per dataset
- ✅ Snapshot sequence at any multiple of 20 s (`f2f.views.to_snapshots`)
- ✅ Aggregated weighted graph, weight = contact seconds, count = windows (`f2f.views.aggregate`)
- ✅ Per-day splits that never overlap (`f2f.views.split_days`, `day_table`)
- ✅ Result cache with schema validation (`f2f.cache.save_result`, `load_result`, `save_table`)
- ✅ Acceptance script (`scripts/check_data.py`): all three datasets pass
- ✅ Shared matplotlib style file (`figures/f2f.mplstyle`)
- ✅ README with setup, data contract and citations
- ⬜ Sensitivity harness over beta, gamma and bin size (week 2, needs the engine from stream 4)

Known assumption: the primary-school file counts seconds from midnight without a date; 1-2 Oct 2009 is assumed. The day split is correct either way.

## Stream 2: Static characterisation and static null models (RQ1, static half)

Needs: loader. Delivers to: centrality table and correlations to streams 4 and 6; spectral radius to stream 4.

- ✅ Lecture-1 metrics on the aggregated graph: N, L, density, degree mean / variance / dispersion, clustering, transitivity, giant component, mean hopcount, diameter, assortativity (`f2f.static_metrics.network_metrics`)
- ✅ Modularity of the metadata partition (class / ward role) and of the Louvain optimum
- ✅ Spectral radius λ1 and the SIS threshold 1/λ1; algebraic connectivity
- ✅ Degree-distribution fits: power law vs exponential on the pmf and on the CCDF, plus Var/E[D] against the ER expectation (`fit_degree_distribution`)
- ✅ Strength-distribution fits (contact minutes per node), same method (`results/strength_fit_<ds>.csv`, `figures/strength_distribution_<ds>.png`)
- ✅ Centrality table per node: degree, strength, betweenness, closeness, eigenvector (unweighted and weighted), k-shell (`centralities`)
- ✅ Spearman correlation matrix between centralities (`results/centrality_correlations_<ds>.csv`)
- ✅ Configuration model (stub matching, loops and duplicates removed) (`f2f.nulls.configuration_model`)
- ✅ Degree-preserving rewiring with exact degrees (`degree_preserving_rewiring`)
- ✅ Rewiring toward a target assortativity and the lecture-1 assortativity-range figure (`rewire_to_assortativity`, `figures/assortativity_range.png`)
- ✅ Null ensembles, z-scores with empirical p-values and effect sizes (`null_ensemble`, `zscores`)
- ✅ Run script for all datasets, 100 realisations per null (`scripts/run_static_characterisation.py`)
- ✅ Outputs: `results/static_metrics_*.csv`, `degree_fit_*.csv`, `centralities_*.csv`, `null_*`, `zscores_*`, `assortativity_range.csv`
- ✅ Figures: `figures/rq1_static_vs_null_{rewired,configuration}.png` (2 x 4 panels, effect-size labels), `degree_distribution_*.png`, `assortativity_range.png`
- ✅ Documented: configuration model loses 10 to 40 % of links on these dense graphs; the rewired null is the default for z-scores
- ⬜ Per-day aggregated graphs and metrics (day-to-day stability, useful for stream 5's calibration story)

## Stream 3: Temporal characterisation, temporal null models and prediction (RQ1, temporal half)

Needs: loader. Delivers to: shuffles and activity potentials to stream 5; predicted contacts to stream 6.

- ⬜ Inter-contact time distribution and burstiness per dataset
- ⬜ Contact duration distribution
- ⬜ Activity potential per node (share of initiated contacts) and its distribution
- ⬜ Jaccard similarity and link-activity autocorrelation as a function of lag (memory decay)
- ⬜ Time-shuffle null (permute timestamps, keeps the aggregated graph) in the contact-table format
- ⬜ Link-shuffle null in the contact-table format
- ⬜ Temporal metrics on the shuffled sequences, z-scores
- ⬜ Self-driven memory model (lecture 3) predicting next-day contacts from previous days
- ⬜ Prediction-quality evaluation with a baseline, within the school week
- ⬜ Predicted next-day contact list written to the cache for stream 6
- ⬜ RQ1 temporal figure

## Stream 4: Spreading engine and node roles (RQ2)

Needs: loader; centralities and λ1 from stream 2. Delivers to: engine to streams 5 and 6 by day 5.

- ⬜ Vectorised discrete-time SIR on the contact table with the agreed signature, fixed seeds
- ⬜ Periodic looping of the recording when the outbreak outlasts it
- ⬜ Test against the analytical SIR on a complete graph
- ⬜ SIS on the aggregated weighted graph; empirical threshold vs 1/λ1
- ⬜ All-seeds sweep: every seed node, 200 runs (100 on the high school), several beta values
- ⬜ Role measures per node: final size as seed, time to 10 % prevalence, probability of being infected
- ⬜ Spearman correlation and top-10 % overlap of each role measure with each centrality
- ⬜ Correlation heat map per dataset and per beta (RQ2 figure)
- ⬜ Results in the cache row format

## Stream 5: Calibrated surrogate and four-way comparison (RQ3, RQ4a)

Needs: loader; activity potentials and shuffles from stream 3; engine from stream 4. Delivers to: surrogate to stream 6.

- ⬜ Activity-driven network generator in the contact-table format
- ⬜ Calibration by least squares over m and η to the observed degree distribution and daily contact volume on calibration days
- ⬜ Least-squares surface figure
- ⬜ Validation on held-out days: degree distribution, contact volume, SIR outbreak
- ⬜ Same SIR on real, time-shuffled, aggregated and surrogate; final size vs beta (RQ3 figure)
- ⬜ Validated surrogate contact sequence written for stream 6

## Stream 6: Interventions (RQ4b)

Needs: engine from stream 4; centralities from stream 2; predictions from stream 3; surrogate from stream 5.

- ⬜ Intervention operators: immunise a node set, cohorting, shorten the day, isolate after first detected case with delay
- ⬜ Rankings at 5 / 10 / 20 % budgets from: full history, static aggregate, first 1-2 days, predicted contacts, random
- ⬜ Equal-cost comparison on the real temporal network with 95 % intervals
- ⬜ Same comparison on the calibrated surrogate, including what-ifs the recording lacks (act a day earlier)
- ⬜ RQ4 figure and the one-sentence practical message

---

## Milestones

| Dates | Milestone | Status |
| --- | --- | --- |
| 9-10 Oct | Kick-off and contracts | ✅ contracts in README; names still to assign |
| 11-13 Oct | Feedback-meeting results: loader, RQ1 static tables, engine prototype, calibration prototype | ✅ loader and RQ1 static; ⬜ engine and calibration prototypes |
| 13 Oct | Feedback deck (5-10 min) | ⬜ |
| 12-16 Oct | Feedback meeting; RQs frozen afterwards | ⬜ |
| 16-22 Oct | Full runs on all datasets | ⬜ |
| 22 Oct | Checkpoint: prediction-based immunisation in or out; extra dataset or not | ⬜ |
| 22-26 Oct | Surrogate interventions and sensitivity | ⬜ |
| 26 Oct | Figure freeze | ⬜ |
| 26-30 Oct | Slide deck and two timed rehearsals | ⬜ |
| before the interview | Slides on Brightspace, repo tagged | ⬜ |
| 26 Oct - 2 Nov | Final interview | ⬜ |

## First findings (stream 2, for the feedback deck)

- Schools: class structure far beyond chance (modularity z ≈ 74 primary, ≈ 151 high school against the rewired null).
- Hospital: ward roles are not communities; role modularity is negative and degree assortativity is negative (nurses bridge patients).
- Degree distributions: exponential tail on the CCDF, Var/E[D] between 5 and 10, so neither scale-free nor Erdős-Rényi.
- Strength (contact minutes) is far more heterogeneous than degree: max/mean 2.5 to 5 and an exponential tail with R² 0.92 to 0.98, still not a power law.
- Heterogeneity lives in contact time, not contact count: the busiest person has 2 to 2.5x the mean degree but 2.5 to 5x the mean strength. Immunisation rankings should use strength or weighted centralities rather than plain degree (prediction for RQ2 and RQ4).
- Tiny worlds: mean hopcount 1.6 to 2.2, diameter 3 to 4. SIS thresholds 1/λ1: 0.027 (hospital), 0.012 (primary), 0.024 (high school).
