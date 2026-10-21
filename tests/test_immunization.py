import unittest

import numpy as np
import pandas as pd

from f2f import immunization as im, sir


def chain(times):
    """Contacts 0-1 then 1-2 at the given times (seconds), plus a 2-3 contact much later."""
    rows = [(times[0], 0, 1, 0), (times[1], 1, 2, 0), (400, 2, 3, 0), (420, 0, 3, 0)]
    return pd.DataFrame(rows, columns=["t", "i", "j", "day"]).sort_values("t").reset_index(drop=True)


NODES = pd.DataFrame({"group": ["a", "a", "b", "b"]}, index=pd.Index([0, 1, 2, 3], name="node"))


class SirTests(unittest.TestCase):
    def test_infection_follows_time_respecting_paths(self):
        ok = sir.simulate(chain((0, 20)), NODES, seeds=[0], beta=1.0, gamma=1e-9, runs=3, max_loops=1)
        self.assertEqual(ok.inf[0, :3].tolist(), [-1, 0, 1])  # one hop per 20-s window
        late = sir.simulate(chain((20, 0)), NODES, seeds=[0], beta=1.0, gamma=1e-9, runs=3, max_loops=1)
        self.assertTrue((late.inf[:, 2] == sir.NEVER).all())  # 1-2 happened before 0-1

    def test_immunised_nodes_are_never_infected(self):
        for simulate in (sir.simulate, sir.simulate_aggregated):
            ep = simulate(chain((0, 20)), NODES, seeds=[0], beta=1.0, gamma=1e-9, immunised=[1], runs=5, max_loops=3)
            self.assertTrue((ep.inf[:, 1] == sir.IMMUNE).all())
            self.assertTrue((ep.final_size() <= 0.75).all())

    def test_empty_immunisation_reproduces_baseline_exactly(self):
        c = chain((0, 20))
        base = sir.simulate(c, NODES, seeds=1, beta=0.5, gamma=0.05, runs=50, rng_seed=7, max_loops=5)
        none = sir.simulate(c, NODES, seeds=1, beta=0.5, gamma=0.05, immunised=np.zeros((50, 4), bool),
                            runs=50, rng_seed=7, max_loops=5)
        np.testing.assert_array_equal(base.inf, none.inf)


class StrategyTests(unittest.TestCase):
    def test_every_strategy_returns_k_distinct_nodes(self):
        c = chain((0, 20))
        scores = im.node_scores(c, NODES, beta=0.1, gamma=0.01)
        scores["oracle"] = [0.3, 0.2, 0.1, 0.0]
        matrix = sir.contact_matrix(c, NODES.index)
        for strategy in im.STRATEGIES:
            mask = im.immunisation_masks(strategy, 2, scores, matrix, runs=10, rng_seed=1)
            self.assertTrue((mask.sum(axis=1) == 2).all(), strategy)
        self.assertEqual(np.flatnonzero(im.immunisation_masks("oracle", 2, scores, matrix, 1)[0]).tolist(), [0, 1])

    def test_temporal_reach_respects_time(self):
        reach = im.temporal_reach(chain((0, 20)), NODES.index)
        self.assertAlmostEqual(reach[0], 1.0)  # 0 -> 1 -> 2 -> 3
        reach = im.temporal_reach(chain((20, 0)), NODES.index)
        self.assertAlmostEqual(reach[0], 2 / 3)  # 0 -> 1 and 0 -> 3; 1-2 and 2-3 came too early


if __name__ == "__main__":
    unittest.main()
