import unittest

import numpy as np
import pandas as pd

from f2f import prediction, temporal_metrics as tm, temporal_nulls as tn


def small_contacts():
    rows = [
        (0, 0, 1, 0), (20, 0, 1, 0), (60, 0, 1, 0),
        (0, 2, 3, 0), (40, 1, 2, 0), (60, 2, 3, 0),
        (86400, 0, 2, 1), (86420, 0, 2, 1), (86440, 1, 3, 1),
    ]
    frame = pd.DataFrame(rows, columns=["t", "i", "j", "day"])
    frame["abs_time"] = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(frame["t"], unit="s")
    return frame.sort_values(["t", "i", "j"]).reset_index(drop=True)


class TemporalMetricTests(unittest.TestCase):
    def test_contact_windows_are_merged_into_episodes(self):
        episodes = tm.contact_episodes(small_contacts())
        first = episodes[(episodes["i"] == 0) & (episodes["j"] == 1)].sort_values("start_t")
        self.assertEqual(first["duration_s"].tolist(), [40, 20])
        self.assertEqual(tm.inter_contact_times(episodes).tolist(), [20, 40])

    def test_activity_potential_is_undirected_and_normalised(self):
        activity = tm.activity_potential(small_contacts(), node_ids=[0, 1, 2, 3])
        self.assertAlmostEqual(activity["activity_potential"].sum(), 1.0)
        self.assertEqual(activity["contact_involvements"].sum(), 2 * len(small_contacts()))

    def test_memory_curve_has_finite_lag_one_values(self):
        curve = tm.memory_curve(small_contacts(), max_lag=2, bin_s=20, node_ids=[0, 1, 2, 3])
        self.assertEqual(curve["lag"].tolist(), [1, 2])
        self.assertTrue(np.isfinite(curve.loc[0, "jaccard"]))
        self.assertTrue(np.isfinite(curve.loc[0, "autocorrelation"]))


class TemporalNullTests(unittest.TestCase):
    def test_time_shuffle_invariants(self):
        contacts = small_contacts()
        shuffled = tn.time_shuffle(contacts, seed=4, swaps_per_contact=20)
        tn.assert_time_shuffle_invariants(contacts, shuffled)

    def test_link_shuffle_invariants(self):
        contacts = small_contacts()
        shuffled = tn.link_shuffle(contacts, seed=5, swaps_per_contact=20)
        tn.assert_link_shuffle_invariants(contacts, shuffled)


class PredictionTests(unittest.TestCase):
    def test_self_driven_weights_recent_activity_more(self):
        history = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        scores = prediction.self_driven_scores(history, tau=1.0)
        self.assertGreater(scores[1], scores[0])

    def test_average_precision(self):
        self.assertAlmostEqual(
            prediction.average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])),
            (1 + 2 / 3) / 2,
        )
        self.assertAlmostEqual(
            prediction.average_precision(np.array([1, 0]), np.array([0.0, 0.0])),
            0.5,
        )

    def test_next_day_prediction_never_uses_future_links_as_candidates(self):
        evaluation, predicted, _ = prediction.evaluate_self_driven(
            small_contacts(), bin_s=20, lookback=2, taus=(0.1, 1.0)
        )
        self.assertEqual(evaluation["day"].unique().tolist(), [1])
        predicted_edges = set(zip(predicted["i"], predicted["j"]))
        self.assertNotIn((0, 2), predicted_edges)


if __name__ == "__main__":
    unittest.main()
