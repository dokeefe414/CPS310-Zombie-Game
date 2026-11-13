"""Run: py -m unittest discover -s tests -v"""
import unittest
from experiment_runner import evaluate_series, evaluate_shift_series, run_simulation


class ExperimentTests(unittest.TestCase):
    def test_metrics_have_matured_targets(self):
        series = list(range(100, 130))
        results = evaluate_series(series, windows=(5, 10), horizons=(2, 4))
        self.assertEqual(len(results), 4)
        self.assertEqual([r["n"] for r in results], [28, 26, 28, 26])
        self.assertTrue(all(r["mae"] >= 0 for r in results))

    def test_small_headless_run(self):
        history, elapsed = run_simulation(agents=30, ticks=3, seed=7)
        self.assertEqual(history[0]["tick"], 0)
        self.assertEqual(history[0]["humans"] + history[0]["zombies"], 30)
        self.assertGreaterEqual(elapsed, 0)
        self.assertLessEqual(len(history), 4)

    def test_shift_phase_boundaries(self):
        counts = [100 + t for t in range(40)]
        rows = evaluate_shift_series(counts, shift_tick=20,
                                     windows=(5,), horizons=(5,))
        pre, post = rows
        self.assertEqual(pre["phase"], "pre")
        self.assertEqual(post["phase"], "post")
        self.assertEqual(pre["n"], 15)
        self.assertEqual(post["n"], 15)

    def test_intervention_is_recorded(self):
        history, _ = run_simulation(agents=30, ticks=3, seed=7,
                                    shift_tick=2)
        self.assertEqual(history[0]["strategy"], "thompson")
        self.assertEqual(history[1]["strategy"], "thompson")
        self.assertEqual(history[2]["strategy"], "rush_only")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            evaluate_series([])


if __name__ == "__main__":
    unittest.main()
