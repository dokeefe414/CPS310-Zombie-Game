"""Run: py -m unittest discover -s tests -v"""
import unittest
from experiment_runner import evaluate_series, run_simulation


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

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            evaluate_series([])


if __name__ == "__main__":
    unittest.main()
