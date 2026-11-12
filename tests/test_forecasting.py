"""Run with: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from forecasting import SlidingWindowAdaptiveKalman, ForecastTracker


class ForecastTests(unittest.TestCase):
    def test_initialization_and_horizon(self):
        model = SlidingWindowAdaptiveKalman(window=5, capacity=1000)
        model.update(200)
        self.assertEqual(model.predict(3), [200., 200., 200.])

    def test_linear_growth_forecast(self):
        model = SlidingWindowAdaptiveKalman(window=10)
        for i in range(80):
            model.update(100 + 4 * i)
        self.assertAlmostEqual(model.predict(10)[-1], 100 + 4 * 89, delta=10)

    def test_bounds(self):
        model = SlidingWindowAdaptiveKalman(window=5, capacity=100)
        for value in [90, 95, 100, 100, 100]:
            model.update(value)
        self.assertTrue(all(0 <= p <= 100 for p in model.predict(50)))

    def test_shift_and_rolling_window(self):
        model = SlidingWindowAdaptiveKalman(window=5)
        for value in list(range(100, 140)) + list(range(140, 100, -1)):
            model.update(value)
        self.assertLess(model.state[1], 0)
        self.assertEqual(len(model.innovations), 5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            SlidingWindowAdaptiveKalman(window=2)
        with self.assertRaises(ValueError):
            SlidingWindowAdaptiveKalman().predict()
        with self.assertRaises(ValueError):
            SlidingWindowAdaptiveKalman().update(float("nan"))

    def test_no_lookahead_metrics(self):
        tracker = ForecastTracker(horizon=2)
        tracker.record(0, 10, 12, 10)
        tracker.record(1, 11, 13, 11)
        self.assertEqual(tracker.metrics()["n"], 0)
        tracker.record(2, 12, 14, 12)
        self.assertEqual(tracker.metrics()["n"], 1)
        self.assertEqual(tracker.metrics()["mae"], 0)
        self.assertEqual(tracker.metrics()["naive_mae"], 2)


if __name__ == "__main__":
    unittest.main()
