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
        self.assertGreater(model.predict(10)[-1], 100 + 4 * 79)
        self.assertLess(model.predict(10)[-1], 100 + 4 * 89)

    def test_windows_change_forecast_after_shift(self):
        # Old positive trend followed by a recent negative trend.
        series = [100 + 3 * t for t in range(100)]
        series += [series[-1] - 2 * (t + 1) for t in range(12)]
        short = SlidingWindowAdaptiveKalman(window=5)
        long = SlidingWindowAdaptiveKalman(window=100)
        for count in series:
            short.update(count)
            long.update(count)
        self.assertLess(short.predict(10)[-1], long.predict(10)[-1])

    def test_flat_series_stays_near_persistence(self):
        model = SlidingWindowAdaptiveKalman(window=20)
        for _ in range(80):
            model.update(200)
        self.assertAlmostEqual(model.predict(50)[-1], 200, delta=1)

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
