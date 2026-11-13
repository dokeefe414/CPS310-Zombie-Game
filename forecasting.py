"""Online sliding-window adaptive Kalman forecasting for population counts.

State is [population, change_per_tick]. The constant-velocity transition
provides a real h-step forecast; rolling innovation variance adapts to shifts.
Only past and present observations are used. No GUI/database dependency.
"""
from collections import deque
import math
import numpy as np


class SlidingWindowAdaptiveKalman:
    """Two-state adaptive Kalman filter with bounded, nonnegative forecasts.

    window: number of *past innovations* used to estimate observation noise.
    process_variance: per-tick uncertainty in population acceleration.
    measurement_variance: initial observation variance, in agents squared.
    capacity: optional total agent population bound.
    """

    def __init__(self, window=50, process_variance=4.0,
                 measurement_variance=25.0, capacity=None):
        if not isinstance(window, int) or window < 3:
            raise ValueError("window must be an integer >= 3")
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("variances must be positive")
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        self.window = window
        self.process_variance = float(process_variance)
        self.initial_measurement_variance = float(measurement_variance)
        self.capacity = capacity
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([1., 0.])
        self.Q = self.process_variance * np.array([[0.25, 0.5], [0.5, 1.]])
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None
        self.innovations = deque(maxlen=self.window)
        self.changes = deque(maxlen=self.window)
        self.last_observation = None
        self.measurement_variance = self.initial_measurement_variance
        self.observations = 0

    def update(self, population):
        """Consume exactly one new tick, returning the filtered population."""
        value = float(population)
        if not math.isfinite(value) or value < 0:
            raise ValueError("population must be finite and nonnegative")
        if self.state is None:
            self.state = np.array([value, 0.], dtype=float)
            self.covariance = np.diag([self.initial_measurement_variance, 100.])
            self.observations = 1
            self.last_observation = value
            return value

        self.changes.append(value - self.last_observation)
        self.last_observation = value
        predicted = self.F @ self.state
        P = self.F @ self.covariance @ self.F.T + self.Q
        innovation = value - float(self.H @ predicted)
        S = float(self.H @ P @ self.H + self.measurement_variance)
        K = (P @ self.H) / S
        self.state = predicted + K * innovation
        I_KH = np.eye(2) - np.outer(K, self.H)
        self.covariance = (I_KH @ P @ I_KH.T +
                           np.outer(K, K) * self.measurement_variance)
        self.innovations.append(innovation)
        if len(self.innovations) >= 3:
            # Estimate recent innovation variance; subtract predicted state
            # uncertainty so observation noise is not inflated by the trend.
            sample_var = float(np.var(self.innovations, ddof=1))
            state_uncertainty = float(self.H @ P @ self.H)
            estimate = max(1.0, sample_var - state_uncertainty)
            self.measurement_variance = max(
                1.0, 0.8 * self.measurement_variance + 0.2 * estimate
            )
        self.observations += 1
        return float(self.state[0])

    def predict(self, horizon=10):
        """Forecast the population at each of the next horizon ticks."""
        if self.state is None:
            raise ValueError("at least one observation is required")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        # The recent *observed* differences determine the local slope.
        # This makes the window a real forecasting hyperparameter, rather
        # than only a noise-estimation hyperparameter.
        if self.changes:
            recent = np.asarray(self.changes, dtype=float)
            local_slope = float(np.mean(recent))
            standard_error = (float(np.std(recent, ddof=1)) /
                              math.sqrt(len(recent))) if len(recent) > 1 else abs(local_slope)
            # Shrink weak/noisy trends toward the persistence baseline.
            confidence = abs(local_slope) / (abs(local_slope) + standard_error + 0.5)
            trend = confidence * (0.75 * local_slope + 0.25 * float(self.state[1]))
        else:
            trend = 0.0
        values = []
        for h in range(1, horizon + 1):
            # Saturating extrapolation prevents implausible linear divergence
            # on long horizons without using future observations.
            effective_horizon = 12.0 * (1.0 - math.exp(-h / 12.0))
            prediction = float(self.last_observation + effective_horizon * trend)
            if self.capacity is not None:
                prediction = min(self.capacity, prediction)
            values.append(max(0.0, prediction))
        return values


class ForecastTracker:
    """Online h-step evaluation without look-ahead leakage."""

    def __init__(self, horizon=10):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.pending = {}
        self.errors = []
        self.baseline_errors = []

    def record(self, tick, actual, forecast, naive):
        """Resolve matured predictions, then register today's future forecast."""
        if tick in self.pending:
            predicted, baseline = self.pending.pop(tick)
            self.errors.append(float(actual) - predicted)
            self.baseline_errors.append(float(actual) - baseline)
        self.pending[tick + self.horizon] = (float(forecast), float(naive))

    def metrics(self):
        if not self.errors:
            return {"n": 0, "mae": None, "rmse": None,
                    "naive_mae": None, "naive_rmse": None}
        e = np.asarray(self.errors)
        b = np.asarray(self.baseline_errors)
        return {"n": len(e), "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "naive_mae": float(np.mean(np.abs(b))),
                "naive_rmse": float(np.sqrt(np.mean(b ** 2)))}
