# Forecasting Mathematics, Evaluation, and Experimental Interpretation

This document derives the **equations actually implemented** in
`forecasting.py` and `experiment_runner.py`. It distinguishes the Kalman
filter's state estimate from the **heuristic** forecast delivered to the GUI.

Use alongside [Forecasting overview](FORECASTING.md),
[Architecture](ARCHITECTURE.md), and [Game Theory](SWARM_GAME_THEORY.md).

## 1. Observations and notation

At tick `t`, let `y_t` be the actual **number of surviving zombies**
after a simulation step. The two-dimensional latent filter state is

```text
x_t = [estimated_zombies_t, estimated_change_per_tick_t]^T
```

The model uses only `y_0, ..., y_t` when predicting `y_(t+h)`.
It does not directly see human locations, zombie tactics, scent, or the
future simulated counts.

Default parameters (all are editable in `SlidingWindowAdaptiveKalman`):

| Symbol | Code | Default | Meaning |
| --- | --- | --- | --- |
| `w` | `window` | `50` | Maximum previous innovations and observed differences retained |
| `q` | `process_variance` | `4.0` | Scale of state-dynamics uncertainty |
| `R_0` | `measurement_variance` | `25.0` | Initial observation-noise estimate |
| `C` | `capacity` | `None` or initial total agents | Optional physical upper bound |
| `h` | `horizon` | `10` in GUI | How many future ticks to forecast |

Here `window` counts **past observations/innovations**, not an offline
training sequence.

## 2. Kalman prediction step

The code defines constant-velocity dynamics with a one-tick transition:

```text
F = [[1, 1],
     [0, 1]]

H = [1, 0]

Q = q * [[0.25, 0.5],
         [0.5,  1.0]]
```

Its one-step predicted state and covariance are

```text
x_minus_t = F @ x_(t-1)
P_minus_t = F @ P_(t-1) @ F.T + Q
```

`F` says population is locally approximated by the previous estimate
plus the estimated per-tick change, while the change itself persists
until new evidence arrives. `H` says the observable measurement is the
population component.

At first observation `y_0`, there is no prediction step: the state
is `[y_0, 0]^T`, initial covariance is
`diag([initial_measurement_variance, 100])`, and observation count is one.

## 3. Measurement update

For each new observed count `y_t`, the **innovation** is the surprise
relative to the prior estimate:

```text
innovation_t = y_t - H @ x_minus_t
S_t = H @ P_minus_t @ H.T + R_t
K_t = (P_minus_t @ H.T) / S_t
x_t = x_minus_t + K_t * innovation_t
```

The code updates covariance using a numerically safer
**Joseph-form covariance update**:

```text
J = I - K_t @ H
P_t = J @ P_minus_t @ J.T + K_t @ K_t.T * R_t
```

In actual NumPy, vectors have simple 1D shapes and outer products
are used instead of repeatedly allocating explicit column matrices.

The **Kalman gain** adjusts how much the measurement changes the
estimated state. Larger estimated measurement uncertainty generally
reduces the update's trust in a given observation, all else equal.

## 4. Sliding innovation variance

Each completed update appends the latest innovation to a
`deque(maxlen=window)`. After at least three innovations are recorded,
the code computes:

```text
sample_var = sample_variance(recent_innovations, ddof=1)
state_uncertainty = H @ P_minus_t @ H.T
noise_estimate = max(1, sample_var - state_uncertainty)
R_next = max(1, 0.8 * R_current + 0.2 * noise_estimate)
```

This is an **adaptive heuristic** for the next measurement-variance
estimate. It subtracts the current predicted state uncertainty and
applies a minimum value of one. It is not an external change-point
detector, a statistical hypothesis test, or a proof that process and
measurement noise are separately identifiable.

A longer window smooths noisy innovation estimates but can lag behind
changes. A shorter window can respond more quickly but be less stable.

## 5. Rolling trend estimation: the important forecasting change

The final `predict(h)` implementation does **not** simply return
`F^h @ x_t`. Instead it uses recent *observed* count differences:

```text
difference_i = y_i - y_(i-1)
s = mean(last w differences)
se = std(last w differences, ddof=1) / sqrt(number of differences)
confidence = abs(s) / (abs(s) + se + 0.5)
blended_slope = 0.75 * s + 0.25 * x_t[1]
effective_trend = confidence * blended_slope
```

For exactly one previous difference, the code uses
`se = abs(s)`; with none, it sets `effective_trend = 0`.
The `+0.5` denominator term deliberately shrinks small or noisy
slopes toward the **persistence baseline** (no change).

This is why the window length materially affects the produced forecast:
it controls *both* rolling innovation variance and the rolling observed
slope. Earlier versions adapted only measurement variance, yielding
similar predictions across windows.

## 6. Saturated future extrapolation

For future tick `t+h`, the code computes

```text
effective_horizon(h) = 12 * (1 - exp(-h / 12))
forecast(t+h) = y_t + effective_horizon(h) * effective_trend
forecast(t+h) = max(0, forecast(t+h))
if capacity is supplied:
    forecast(t+h) = min(capacity, forecast(t+h))
```

The function grows slowly as `h` becomes large and asymptotically
approaches 12, rather than increasing linearly without limit.
This prevents a short-lived trend from exploding at long horizons.

**Important terminology:** this is a **Kalman-informed, conservatively
extrapolated forecast**, not an exact optimal multi-step Kalman
prediction. It also makes a point prediction, not a calibrated
confidence interval or full predictive probability distribution.

### Numerical example (illustrative)

Suppose at tick `t` the observed count is 200, the local mean
difference is +2 zombies/tick, the estimated standard error is 1,
and the filtered slope is +1.

```text
confidence = 2 / (2 + 1 + 0.5) ≈ 0.5714
blended_slope = 0.75*2 + 0.25*1 = 1.75
effective_trend = 0.5714 * 1.75 = 1.00

at h=10: effective_horizon ≈ 12*(1-exp(-10/12)) ≈ 6.785
forecast ≈ 200 + 6.785*1.00 = 206.8 zombies
```

This demonstrates the formula; it is not a logged game result.

## 7. Online scoring without future leakage

`ForecastTracker(horizon=h)` keeps a dictionary of pending forecasts
indexed by **target tick**, not origin tick.

At origin tick `t`:

1. Resolve any forecast stored for the **current** tick using actual
   `y_t` (such a forecast was recorded `h` ticks earlier).
2. Queue the new forecast for key `t+h`.
3. Queue a persistence prediction of `y_t` as that forecast's baseline.

The scoring formulas are:

```text
error_i = actual_target_i - predicted_target_i
MAE = mean(abs(error_i))
RMSE = sqrt(mean(error_i^2))

naive_error_i = actual_target_i - actual_origin_i
skill = 1 - (model_MAE / naive_MAE)   # only if naive_MAE > 0
```

Positive skill means the model has lower MAE than persistence; zero
means a tie; negative skill means persistence is better. Smaller
MAE/RMSE is better. RMSE weights larger misses more heavily.

The first `h` predictions have **not matured** yet. If a run has
`T+1` population observations from tick 0 through tick `T`, and it
runs all those ticks, then the count of evaluated origin-target
pairs is `T + 1 - h` (when positive). For example, 301 observations
and `h=10` yield `291` completed forecasts.

## 8. The phase-split intervention experiment

The headless runner begins with Thompson Sampling. When
`--shift-tick 150` is supplied, it switches action selection
immediately **before** executing tick 150.

For reporting phase-specific errors, `evaluate_shift_series` uses:

```text
pre:  origin + horizon < shift_tick
post: origin >= shift_tick
```

Both forecast origin and target must fall in the same phase.
Predictions that cross the tick-150 boundary are **excluded**.
The forecaster itself **is not reset** at the shift: post-phase
predictions can still be informed by observations before tick 150,
which is important for studying adaptation.

Caution: a changed policy can change subsequent random-number
consumption, and a policy switch is not automatically proof of a
detectable statistical distribution shift.

## 9. Results that are safe to claim

The completed five-seed, 1,000-agent, 300-tick **no-intervention
control** with a 50-observation window yielded average model MAE
versus persistence MAE of:

| Horizon | Model | Persistence |
| --- | ---: | ---: |
| 10 ticks | 2.66 | 2.79 |
| 25 ticks | 4.86 | 5.16 |
| 50 ticks | 7.82 | 8.33 |

Across the matched control and switch conditions, *post-phase
persistence-baseline* errors were:

| Horizon | Control | Switch at tick 150 |
| --- | ---: | ---: |
| 10 ticks | 2.51 | 2.61 |
| 25 ticks | 4.81 | 5.32 |
| 50 ticks | 8.10 | 9.55 |

The second table concerns the **baseline's** predictability of
the zombie population; it must **not** be misreported as a direct
comparison of Adaptive Kalman model errors. The effect varies by
seed and was not positive for every individual trial.

With only five seeds and no paired uncertainty analysis, neither table
establishes statistical significance or universal improvement.

## 10. Ways to improve the research without changing the game

- Compute matched model MAE for control and shifted runs at the same
  seeds, horizons, windows, and phases.
- Report per-seed paired differences with uncertainty estimates rather
  than presenting only aggregate averages.
- Plot how forecast error responds during the 10–50 ticks immediately
  after the policy intervention.
- Compare against at least one additional forecasting baseline and
  report computation/runtime alongside MAE/RMSE.
- Evaluate robustness across larger agent populations and distinct
  simulation conditions.

These are **future experiments**, not features claimed to be implemented
in the current source.

## 11. Reproduction commands

```powershell
py -m unittest discover -s tests -v

py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --output results/control_v1
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --shift-tick 150 --output results/shift_v1

py plot_experiments.py --input results/control_v1
py plot_experiments.py --input results/shift_v1
```

The standard plotter produces `mae_h10.png`, `mae_h25.png`,
`mae_h50.png`, and `population_trajectory.png` for each output
directory. The separately committed `strategy_shift_comparison.png`
illustrates the matched persistence comparison; the standard plotter
does not currently recreate that specialized image automatically.
