# Time Series Forecasting Specialization

## Research focus

**Sliding-Window Adaptive Kalman Filtering for Zombie Population Forecasting**

Research question: How does the sliding-window length affect the accuracy and
adaptation of short-horizon zombie population forecasts when the agent policy
changes (e.g., Owen's Thompson Sampling swarm policy)?

The forecasting module observes simulation output only. It does not change
Thompson Sampling, agent movement, rewards, or the simulation's event ordering.

## Implementation

- `forecasting.py`: 2-dimensional state [zombie count, zombies-per-tick trend].
- Constant-velocity state transition `F=[[1,1],[0,1]]`.
- Online Kalman prediction/update with Joseph-form covariance update.
- Rolling innovation variance (default 50 ticks) adapts measurement noise.
- 10-tick-ahead point forecasts are extrapolated from the filtered state.
- Forecasts are clipped to [0, initial total agents] for the current closed
  simulation, where conversions conserve agent count.
- `ForecastTracker` stores forecasts indexed by *future target tick* and only
  scores them when the true count arrives; no future observations leak into
  training or evaluation.
- Naive baseline: next 10-tick count equals today's count.
- GUI: orange dashed forecast extension, status-bar predicted zombie count
  and observed 10-tick MAE. Human and zombie historical curves remain.

The window adapts **measurement-noise estimates**, not the state transition
parameters. The trend state updates each tick, so the model can respond to
changing growth/decline. It is a heuristic adaptive Kalman model, not a
guaranteed optimal estimator or a demonstrated performance improvement.

## Run

Requires Python 3.10+, NumPy, and the simulation's existing dependencies
(`psycopg2`/PostgreSQL and Tkinter for the GUI).

```powershell
python -m pip install numpy psycopg2-binary
python -m unittest discover -s tests -v
python Zombie_Game.py
```

The existing database credentials and PostgreSQL setup still apply.
Do not commit database passwords to public repositories; move them to
environment variables before deployment.

## Planned experiment

1. Fix the simulation configuration and run many independent random seeds.
2. Record full per-tick zombie counts and strategy/configuration metadata.
3. Compare window lengths 20, 50, and 100, holding other settings fixed.
4. Compare 10-tick MAE/RMSE against the naive forecast, with no look-ahead.
5. Repeat under the original agent policy and Thompson Sampling, if a
   controlled policy toggle is added. Record when any strategy change occurs.
6. Report mean, variability, and run counts; measure update time separately.

**Current limitation:** the GUI keeps history in memory and the database
saves downsampled population snapshots. A full-resolution CSV export or
experiment runner is recommended before collecting research-grade datasets.
No benchmark or forecasting accuracy results are claimed yet.

## Collaboration

Work on branch `feature/time-series-forecasting`. Owen can continue
modifying `models.py` and the simulation on `main`. Merge only after
reviewing conflicts in `Zombie_Game.py` and rerunning tests.
