# Time Series Forecasting: Implementation and Experiments

## Research question
How does innovation-window length affect the accuracy and responsiveness of an
online adaptive Kalman filter forecasting zombie population under evolving
Thompson Sampling agent behavior?

The forecasting system **observes** the game and does not alter Owen's
`models.py` or the zombie decision policy.

## Quick start (Windows PowerShell)
```powershell
git fetch origin
git switch feature/time-series-forecasting
git pull origin feature/time-series-forecasting
py -m pip install numpy psycopg2-binary matplotlib
py -m unittest discover -s tests -v
py Zombie_Game.py
```

The GUI starts at 1,000 agents by default (750 regular humans, 50 armed,
200 zombies), with a single process to avoid unnecessary Windows worker
startup overhead. The 20/50/100 radio buttons change the sliding innovation
window and replay the *observed* history to produce comparable forecasts.
The live status displays forecast MAE, average simulation-step milliseconds,
and average drawing milliseconds. The arena and graph redraw every fourth
simulation tick; **simulation and forecasting still update every tick**.

The game still requires its existing PostgreSQL setup for database persistence.
Use environment variables for database credentials before publishing or
deploying. The headless experiment runner does not instantiate DatabaseManager
or Tkinter windows, but it imports the main module and therefore needs
`psycopg2-binary` and a Python build with Tkinter available.

## Automated experiments
```powershell
py experiment_runner.py --agents 1000 --ticks 250 --seeds 1 2 3 --windows 20 50 100 --horizons 10 25 50
py plot_experiments.py --input results/forecasting
```

For a larger stress test (may take significantly longer):
```powershell
py experiment_runner.py --agents 5000 --ticks 100 --seeds 1 --output results/forecasting_5k
py experiment_runner.py --agents 10000 --ticks 100 --seeds 1 --output results/forecasting_10k
```

Outputs:
- `population_seed_N.csv`: **full-resolution** per-tick population counts.
- `metrics.csv`: per-seed/window/horizon MAE, RMSE, naive MAE/RMSE,
  number of completed predictions, simulation runtime, and forecasting runtime.
- `config.json`: exact experiment arguments.
- `mae_h10.png`, `mae_h25.png`, `mae_h50.png`: model vs naive baseline.
- `population_trajectory.png`: human/zombie counts over simulation time.

Every window/horizon combination is evaluated on **the same simulation
trajectory for each seed**, ensuring a paired comparison. Forecasts are
registered at origin tick t and scored only when actual count at t+h arrives.
Short runs may have zero completed h-step forecasts. Terminal/extinction
trajectories are not artificially extended. The simulation uses Python and
NumPy global RNG seeds; do not run multiple seeded simulations concurrently
in the same process.

## Model details
The filter's state is [zombie population, growth rate]. The state transition
is a local linear trend, and an innovation window estimates changing
measurement noise. The window is **not** a supervised training sequence or a
fixed lag regression model. Window sizes affect adaptation of observation
noise, not direct truncation of the Kalman state. Predictions are clipped to
[0, total initial agents] for the closed population game.

The baseline predicts that the population stays unchanged over the horizon.
We make **no claim** that the adaptive filter beats the baseline until
replicated experiments demonstrate it. Current results are not validated
against a controlled policy switch: Thompson Sampling adapts during each run,
but no explicit before/after strategy intervention has been implemented.

## Research methodology and limitations
1. Compare 20/50/100 windows on the same seeds and initial conditions.
2. Report paired per-seed differences versus naive MAE/RMSE at 10/25/50 ticks.
3. Compare computational runtimes at 1k, 5k and 10k agents.
4. For a causal strategy-shift experiment, later add a controlled policy toggle
   with an exact intervention tick, then report pre/post metrics separately.
5. Record seeds, parameters, early extinction and total completed forecasts.

**Performance caveat:** decimating drawing reduces UI work but does not
optimize the per-agent simulation. Large runs may remain slow due to Python
agent loops, spatial queries and Thompson Sampling. Benchmark first; do not
assume 10k agents will run in real time. ProcessPoolExecutor parallelizes
nearest-target searches, not the entire event update loop.

## Collaboration
Work remains on `feature/time-series-forecasting`, separate from main.
Review and merge carefully with concurrent work on `Zombie_Game.py`.
