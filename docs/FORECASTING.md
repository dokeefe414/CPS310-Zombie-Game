# Time Series Forecasting: Implementation and Experiments

## Research question
How does innovation-window length affect the accuracy and responsiveness of an
online adaptive Kalman filter forecasting zombie population under evolving
Thompson Sampling agent behavior?

The forecasting system **observes** the game and does not modify Owen's
`models.py`. During normal GUI play, zombies still use the unchanged
Thompson Sampling action selector. Separately, the headless research runner
supports an opt-in `rush_only` action-selection intervention. See
[INTEGRATION.md](INTEGRATION.md) for the complete ownership and interaction
boundaries.

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
is a local linear trend. The sliding window now has **two explicit roles**:
(1) estimating innovation variance for the adaptive measurement noise and
(2) estimating the recent average *observed population change per tick*.
The second role was added after initial 1,000-agent experiments showed identical
window results and substantially worse MAE than persistence.

The forecast blends recent observed growth with the Kalman growth estimate,
shrinks uncertain trends toward zero, and saturates long-horizon extrapolation.
These are deliberately conservative heuristics, not theoretically optimal
Kalman forecasts. They are evaluated without looking at future observations.
Window size is a forecasting hyperparameter, not a supervised lagged-input
training sequence. Predictions are clipped to [0, total initial agents].

`metrics.csv` includes `skill_vs_naive = 1 - model_MAE/naive_MAE`.
Positive values mean improvement; negative values mean worse performance.
Skill is undefined when the naive baseline has zero MAE. Re-run the same
seeded experiments after this change; do not compare different random
trajectories as if they were paired.

The baseline predicts that the population stays unchanged over the horizon.
The revised model has beaten persistence in several individual seed/horizon
comparisons, but not consistently across all tested conditions. A controlled
policy intervention has now been implemented and evaluated on five seeds;
a matched no-intervention control has now been run using five seeds.
The post-intervention persistence-baseline MAEs for control vs strategy switch
were 2.51 vs 2.61 at 10 ticks, 4.81 vs 5.32 at 25 ticks, and 8.10 vs
9.55 at 50 ticks. These differences are **descriptive**, vary by seed, and
do not establish a statistically significant or causal effect. A matched
model-error comparison and uncertainty estimates are still future work.

## Research methodology and limitations
1. Compare 20/50/100 windows on the same seeds and initial conditions.
2. Report paired per-seed differences versus naive MAE/RMSE at 10/25/50 ticks.
3. Compare computational runtimes at 1k, 5k and 10k agents.
4. Use the implemented tick-150 strategy intervention and compare pre/post
   metrics against a matched no-intervention control.
5. Record seeds, parameters, early extinction and total completed forecasts.

**Performance caveat:** decimating drawing reduces UI work but does not
optimize the per-agent simulation. Large runs may remain slow due to Python
agent loops, spatial queries and Thompson Sampling. Benchmark first; do not
assume 10k agents will run in real time. ProcessPoolExecutor parallelizes
nearest-target searches, not the entire event update loop.

## Collaboration and integration checks
The branch integrates forecasting into the game's Tkinter GUI and retains
the existing game-theory implementation in `models.py` without edits.
`tests/test_swarm_integration.py` provides additional regression coverage
for action selection, posterior updates, learner resampling, and replaying
forecast history without advancing the simulation or learner.

**Test-status caution:** Previous local test results predate these newly added
integration tests. Run `py -m unittest discover -s tests -v` again and review
the GitHub Actions checks for the latest commit. Manual GUI/database testing
and review by the original game-theory contributor are still necessary.

The changes remain on `feature/time-series-forecasting` until a reviewed PR
is merged. For a source-level flow description, test matrix, review procedure,
and limitations, see [INTEGRATION.md](INTEGRATION.md).

## Controlled distribution-shift experiment (new)

The research runner now supports a **policy intervention** at an exact tick:
Thompson Sampling (the normal Owen policy) switches to a deterministic
`rush_only` action choice. This intervention is **opt-in** and does not change
the regular GUI or the contents of `models.py`. The rush-only mode bypasses
action selection but retains the game's original reward updates, movement,
scent, damage and simulation mechanics.

```powershell
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --shift-tick 150 --output results/shift_v1
py plot_experiments.py --input results/shift_v1
```

The runner saves `strategy` at every tick in each population CSV and
`phase_metrics.csv` with pre/post intervention MAE, RMSE and persistence
baseline errors. For phase metrics, **both the forecast origin and target**
must be on the same side of the intervention. Predictions spanning the
change point are excluded from the phase comparison to avoid mislabeling
them. The forecaster itself is **not reset** at the intervention; its online
adaptation can therefore be studied.

The matched control run has been completed on seeds 1–5 with 1,000 agents
and 300 ticks, omitting `--shift-tick` and saving to `results/control_v1`.
The `strategy_shift_comparison.png` figure in `figures/forecasting/`
compares **persistence-baseline errors**, not full Adaptive Kalman errors.
The two runs have identical pre-intervention population histories; once
the policy differs, the two policies may consume random draws differently.
More seeds and a paired adaptive-model comparison are needed to support
stronger generalizations.

Cautions: a policy switch does not guarantee a detectable population shift.
Inspect the trajectories and report when an intervention has no observable
effect. The results do not yet estimate uncertainty or statistical
significance automatically; aggregate paired seed-level differences and
confidence intervals before claiming improvements. For 25/50-tick forecasts
allow enough ticks on both sides of the shift to obtain meaningful samples.
