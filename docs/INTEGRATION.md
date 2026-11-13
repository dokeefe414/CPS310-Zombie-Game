# Integration Guide: Zombie Simulation, Swarm Game Theory, and Time Series Forecasting

This guide explains how the **existing contextual Thompson Sampling zombie policy**
and the **new online zombie-population forecast** work together. It is intended
for contributors, reviewers, and an oral project defense.

> **Integration provenance:** This work was developed on
> `feature/time-series-forecasting` and integrated through
> [pull request #3](https://github.com/dokeefe414/CPS310-Zombie-Game/pull/3).
> The pull request records the merge status, exact code changes, and tests.

## 1. Responsibilities and boundaries

| Component | Main responsibility | What it reads | What it changes |
| --- | --- | --- | --- |
| `models.py`: `LinearThompsonSwarmLearner` | Select `rush`, `flank`, or `swarm_rally`; update contextual reward posteriors | Four-element zombie context and rewards | Its own posterior parameters / sampled weights |
| `Zombie_Game.py`: `SimulationPlayground` | Own and update agents, spatial grids, scent map, and tick order | Agents and policy decisions | Live game state; learner clock and reward updates |
| `Zombie_Game.py`: `SimulationGUI` | Draw state, drive simulation ticks, record population observations, and display forecasts | Zombie counts and forecast output | Tkinter views; forecast state; optional end-of-run database records |
| `forecasting.py`: `SlidingWindowAdaptiveKalman` | Estimate population and its trend; forecast future zombie counts | Historical zombie population counts | Only its private filter state; **not** the simulation or learner |
| `forecasting.py`: `ForecastTracker` | Score matured predictions against reality and persistence | Forecast origins, future realized counts | Metrics and pending forecasts |
| `experiment_runner.py` | Run reproducible headless simulations and optional strategy interventions | Simulation state, seed, selected configuration | Research CSV files under `results/`; optional `strategy_mode` |
| `db_manager.py`: `DatabaseManager` | Store run metadata and sampled population histories | GUI run outcome and history | PostgreSQL tables |

**Ownership protection:** The forecasting branch does **not** edit
`models.py`. In the default game, zombies still invoke
`arena.swarm_learner.select_action(context)`; bite and armed-civilian
reward penalties still call `arena.swarm_learner.update(...)`.

## 2. One simulation tick — source-level walkthrough

Each `SimulationPlayground.step()` call:

1. Advances `tick_count`, clears pending conversions.
2. Diffuses/decays the scent grid; calls `swarm_learner.step()`.
3. Rebuilds the human and zombie spatial grids from current agents.
4. Optionally performs parallel target queries if more than one worker is enabled.
5. Updates all humans; armed humans may fire and record negative feedback for
   the targeted zombie's last action.
6. Updates zombies. A zombie with a nearby human extracts a four-value context:
   normalized target distance, local armed-human fraction, nearby ally density,
   and constant bias. The default learner selects one of three strategies.
   Flanking adjusts its heading; swarm rally weights nearby allies; rushing
   pursues the target directly. Successful bites update reward posteriors;
   lethal bites queue conversions.
7. Adds converted zombies and removes dead agents.

In the GUI, **after** this tick finishes, `SimulationGUI.update_loop()`
appends the new human/zombie counts, calls
`forecaster.update(actual_zombie_count)`, calculates the next ten zombie
population predictions, and registers the ten-tick endpoint with
`ForecastTracker`. It only redraws every fourth tick by default, but
**simulation updates, posterior updates, and forecasting happen every tick**.

Important distinction: a forecast made at tick `t` is about `t+10`;
it cannot be scored until tick `t+10` is observed. Graph redraw timing
does not determine when a forecast becomes valid.

## 3. The game-theory boundary and the optional experiment

The simulation constructor sets `strategy_mode = "thompson"` by default.
In the usual GUI the setting is **never changed**. The action-selection
statement is equivalent to:

```python
if arena.strategy_mode == "rush_only":
    action = 0
else:
    action = arena.swarm_learner.select_action(context)
```

The `rush_only` setting exists for **headless research**: `--shift-tick 150`
switches the policy immediately before simulation tick 150. The three
original action names, context dimension, learner implementation, scent
mechanics, movement rules, and reward-update function are preserved.

In `rush_only` mode, action selection is bypassed, but the simulation still
calls the original learner's `step()` and `update()` paths. Consequently,
**the research intervention is a change to action selection**, not a complete
suspension of Thompson-learning state. Be precise about this distinction when
describing the experiment.

The forecaster does not set `strategy_mode` and is never used as a direct
input to `select_action()`. The design is **observation-only forecasting**,
not closed-loop forecast-driven zombie control.

## 4. How the forecast works

`SlidingWindowAdaptiveKalman` keeps a two-component latent state:
`[zombie_population, population_change_per_tick]`. A constant-velocity
Kalman transition and online updates track current counts. Its window is
applied to (a) recent Kalman innovations and (b) observed differences between
consecutive zombie counts.

For a forecast horizon `h`, the model combines a rolling empirical slope
with the filtered slope, shrinks uncertain trends, and saturates the effective
horizon to prevent explosive long-range extrapolation. Forecast counts are
clipped to zero and the initial population bound.

These are **Kalman-informed heuristic forecasts**, not a proof of optimal
Bayesian prediction. A baseline simply predicts the count will remain
unchanged. Evaluate each horizon with MAE, RMSE, sample count, and
`1 - model_MAE / baseline_MAE` when baseline error is nonzero.

The GUI's 20/50/100 selection reconstructs a **new forecaster from the
already-observed population history**. It neither reruns the simulation nor
reinitializes the swarm learner. Changing the window also resets the
in-window forecasting score, which subsequently uses the replayed history.

## 5. Files and commands (Windows PowerShell)

At the project root:

```powershell
git switch feature/time-series-forecasting
git pull origin feature/time-series-forecasting
py -m pip install numpy matplotlib psycopg2-binary

# Full regression suite, including swarm integration
py -m unittest discover -s tests -v

# Interactive game (PostgreSQL must be configured for persistence)
py Zombie_Game.py

# Control and research intervention: same seeds and number of ticks
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --output results/control_v1
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --shift-tick 150 --output results/shift_v1

# Default summary figures for an experiment
py plot_experiments.py --input results/control_v1
```

The automated headless experiment does **not** require a running PostgreSQL
server or an open GUI window, although importing the game module still
requires installed Python packages and Tkinter availability. The GUI's
existing database integration requires a reachable PostgreSQL database and
working credentials. Do not publish a real password.

The runner writes `metrics.csv`, per-seed population histories, and
`config.json`. A shift run additionally writes `phase_metrics.csv`.
The `results/` directory is ignored by Git; **six selected figures**
are committed under `figures/forecasting/`.

## 6. Regression tests and what they protect

| Test file | Main checks |
| --- | --- |
| `tests/test_forecasting.py` | Initialization, bounds, trend behavior, rolling windows, no-lookahead tracker scoring |
| `tests/test_experiments.py` | Matured forecast sample counts, small headless runs, shift boundary, intervention history |
| `tests/test_swarm_integration.py` | Thompson is default, all three actions reachable, rush-only opt-in/reversibility, bite-to-posterior reward path, five-tick learner resampling, GUI window replay isolation, reproducible pre-intervention histories |

**Scope:** These tests cover key automated integration behaviors without
requiring a PostgreSQL server or a Tkinter display. They cannot certify that
every GUI control looks correct, that database persistence is correctly
configured on each machine, that 10,000-agent performance is sufficient,
or that the learner's long-term strategy improves game outcomes.

**CI verification:** [Forecasting tests run #37](https://github.com/dokeefe414/CPS310-Zombie-Game/actions/runs/43351907498)
completed successfully with the expanded regression suite on the feature
branch. Re-run the full suite locally and confirm the **latest** PR checks
remain green before merging. Automated verification does not prove the GUI
or database works on every development machine.

## 7. Manual review checklist with the other contributor

- [ ] Confirm default GUI behavior remains `thompson`; no research-mode
      setting is toggled when running `py Zombie_Game.py`.
- [ ] Verify zombie `rush`, `flank`, and `swarm_rally` decisions still
      route through `LinearThompsonSwarmLearner.select_action()`.
- [ ] Observe bite events / armed-civilian penalties and confirm rewards
      still update the appropriate posterior.
- [ ] Run the live game at 1,000 starting agents and confirm the orange
      forecast is drawn while humans and zombies continue acting.
- [ ] Change the forecast window to 20, 50, then 100: the game should
      continue with the same agents and simulation tick.
- [ ] Verify 10-tick MAE appears after enough ticks, and never uses future
      observations when recording a prediction.
- [ ] Let one game end naturally. Separately, close an active run. Check
      that the expected outcome/early-termination metadata is saved when
      PostgreSQL is available.
- [ ] Confirm the end-to-end regression suite passes locally and on CI.
- [ ] Review the branch diff and approve the PR before merging into `main`.

## 8. Known limitations / non-goals

1. **No claim of perfect correctness.** Automated tests + human review
   increase confidence but cannot prove every possible agent interaction.
2. **Policy intervention alters random paths.** Identical seeds imply identical
   pre-intervention trajectories; different policies may consume randomness
   differently afterward. Interpret post-shift differences carefully.
3. **No established causal/statistical significance.** The documented five
   seeds and figures are descriptive. More seeds, paired uncertainty, and
   model-error controls would strengthen a research claim.
4. **GUI performance differs from headless throughput.** Rendering is sampled;
   tick execution remains CPU-intensive and may be slow at 5k/10k agents.
5. **Credentials are a deployment concern.** The current GUI entry point
   includes a development database password. Use environment variables and
   rotate any reused credential before deployment; avoid unrelated edits
   to collaborator-owned code during this integration.
6. **Manual GUI verification is still required.** CI does not exercise
   real Tkinter rendering or database persistence.

## 9. Release provenance, maintenance, and rollback

- [Pull request #3](https://github.com/dokeefe414/CPS310-Zombie-Game/pull/3)
  records the integration changes and their final merge status.
- [Forecasting tests run #42](https://github.com/dokeefe414/CPS310-Zombie-Game/actions/runs/43352073356)
  passed on the integration branch. Later documentation edits and merge
  commits should also be verified by their own latest checks.
- The original `models.py` file must stay compatible with its existing
  action names, context shape, posterior update methods, and sampling clock.
- Before changing gameplay mechanics, coordinate with the game-theory
  contributor and add or update regression tests.
- The selected research figures are committed; raw results stay local under
  `results/` and are intentionally Git-ignored.
- If a regression appears after merging, revert the merge commit through
  GitHub (or use an appropriately reviewed corrective PR); do not rewrite
  shared `main` history.

This document describes what the source and tests establish. A passing
headless CI suite cannot by itself guarantee Tkinter rendering, live database
persistence, or every possible run of the game.
