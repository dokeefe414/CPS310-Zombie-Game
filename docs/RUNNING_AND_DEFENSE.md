# Running, Troubleshooting, Testing, and Defending the Project

This is a practical guide for running the **merged `main` branch** on Windows,
reproducing the forecasting research, checking database persistence, and
explaining the project during a CPS 310 demonstration.

For the code-level details use [Architecture](ARCHITECTURE.md),
[Game Theory](SWARM_GAME_THEORY.md), [Forecast mathematics](FORECAST_MATH.md),
and [Integration](INTEGRATION.md).

## 1. Safe setup on Windows

Start in a PowerShell terminal inside VS Code:

```powershell
cd C:\Users\danie\CPS310-Zombie-Game
git status
git switch main
git pull origin main
py --version
py -m pip install numpy matplotlib psycopg2-binary
```

These instructions use `py`, the Windows Python launcher. The path above is
an **example existing working-copy path**, not a requirement for all users.

If `git status` shows local edits, review and preserve those edits before
switching branches or pulling; do not force-reset or overwrite your work.
Python 3.10+ is expected by the project, and Python 3.11 is used in CI.

Tkinter is part of typical Windows Python installations. `matplotlib` is
used for batch figures, `numpy` by the learning/forecasting model, and
`psycopg2-binary` by the PostgreSQL persistence layer.

## 2. Run the interactive game

```powershell
py Zombie_Game.py
```

The current GUI entry point initializes:
- A 1,200 × 1,200 logical simulation arena.
- 750 civilian humans, 50 armed humans, and 200 zombies.
- A `SimulationPlayground` in normal **Thompson Sampling** mode.
- An adaptive zombie-population forecast with a default 50-observation
  window and 10-tick forecast endpoint.
- A Tkinter arena and history graph, redrawn every fourth simulation tick.
- An attempted PostgreSQL schema initialization on startup.

### What you should see

| UI element | Meaning |
| --- | --- |
| Cyan shapes | Unarmed humans |
| Yellow shapes | Armed humans |
| Red shapes | Zombies |
| Human and zombie graph lines | Observed population history |
| Dashed orange graph line | Predicted zombie population for the next 10 ticks |
| 20 / 50 / 100 radio buttons | Alternative rolling-window lengths (no game reset) |
| `Zombies +10` status | Model's predicted count ten ticks ahead |
| `10-tick MAE` status | Error of **matured** ten-tick predictions, after warm-up |
| `Step` and `Draw` timings | Recent mean simulation/GUI work time, **not** an FPS guarantee |

The labels display actual counts measured from the current agent lists.
The orange future line is a prediction, not another observed trajectory.

**Smoke-test procedure:** Let the game run at least 20 ticks, observe
chasing/shooting and the forecast, switch to windows 20 → 50 → 100,
and confirm the simulation tick continues to increase without
repopulating agents.

## 3. Tests

Run every discovered test:

```powershell
py -m unittest discover -s tests -v
```

The repository includes:
- `test_forecasting.py`: initialization, trend response, bounds, window
  sensitivity, invalid inputs, and no-lookahead tracker scoring.
- `test_experiments.py`: completed forecast samples, small headless runs,
  phase boundaries, and strategy intervention timing.
- `test_swarm_integration.py`: default Thompson mode, all three actions,
  bite/reward posterior updates, sampling clock, optional rush-only mode,
  unchanged policy state on forecast-window replay, and matched pre-shift
  simulation history.

The workflow `.github/workflows/forecasting-tests.yml` installs required
packages and runs this suite on GitHub Actions. A passing test suite is
evidence about those specific tested behaviors, not a proof that all GUI,
database, performance, and randomness paths are error-free.

## 4. Headless research experiments

The script does not open a game window and does not instantiate a database
manager. It **does** import the game module and therefore needs the Python
dependencies and a working Tkinter installation.

### Basic population forecast experiment

```powershell
py experiment_runner.py --agents 1000 --ticks 250 --seeds 1 2 3 --windows 20 50 100 --horizons 10 25 50 --output results/forecasting_v2
py plot_experiments.py --input results/forecasting_v2
explorer .\results\forecasting_v2
```

### Matched policy comparison

```powershell
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --output results/control_v1
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --shift-tick 150 --output results/shift_v1
py plot_experiments.py --input results/control_v1
py plot_experiments.py --input results/shift_v1
```

The `--shift-tick 150` option changes **action selection only**, beginning
at tick 150. Without that option, Owen's Thompson policy remains active.

### Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--agents` | `1000` | Initial total agent count (75% human, 5% armed, balance zombies, subject to rounding) |
| `--ticks` | `200` | Maximum number of simulation ticks; an extinct side ends the run early |
| `--seeds` | `1 2 3` | List of integer random seeds |
| `--windows` | `20 50 100` | Online filter window lengths (integers ≥ 3) |
| `--horizons` | `10 25 50` | Forecast horizons (positive integers) |
| `--workers` | `1` | Worker count; multiprocessing-assisted target search when > 1 |
| `--shift-tick` | Omitted | Tick at which action selection switches to rush-only |
| `--output` | `results/forecasting` | Directory for CSV, JSON, and generated graphs |

**Output files** include:

```text
results/<experiment>/
    population_seed_1.csv    # tick, humans, zombies, strategy
    population_seed_2.csv
    ...
    metrics.csv              # one row per seed/window/horizon
    phase_metrics.csv        # only produced for a --shift-tick run
    config.json              # CLI configuration for reproducibility
    mae_h10.png              # created by plot_experiments.py
    mae_h25.png
    mae_h50.png
    population_trajectory.png
```

The `results/` directory is ignored by Git. Selected figures are
intentionally committed under `figures/forecasting/`. Do not add
large raw experimental files without a project decision to version
them. The specialized `strategy_shift_comparison.png` is committed
but is **not generated by the standard plotter**.

## 5. PostgreSQL database persistence

On GUI launch, `DatabaseManager` tries to create two tables:

```text
simulation_runs
  run_id (primary key)
  recorded_at
  initial_regular_humans
  initial_armed_humans
  initial_zombies
  total_ticks
  outcome

population_snapshots
  snapshot_id (primary key)
  run_id (foreign key to simulation_runs)
  tick_number
  human_count
  zombie_count
```

The GUI attempts to save a run when the simulation naturally ends;
it also attempts to save a run tagged
`TERMINATED_EARLY (Manual Exit)` on window close after a tick.
History is **downsampled** for SQL storage using
`stride = max(1, len(history) // 300)`. The headless runner's
`population_seed_N.csv` instead retains every sampled tick.

**Checking persistence:** use your configured PostgreSQL client and
inspect recent rows with read-only queries such as:

```sql
SELECT run_id, recorded_at, total_ticks, outcome
FROM simulation_runs
ORDER BY run_id DESC
LIMIT 10;

SELECT tick_number, human_count, zombie_count
FROM population_snapshots
WHERE run_id = 1
ORDER BY tick_number;
```

Replace `1` with a run ID returned by the first query. The source
currently uses local database configuration in the GUI entry point;
do **not** commit real credentials or reuse the development password
in any production environment. Database setup and write errors are
caught and printed by `DatabaseManager`, so **a game window appearing
is not sufficient evidence that database saving succeeded**.

Potential caveat to inspect if testing repeated runs: the GUI's
`on_close()` persists an early-termination row after at least one
observation even if the game has already reached a natural end, which
can lead to a **second run record** if the window is closed later.
The documentation flags this behavior; fixing it would require a
separate reviewed code change.

## 6. Troubleshooting by symptom

| Problem | Check | Suggested action |
| --- | --- | --- |
| PowerShell: `python` not found | Windows Python launcher | Use `py --version` and `py Zombie_Game.py`, not `python` |
| `ModuleNotFoundError: numpy` | Environment/package install | `py -m pip install numpy` |
| `ModuleNotFoundError: psycopg2` | Database driver | `py -m pip install psycopg2-binary` |
| `ModuleNotFoundError: matplotlib` | Batch plotting dependency | `py -m pip install matplotlib` |
| `tkinter` import or no GUI display | Python installation/display | Verify `py -m tkinter` opens a test window; use a Tk-capable Python |
| PostgreSQL connection refused | Server/host/port/config | Verify PostgreSQL is running and local connection settings match the configured server |
| `permission denied` or table creation fails | Database user grants | Give your local development role appropriate permissions; do not change credentials in a public commit |
| GUI seems slow | Number of agents / rendering workload | Start with the 1,000-agent default; use headless mode for large experiments |
| No MAE shown yet | Forecast warm-up | Allow at least ten future ticks to complete so the first prediction can be scored |
| No rows in `phase_metrics.csv` | Shift not enabled | Re-run with `--shift-tick`; this file exists only for intervention runs |
| `No completed forecasts` | Too few ticks or early extinction | Use more ticks, fewer agents if necessary, and/or shorter horizons |
| GUI works but no DB rows | Save failure printed in terminal | Inspect server connectivity, permissions and save-run console output |
| Git pull refuses due to local edits | Working-tree conflict | Review `git status`; commit or stash intentional work before pulling, rather than discarding it |

## 7. What to demonstrate to a professor

A compact presentation order:

1. **Purpose:** Simulate humans, armed humans, and zombies as
   interacting objects; compare different agent behaviors and forecast
   the changing zombie population.
2. **OOP:** Explain `Agent` as an abstract parent,
   `Human(Agent)`, `ArmedHuman(Human)`, and `Zombie(Agent)`.
   Show `SimulationPlayground.populate()` and `step()`.
3. **Game Theory:** Show the unchanged `models.py`: three
   Thompson actions, a four-feature context, sampled action weights,
   and a reward/posterior update after combat.
4. **Forecasting:** Show `forecasting.py`: two-state online filter,
   rolling differences and innovations, and a conservative ten-tick
   extrapolation. Show the orange dashed graph and window switches.
5. **Research:** Describe the 5-seed control run and the tick-150
   action-selection intervention. Point out MAE, persistence baseline,
   and the distinction between baseline-error and model-error figures.
6. **Engineering:** Show GitHub Actions unit tests, integration
   tests, comprehensive documentation, and SQL run history (if
   verified locally).
7. **Limitations:** No guarantee of 10,000-agent real-time speed,
   only five seeds, no proof of statistical significance, and
   GUI/SQL behaviors require their own smoke tests.

## 8. Questions with concise defense answers

**Where is the code that creates agents?**
`SimulationPlayground.populate()` in `Zombie_Game.py` calls each
agent constructor and appends the objects to Python lists.

**Is the database storing the live simulation?**
No. The Python lists own live agents; PostgreSQL stores run summaries
and sampled population history.

**What is polymorphism here?**
The game calls a shared `update(arena)` interface, and each agent
type supplies different behavior.

**Does the forecast change how a zombie moves?**
No. The forecast only observes completed population counts.
The Thompson learner chooses actions. A separate research-only
intervention can force action 0 starting at a specified tick.

**How do you avoid cheating with future data?**
The model updates using observations through tick `t`; an
`h`-tick prediction is only scored after tick `t+h` occurs.

**What did the experiments show?**
Some reductions in mean error relative to persistence, as well
as differences between control and strategy-switch trajectories.
Results vary between seeds; do not claim universal gains or
statistical significance.

**How did you ensure Owen's work was not overwritten?**
The merged `models.py` remained unchanged; integration regression
tests check the default policy, three actions, posterior rewards,
resampling, and forecast-window isolation.

**What would you improve next?**
Paired model-error analysis with uncertainty, more seeds, and
additional runtime testing. These are research extensions rather than
requirements for demonstrating the current integrated project.

## 9. Maintenance and safe future changes

After new commits, run the full tests and open the GUI for a smoke test.
Changes to `models.py`, context extraction, reward paths, action IDs,
or the tick order may affect Owen's research even if the forecaster
still passes its tests. Coordinate such changes across contributors.

For GitHub's original integration and verification history, consult
[pull request #3](https://github.com/dokeefe414/CPS310-Zombie-Game/pull/3).
