# Software Architecture and Code Walkthrough

This guide is a source-grounded map of the merged CPS310 Zombie Game. It is for a new contributor who wants to understand **what owns the state, where each agent is created, when behavior runs, and what changes during a tick**.

Related reading: [Integration guide](INTEGRATION.md) · [Swarm policy explanation](SWARM_GAME_THEORY.md) · [Forecast mathematics](FORECAST_MATH.md) · [Running and defense guide](RUNNING_AND_DEFENSE.md).

## 1. System at a glance

The repository has four distinct responsibilities:

```text
Zombie_Game.py                        models.py
  SimulationPlayground ----------------> LinearThompsonSwarmLearner
       |                                     | select_action(context)
       | agent lists, grids, scent           | update(action, context, reward)
       | step()                               | step() / posterior resampling
       |
       +-- Human / ArmedHuman / Zombie
       |     [OOP update() + move/attack rules]
       |
       +-- SimulationGUI --------------> forecasting.py
       |     |                              SlidingWindowAdaptiveKalman
       |     | observed zombie count        ForecastTracker
       |     | <---- future forecasts
       |     |
       |     +--------------------------> db_manager.py
       |                                     DatabaseManager.save_run(...)
       |
experiment_runner.py
  SimulationPlayground (headless; no Tkinter window or DB object)
       +--> forecasting.py --> metrics.csv, phase_metrics.csv, population_seed_N.csv
                                            |
plot_experiments.py -------------------------+--> selected PNG figures
```

The **agent simulation is authoritative** about actual human/zombie counts.
The **swarm learner chooses zombie behavior**; it is not a forecaster.
The **forecasting code observes historical zombie counts**; it does not
select actions, train the policy, or change the simulation.

The normal GUI's `strategy_mode` is `"thompson"`. The controlled
research runner alone may switch it to `"rush_only"` at a chosen tick.

## 2. File-by-file responsibilities

| File | Main public symbols | Why it exists |
| --- | --- | --- |
| `Zombie_Game.py` | `Agent`, `Human`, `ArmedHuman`, `Zombie`, `SpatialGrid`, `SimulationPlayground`, `SimulationGUI` | Physics/gameplay, agent creation and updates, drawing, entry point |
| `models.py` | `LinearThompsonSwarmLearner` | Shared contextual Thompson action policy and posterior update |
| `forecasting.py` | `SlidingWindowAdaptiveKalman`, `ForecastTracker` | Online state estimation, forecasts, and matured forecast scoring |
| `experiment_runner.py` | `run_simulation`, `evaluate_series`, `evaluate_shift_series`, `main` | Scriptable seeded experiments and evaluation CSVs |
| `plot_experiments.py` | `main` | Batch graph generation with Matplotlib's noninteractive Agg backend |
| `db_manager.py` | `DatabaseManager` | PostgreSQL schema creation and end-of-run storage |
| `tests/` | `unittest` test cases | Regressions for forecast, experiment, and Game Theory integration |
| `.github/workflows/forecasting-tests.yml` | GitHub Actions job | Run Python unit tests on pushes and pull requests |

## 3. Object-oriented class hierarchy

```text
                Agent (abstract)
                - x, y, base_speed, health, alive
                - distance_to(), move_towards(), move_away_from()
                - take_damage()
                - update(arena) [abstract]
                   /                 \
                  /                   \
               Human                 Zombie
               - wander/flee         - hunt/bite/scent
               - deposit scent      - contextual action
                  |
               ArmedHuman
               - shoot + cooldown
               - flee and wander

SpatialGrid           SimulationPlayground         SimulationGUI
(non-agent helper)    (simulation owner)          (Tkinter presenter)
```

### Inheritance versus composition

- **Inheritance:** `Human` and `Zombie` extend `Agent`; `ArmedHuman`
  extends `Human`. Each implements or overrides `update(arena)`.
- **Polymorphism:** `SimulationPlayground.step()` invokes `update(self)`
  on each agent without duplicating movement logic in the arena.
- **Composition:** `SimulationPlayground` owns a `LinearThompsonSwarmLearner`,
  two `SpatialGrid` objects, a scent map, and live agent lists.
  `SimulationGUI` holds a reference to the arena plus a separate forecaster
  and tracker.
- **State encapsulation:** the learner owns `A_inv`, `b`, and sampled
  coefficients. The forecaster owns its state, covariance, rolling innovations,
  and rolling observed differences. The database manager owns connection config.

The program does **not** reconstruct all agents from SQL for each tick.
The active agent objects live in memory.

## 4. Where the instances are created

The bottom of `Zombie_Game.py` has the executable entry point:
`if __name__ == "__main__":`.

1. Construct `DatabaseManager(...)` (schema initialization is attempted).
2. Construct `SimulationPlayground(width=1200, height=1200, workers=1)`.
3. Call `populate(num_regular=750, num_armed=50, num_zombies=200)`.
4. Create the Tk root and `SimulationGUI(...)`.
5. Enter `app.mainloop()`, which schedules the repeated GUI update loop.

Within `populate()`, separate loops call `Human(...)`, `ArmedHuman(...)`,
and `Zombie(...)`. The new Python objects are appended to
`self.humans` or `self.zombies`. A lethal bite calls
`convert_human_to_zombie()`, which queues a new `Zombie(...)` in
`new_zombies`, to be appended after agent updates.

For headless experiments, `experiment_runner.run_simulation()` creates a
new `SimulationPlayground` and calls `populate()` but does not create
`DatabaseManager` or a Tkinter root. It records counts directly in CSV.

## 5. Tick order, including temporal subtleties

`SimulationPlayground.step()` has a deliberate order:

1. Increase `tick_count`; clear `new_zombies`.
2. Update scent via `diffuse_and_decay_scent()`.
3. Advance the policy's clock with `swarm_learner.step()`. The learner
   refreshes sampled action parameters every five steps by default.
4. Rebuild separate human and zombie spatial grids.
5. Optionally calculate nearest-target snapshots in subprocesses when
   `use_multiprocessing` is enabled **and** `workers > 1`.
6. Update all humans (including armed shots and negative policy feedback).
7. Update all zombies (starvation, target selection, pursuit, bite,
   positive reward, possible conversion).
8. Remove temporary `_parallel_target` fields; append queued conversions;
   filter dead humans and zombies from the lists.

**Why the exact ordering matters:** the grids represent positions before
agents have moved on the current tick. A cached target can die later during
the same tick, so some code rechecks `target.alive` before acting. Agent
movement and policy updates happen sequentially in the parent process;
multiprocessing only accelerates nearest-target lookup, when enabled.

**Agent rules versus step order:** human movement can alter positions before
zombies update, while spatial-grid candidate membership still reflects the
start-of-tick snapshot. Do not describe the implementation as a physically
simultaneous update.

## 6. Spatial partitioning

`SpatialGrid(cell_size=80.0)` maps integer cell coordinates to agents.
`insert(agent)` computes the cell from position, while `query_radius()`
collects agents from any grid cells overlapping the axis-aligned bounding
square of the requested search radius.

The returned list is a **candidate set**, not an exact circle:
callers separately test Euclidean distance and whether the agent is alive.
This improves typical local-neighbor search but can still return many
candidates in a crowded area. No fixed performance speedup is guaranteed.

The optional parallel path uses `_snapshot_grid()` to serialize position
and alive state and `_nearest_targets()` to find indices of candidates;
the main process maps those indices back to the real live objects.

## 7. Scent system and fallback movement

Humans deposit a value in the 50-unit-cell scent map when they update.
At the start of each tick the map decays and diffuses into neighboring
cells. A zombie without a visible target samples nearby scent cells and
moves toward a stronger one if it exists; otherwise it steers away from
boundaries or wanders.

The scent map is not a learned neural network or additional Thompson
Sampling context feature. It is a separate, hand-coded pursuit fallback.
The use of `np.roll` for diffusion implies wraparound at map edges; this
is a mathematical implementation detail, not a guarantee of physically
realistic diffusion.

## 8. GUI, filtering, and forecast ownership

`SimulationGUI.update_loop()` calls the arena's `step()` first, then
appends the new human/zombie counts. It calls the forecaster with the
**observed** zombie count and registers a new prediction for tick `t+10`.

`ForecastTracker` resolves a pending forecast only after its future tick
has actually happened. GUI rendering is throttled (every fourth tick by
default), whereas simulation, learning, and forecasting still run on
every tick. The dashed orange line shows a prediction rather than a
second measured population.

Changing the 20/50/100 radio button rebuilds the forecaster and replays
**historical observed counts**. It does not restart the game, alter
`swarm_learner`, or change any agent objects.

## 9. Database is an output, not the simulation engine

`DatabaseManager` creates `simulation_runs` and
`population_snapshots` (foreign-keyed by `run_id`). At an end condition
or a manual GUI close, it attempts to save a run summary and a sampled
history (stride `max(1, len(history) // 300)`).

The database does not own or control the current agent objects. An
unavailable server may cause schema initialization or logging messages
without affecting headless simulation. Avoid claiming persistence works
without checking PostgreSQL separately.

## 10. Safe extension points

- To change *forecasts*, edit `forecasting.py` and add unit tests; do not
  alter the learner's action logic.
- To change *plots or evaluation*, edit `experiment_runner.py` or
  `plot_experiments.py`, keeping reported seeds and metric definitions stable.
- To change *game-theory decisions*, coordinate with the learner's contributor
  and add regression cases for action selection, reward updates, and timing.
- To change *simulation mechanics*, review how update ordering affects
  learning rewards and forecast targets.
- To change *database persistence*, test both completed and manually
  stopped GUI runs; changing DB storage must not implicitly change the
  simulation state.

See [the integration regression tests](../tests/test_swarm_integration.py)
and [the merged integration PR](https://github.com/dokeefe414/CPS310-Zombie-Game/pull/3).
