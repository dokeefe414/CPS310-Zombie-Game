# CPS310 Zombie Game

## Contributors
Daniel O'Keefe, Owen Bowers, Rahsaun Jeffrey

## About the Project
This CPS310 project is an agent-based zombie outbreak simulation. Civilians, armed civilians, and zombies move through a square arena and interact according to their own behaviors. Civilians try to avoid zombies, armed civilians can shoot nearby zombies, and zombies pursue humans and convert those they kill into new zombies.

The project began as a small Stage 1 prototype. **Stage 2 implements a larger 10,000-agent setup**, spatial partitioning, multiprocessing-assisted target searches, PostgreSQL run logging, and a redesigned Tkinter display.

## Stage 1: Initial Prototype
The original prototype started with 25 civilians, 8 armed civilians, and 6 zombies. It demonstrated basic agent movement, chasing and fleeing, ranged attacks, bites and infection, health displays, and a live population graph. Stage 2 builds on those core interactions while changing several movement, combat, and rendering parameters.

## Stage 2: Implementation

### 1. Larger Simulation
The Stage 2 entry point in `Zombie_Game.py` initializes a **1,200 × 1,200-unit** arena and populates it with **10,000 agents**:

| Agent type | Initial count |
| --- | ---: |
| Civilians (`Human`) | 7,500 |
| Armed civilians (`ArmedHuman`) | 500 |
| Zombies (`Zombie`) | 2,000 |
| **Total** | **10,000** |

Agents are assigned random initial positions. The population can change as humans die or are converted into zombies, and zombies die from damage or starvation.

### 2. Object-Oriented Design
The simulation uses inheritance and polymorphism to share behavior while giving each agent type its own update logic:

- **`Agent`** is an abstract base class with position, speed, health, alive status, distance calculations, movement methods, arena-boundary checks, and damage handling. It declares an abstract `update()` method.
- **`Human(Agent)`** represents an unarmed civilian. It searches for nearby zombies and flees when threatened; otherwise, it wanders randomly.
- **`ArmedHuman(Human)`** extends civilian behavior with ranged attacks, shot damage, and a cooldown timer.
- **`Zombie(Agent)`** hunts humans, loses a small amount of health each tick, bites targets at close range, and triggers conversions when bites kill a human.
- **`SpatialGrid`** groups agents by position to make nearby-entity queries more efficient.
- **`SimulationPlayground`** owns the live agent collections, spatial grids, tick counter, multiprocessing worker pool, and simulation update sequence.
- **`SimulationGUI`** manages the Tkinter arena, status display, population graph, and end-of-run logging.
- **`DatabaseManager`** (in `db_manager.py`) creates the PostgreSQL tables and saves completed runs and sampled population histories.

Live agents are stored in Python lists (`humans` and `zombies`) during the simulation, rather than loaded from PostgreSQL every tick.

### 3. Agent Behavior and Combat
The current Stage 2 code uses these parameters:

| Property | Civilian | Armed civilian | Zombie |
| --- | ---: | ---: | ---: |
| Starting health | 100 | 100 | 120 |
| Movement speed | 2.4 | 2.1 | 2.1 |
| Detection / hunt radius | 25 | 25 (flee), 35 (attack) | 120 |
| Attack damage | — | 45 per shot | 40 per bite |
| Attack range | — | 35 | 3.2 |

Additional mechanics:
- Civilians and armed civilians move away from a detected nearby zombie; otherwise they wander at 40% of their base speed.
- Armed civilians have a six-tick shot cooldown, with a randomized initial cooldown.
- Zombies lose 0.05 health per tick from starvation.
- A successful zombie bite restores up to 18 health, capped at the zombie's maximum health.
- If a bite kills a human, a new zombie is queued at that human's position and added after the current update sequence.
- Agents are kept within arena boundaries, and dead agents are removed from the active collections after each tick.

### 4. Spatial Partitioning and Multiprocessing
A major Stage 2 change is reducing the cost of finding nearby targets among thousands of agents.

**Spatial grids:** The `SpatialGrid` class divides the arena into **80-unit cells**. At the start of each tick, the simulation rebuilds separate grids for humans and zombies. A search examines candidate agents in cells overlapping the requested radius, instead of always scanning every agent in the arena.

**Parallel target finding:** When multiprocessing is enabled and more than one worker is configured, `SimulationPlayground._parallel_target_search()` uses Python's `ProcessPoolExecutor`. It creates serializable snapshots of agent positions, splits target-finding queries into chunks, and distributes them among worker processes. Workers return indexes of nearest in-range targets; the main process maps those indexes back to live agent objects.

The default Stage 2 entry point configures **four worker processes**. The agent `update()` calls and state changes are then performed sequentially in the main process. In other words, **nearest-target calculations are parallelized, but movement, combat, and event application are not**. This helps avoid concurrent mutations of live agents.

Spatial partitioning also supports the non-multiprocessing path. The implementation is intended to reduce search work, but actual speedup has not been established by published benchmarks in this repository.

### 5. Simulation Tick Sequence
Each call to `SimulationPlayground.step()`:

1. Increments the tick counter and clears the pending-new-zombies list.
2. Rebuilds the human and zombie spatial grids.
3. Performs multiprocessing-assisted target searches if enabled with more than one worker.
4. Updates all humans, including armed civilians and their attacks.
5. Updates all zombies, including starvation, pursuit, bites, and conversions.
6. Clears temporary parallel-target references.
7. Adds newly converted zombies and removes dead agents.

Because spatial grids and parallel target snapshots are built **before** movement and attacks for that tick, target selection uses the positions and alive states captured at the start of the tick. This is an implementation detail to keep in mind when evaluating simulation behavior.

### 6. PostgreSQL Database Integration
Stage 2 includes a `DatabaseManager` class using **psycopg2** to connect to PostgreSQL and create two tables automatically if they do not already exist:

**`simulation_runs`** stores one record per completed run, including:
- Unique run ID and recording timestamp
- Initial regular-civilian, armed-civilian, and zombie counts
- Total number of ticks
- Final outcome (human containment or zombie extinction of humans)

**`population_snapshots`** stores sampled population history linked to a run:
- Snapshot ID and associated run ID
- Tick number
- Human count
- Zombie count

The simulation records human and zombie population counts in memory as it runs. When either side reaches zero, the GUI calls `save_run()` to insert the run metadata and a **downsampled batch** of population snapshots. The snapshot stride is calculated from the history length to reduce database write volume. `psycopg2.extras.execute_values` performs the batch insert.

**Important:** The current implementation does **not** persist each individual agent's coordinates, health, or combat events. Database writes happen at the end of a naturally completed simulation, not continuously each tick. If the window is closed early, the current code does not call `save_run()`.

### 7. Tkinter Visualization
The Stage 2 GUI uses a dark-themed Tkinter interface with a simulation arena and a live population graph.

- **Cyan:** unarmed civilians
- **Yellow:** armed civilians
- **Green:** zombies

Agents are drawn as small rectangles, allowing the interface to represent much larger populations than the original prototype. A status bar shows the tick number, surviving civilians, armed civilians, zombies, and total population. The graph tracks human and zombie counts over time and samples its plotted points to limit drawing work.

The update loop is scheduled with Tkinter's `after(20, ...)`. This is a requested delay between GUI callbacks, **not a guaranteed simulation frame rate**. A run ends when all humans or all zombies have been eliminated, at which point the GUI displays the winning side and attempts to log the results.

### 8. Running Stage 2

**Requirements**
- Python 3.10+ (the code uses modern type annotations)
- Tkinter (often included with Python; some Linux installations require a separate system package)
- PostgreSQL server
- Python package `psycopg2` (or `psycopg2-binary` for a convenient local development installation)

Install the Python database driver:

```bash
python -m pip install psycopg2-binary
```

Create a PostgreSQL database and a user with permission to connect and create tables. The current entry point expects database `zombie_db` on `127.0.0.1:5432`, using role `zombie_admin`. **Update the connection settings in `Zombie_Game.py` and `db_manager.py` for your local environment. Do not reuse or commit real database passwords.** The current source contains a hardcoded development password; moving credentials to environment variables is recommended.

Then run:

```bash
python Zombie_Game.py
```

The program attempts to initialize the database schema at startup. Database setup or write errors are printed to the console. A running PostgreSQL instance and correct credentials are required for successful persistence.

### 9. Stage 2 Status and Future Improvements
**Implemented in the current source:** 10,000-agent initialization, updated agent rules, spatial grids, multiprocessing-assisted nearest-target searches, Tkinter visualization, PostgreSQL schema creation, and end-of-run population-history logging.

**Areas for testing or improvement:**
- Benchmark sequential versus multiprocessing performance at different population sizes.
- Measure simulation tick times and GUI rendering overhead.
- Validate behavior when multiple agents target the same entity in one tick.
- Move database credentials to environment variables.
- Add automated tests, dependency documentation, and database failure handling.
- Consider optional intermediate checkpoints for long-running simulations.

The source shows these features have been coded; it does not by itself establish that the complete 10,000-agent configuration has been performance-tested or runs at a particular speed.
