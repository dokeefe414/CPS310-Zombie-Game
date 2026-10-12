# CPS310-Zombie-Game

# Contributors

Daniel O'Keefe, Owen Bowers, Rahsaun Jeffrey
## About

This project is an agent-based simulation of a zombie outbreak on a square
playground. We have three types of agents, humans, armed humans, and zombies, that move around and interact:
zombies chase and bite humans, a bite infects the human, and the human mutates 
into a zombie right as its health turns to zero.

Every agent has a life value (0-100) and a movement speed (Civilians is 2.2, Armed Human 2.0 and zombie is 1.65). 
When humans are outside the perception radius (28 units) of a zombie they wander at 1.1 speed.
When a zombie enters the humans perception radius (28 units) the human moves away at their full base speed (2.2 or 2.0). 
The armed human has 40 damage per shot with a cooldown of 8 ticks.
Zombies also have a deadlines of 35 damage per bite and the zombie gains 10 health for every bite of a human.

There is also an updating health bar above each agent as the simulation runs. (100 starting health)
A live population graph is included that updates as the simulations runs. (Starting 25 civilians, 8-armed human, 6 zombies)
Every tick an armed human evaluates living zombies in the arena to find the closest one using Euclidean distance.
If the closest zombie is within attack range <= 35 units and the armed humans cooldown_timer = 0, the weapon fires.
Damage is resolved instantanaously, the targeted zombie recieves take_damage(40) and the coordinates of both
the human and the target are recorded to render a moemtary visual tracer line (bullet_tracers)

This is the Stage 1 prototype for CPS310, built as our first assignment

# Stage 2 Prototype for CPS 310

Stage 2 focuses on scaling the original agent-based zombie outbreak simulation from a small prototype to a much larger environment. The goal is to support **10,000 agents**, introduce **PostgreSQL** for structured data storage, and explore **parallel processing** to improve performance. These are Stage 2 development goals; this section does not imply that all features have already been implemented.

## 1. Scale to 10,000 Agents

The target starting population is:

| Agent type | Count |
| --- | ---: |
| Zombies | 2,000 |
| Civilians | 7,500 |
| Armed civilians | 500 |
| **Total** | **10,000** |

Each agent will continue to have its own state, including position, health, movement speed, and behavior. Civilians will wander or flee from nearby zombies, armed civilians will locate and attack zombies, and zombies will pursue and infect humans.

With 10,000 agents, checking every other agent during every simulation tick becomes expensive. A naive all-pairs neighbor search would involve roughly 100 million comparisons per tick. Stage 2 will investigate ways to reduce unnecessary distance calculations while keeping interactions accurate.

## 2. PostgreSQL Database

Stage 1 manages the simulation's agent state in memory. For Stage 2, we plan to introduce a **PostgreSQL database** to organize and store simulation data.

Potential stored data includes:
- Agent identifiers and types (civilian, armed civilian, or zombie)
- Agent positions, health, and alive/infected status
- Simulation tick numbers and population counts
- Events such as attacks, bites, deaths, and transformations

The database will make it easier to save results, inspect how the outbreak changes over time, and compare simulation runs. We will need to decide which data should remain in memory during each tick and which data should be written to PostgreSQL so database operations do not slow down the simulation.

## 3. Parallel Processing and Performance

Stage 2 will explore parallel techniques for two of the most computationally expensive parts of the simulation:

**Neighbor finding:** Agents need to identify nearby zombies or humans to determine whether they should flee, chase, or attack. We plan to investigate spatial partitioning (such as a uniform grid) to narrow the search to nearby areas, along with parallel processing where appropriate.

**Event processing:** Many agents may attack, move, or change state during the same tick. We plan to explore processing independent calculations concurrently, then applying results in a controlled update step to avoid conflicting changes to the same agent.

Because agents interact with one another, parallel updates must be designed carefully. For example, two zombies might target the same civilian during a tick. The simulation must resolve damage and transformations consistently without double-counting events.

## 4. Testing and Evaluation

We plan to compare Stage 2 with the Stage 1 prototype and evaluate:

- **Performance:** Time taken to process each simulation tick
- **Scalability:** How runtime changes as the number of agents increases
- **Correctness:** Whether movement, attacks, infection, and population counts remain consistent
- **Database overhead:** The impact of recording simulation data
- **Parallel speedup:** Whether parallel processing improves runtime compared with a sequential approach

Testing at multiple population sizes will help identify bottlenecks before running the full 10,000-agent simulation.

## Expected Outcome

By the end of Stage 2, our goal is to have a larger, more efficient zombie outbreak simulation that can handle thousands of interacting agents, record useful data in PostgreSQL, and demonstrate the benefits and limitations of parallel processing.
