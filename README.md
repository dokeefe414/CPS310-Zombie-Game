# CPS310-Zombie-Game

# Contributors

Daniel O'Keefe, Owen Bowers, Rahsaun Jeffrey
## About

This project is an agent-based simulation of a zombie outbreak on a square
playground. Two types of agents, humans and zombies, move around and interact:
zombies chase and bite humans, a bite infects the human, and an infected human
either dies or mutates into a zombie after a set mutation period.

Every agent has a life value (0-100) and a movement speed. Zombies also have a
deadliness (damage per bite), and humans have a mutation period. Humans can be
healthy, infected, or dead.

To produce non-trivial population dynamics, the model adds [human births with a
carrying capacity] and [zombie life decay that bites replenish], so human and
zombie populations rise and fall over time instead of only declining. Agent
data and per-tick population counts are stored in PostgreSQL, and results are
plotted from that data.

This is the Stage 1 prototype for CPS310, built as our first assignment.
