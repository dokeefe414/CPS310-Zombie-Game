# CPS310-Zombie-Game

# Contributors

Daniel O'Keefe, Owen Bowers, Rahsaun Jeffrey
## About

This project is an agent-based simulation of a zombie outbreak on a square
playground. We have three types of agents, humans, armed humans, and zombies, that move around and interact:
zombies chase and bite humans, a bite infects the human, and the human mutates 
into a zombie right as its health turns to zero.

Every agent has a life value (0-100) and a movement speed (Human is 2.2, Armed Human 2.0 and zombie is 1.65). 
When humans are outside the perception radius (28 units) of a zombie they wander at 1.1 speed.
When a zombie enters the humans perception radius (28 units) the human moves away at their full base speed (2.2 or 2.0). 
The armed human has 40 damage per shot with a cooldown of 8 ticks.
Zombies also have a deadlines of 35 damage per bite and the zombie gains 10 health for every bite of a human.


This is the Stage 1 prototype for CPS310, built as our first assignment.
