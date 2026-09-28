from abc import ABC, abstractmethod
import math
import random


class Agent(ABC):
    """Abstract base class representing an entity in 2D bounded Euclidean space."""

    def __init__(
        self, x: float, y: float, speed: float, max_health: float = 100.0
    ):
        self.x = x
        self.y = y
        self.speed = speed
        self.health = max_health
        self.max_health = max_health
        self.alive = True

    def distance_to(self, other: "Agent") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def move_towards(
        self,
        target_x: float,
        target_y: float,
        arena_width: float,
        arena_height: float,
    ):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        self.clamp_position(arena_width, arena_height)

    def move_away_from(
        self,
        threat_x: float,
        threat_y: float,
        arena_width: float,
        arena_height: float,
    ):
        dx = self.x - threat_x
        dy = self.y - threat_y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        else:
            # If coordinates completely overlap, scatter randomly
            self.x += random.uniform(-1.0, 1.0) * self.speed
            self.y += random.uniform(-1.0, 1.0) * self.speed
        self.clamp_position(arena_width, arena_height)

    def clamp_position(self, arena_width: float, arena_height: float):
        self.x = max(0.0, min(self.x, arena_width))
        self.y = max(0.0, min(self.y, arena_height))

    def take_damage(self, amount: float):
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    @abstractmethod
    def update(self, arena: "SimulationPlayground"):
        """Subclasses must define their movement and interaction behavior."""
        pass


class Human(Agent):
    """Humans flee from the nearest zombie within perception radius, or wander randomly."""

    def __init__(self, x: float, y: float, speed: float = 2.2):
        super().__init__(x=x, y=y, speed=speed)
        self.perception_radius = 25.0

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        nearest_zombie = None
        min_dist = float("inf")
        for zombie in arena.zombies:
            if not zombie.alive:
                continue
            dist = self.distance_to(zombie)
            if dist < min_dist:
                min_dist = dist
                nearest_zombie = zombie

        if nearest_zombie and min_dist <= self.perception_radius:
            self.move_away_from(
                nearest_zombie.x, nearest_zombie.y, arena.width, arena.height
            )
        else:
            # Slow wander when safe
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.5)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.5)
            self.clamp_position(arena.width, arena.height)


class Zombie(Agent):
    """Zombies chase nearest living humans, deal bite damage, and suffer small hunger decay."""

    def __init__(
        self, x: float, y: float, speed: float = 1.6, bite_damage: float = 35.0
    ):
        super().__init__(x=x, y=y, speed=speed)
        self.bite_range = 2.5
        self.bite_damage = bite_damage

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        # Gradual hunger/starvation decay
        self.take_damage(0.1)
        if not self.alive:
            return

        nearest_human = None
        min_dist = float("inf")
        for human in arena.humans:
            if not human.alive:
                continue
            dist = self.distance_to(human)
            if dist < min_dist:
                min_dist = dist
                nearest_human = human

        if nearest_human:
            self.move_towards(
                nearest_human.x, nearest_human.y, arena.width, arena.height
            )
            if self.distance_to(nearest_human) <= self.bite_range:
                self.bite(nearest_human, arena)
        else:
            # Shuffle randomly if no living humans exist
            self.x += random.uniform(-1.0, 1.0) * self.speed
            self.y += random.uniform(-1.0, 1.0) * self.speed
            self.clamp_position(arena.width, arena.height)

    def bite(self, human: Human, arena: "SimulationPlayground"):
        human.take_damage(self.bite_damage)
        self.health = min(self.max_health, self.health + 10.0)  # Feeding heals

        # Reanimation dynamic
        if not human.alive:
            arena.convert_human_to_zombie(human)


class SimulationPlayground:
    """Bounded 2D Euclidean space orchestrating agent updates and state transitions."""

    def __init__(self, width: float = 150.0, height: float = 150.0):
        self.width = width
        self.height = height
        self.humans: list[Human] = []
        self.zombies: list[Zombie] = []
        self.new_zombies: list[Zombie] = []  # Buffer for conversions
        self.tick_count = 0

    def populate(self, num_humans: int, num_zombies: int):
        for _ in range(num_humans):
            self.humans.append(
                Human(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, self.height),
                )
            )
        for _ in range(num_zombies):
            self.zombies.append(
                Zombie(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, self.height),
                )
            )

    def convert_human_to_zombie(self, human: Human):
        # Buffer newly turned zombies so lists aren't modified during active loops
        self.new_zombies.append(Zombie(x=human.x, y=human.y))

    def step(self):
        self.tick_count += 1
        self.new_zombies.clear()

        # Update humans (iterating over a shallow copy is safe against state changes)
        for human in list(self.humans):
            human.update(self)

        # Update zombies
        for zombie in list(self.zombies):
            zombie.update(self)

        # Safely integrate newly converted zombies into main pool
        self.zombies.extend(self.new_zombies)

        # Cull dead entities
        self.humans = [h for h in self.humans if h.alive]
        self.zombies = [z for z in self.zombies if z.alive]

    def display_metrics(self):
        avg_human_hp = (
            sum(h.health for h in self.humans) / len(self.humans)
            if self.humans
            else 0.0
        )
        avg_zombie_hp = (
            sum(z.health for z in self.zombies) / len(self.zombies)
            if self.zombies
            else 0.0
        )
        print(
            f"Step {self.tick_count:03d} | "
            f"Humans: {len(self.humans):2d} (Avg HP: {avg_human_hp:5.1f}) | "
            f"Zombies: {len(self.zombies):2d} (Avg HP: {avg_zombie_hp:5.1f})"
        )


if __name__ == "__main__":
    random.seed(42)

    # Initial setup
    sim = SimulationPlayground(width=150.0, height=150.0)
    sim.populate(num_humans=30, num_zombies=4)

    print("=== Zombie Game Simulation: Stage 1 ===")
    print(f"Initial: {len(sim.humans)} Humans vs {len(sim.zombies)} Zombies\n")

    max_steps = 10000
    for _ in range(max_steps):
        sim.step()
        sim.display_metrics()

        if not sim.humans:
            print("\n--- Outbreak Complete: Humans went extinct! ---")
            break
        if not sim.zombies:
            print("\n--- Outbreak Contained: All zombies decayed! ---")
            break
    else:
        print(f"\n--- Simulation finished max steps ({max_steps}) ---")