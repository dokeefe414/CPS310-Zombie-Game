"""Zombie crowd simulation with multiprocessing-assisted event processing.

Architecture
------------
* Agents represent civilians, armed civilians, and zombies.
* SpatialGrid organizes nearby agents into buckets to avoid searching everyone.
* Separate CPU worker processes locate nearby targets using read-only snapshots.
* The main process applies movement, damage, and zombie conversions in order.
* Tkinter GUI rendering must remain on the main process.

To run: python Zombie_Game.py
Requires Python 3.10+ and tkinter. No external Python packages are needed.
Set use_multiprocessing=False when creating SimulationPlayground to compare
against the original single-process event-processing approach.

Note: Inter-process data transfer has overhead, so multiprocessing is not
guaranteed to make every simulation or computer faster.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import os
import random
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor


def _snapshot_grid(agents, cell_size):
    """Create safe, serializable copies of agent positions for worker processes.

    Each target is represented by (x, y, alive). Bucket values contain list
    indexes rather than actual Agent objects. This prevents workers from
    accidentally modifying the live game state.
    """
    targets = [(agent.x, agent.y, agent.alive) for agent in agents]
    buckets = defaultdict(list)
    for index, (x, y, _) in enumerate(targets):
        buckets[(int(x // cell_size), int(y // cell_size))].append(index)
    return targets, dict(buckets)


def _nearest_targets(task):
    """Find the nearest in-range target for every query in one CPU work chunk.

    Workers receive only plain Python data: query coordinates, target
    snapshots and grid buckets. A result of -1 means no target was found.
    The function is defined at module scope so Windows multiprocessing can
    import (pickle) it when spawning a child process.
    """
    queries, targets, buckets, cell_size = task
    results = []

    for x, y, radius in queries:
        best_index = -1
        best_distance = radius

        # Inspect only buckets that overlap this agent's search area.
        min_cx = int((x - radius) // cell_size)
        max_cx = int((x + radius) // cell_size)
        min_cy = int((y - radius) // cell_size)
        max_cy = int((y + radius) // cell_size)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                for index in buckets.get((cx, cy), ()):
                    tx, ty, alive = targets[index]
                    if not alive:
                        continue
                    distance = math.hypot(x - tx, y - ty)
                    if distance < best_distance:
                        best_index = index
                        best_distance = distance
        results.append(best_index)
    return results


class SpatialGrid:
    """Group agents into square cells to reduce nearby-entity searches.

    A query checks only grid cells inside the specified radius' bounding box.
    The returned list is a set of candidates: the caller still checks exact
    Euclidean distances. Work depends on local crowd density, not strictly O(1).
    """

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list["Agent"]] = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def insert(self, agent: "Agent"):
        # Floor division converts a world coordinate into a cell index.
        cx = int(agent.x // self.cell_size)
        cy = int(agent.y // self.cell_size)
        self.grid[(cx, cy)].append(agent)

    def query_radius(self, x: float, y: float, radius: float) -> list["Agent"]:
        """Returns candidate entities from nearby grid buckets."""
        min_cx = int((x - radius) // self.cell_size)
        max_cx = int((x + radius) // self.cell_size)
        min_cy = int((y - radius) // self.cell_size)
        max_cy = int((y + radius) // self.cell_size)

        candidates = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = self.grid.get((cx, cy))
                if cell:
                    candidates.extend(cell)
        return candidates


class Agent(ABC):
    """Shared position, motion, health and alive/dead behavior for agents.

    ABC means subclasses must supply their own update() decision logic.
    Coordinates are floating-point distances within the arena boundaries.
    """

    def __init__(self, x: float, y: float, speed: float, max_health: float = 100.0):
        self.x = x
        self.y = y
        self.speed = speed
        self.health = max_health
        self.max_health = max_health
        self.alive = True

    def distance_to(self, other: "Agent") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def move_towards(self, target_x: float, target_y: float, arena_w: float, arena_h: float):
        # Normalize the direction vector so speed is the distance per tick.
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        self.clamp_position(arena_w, arena_h)

    def move_away_from(self, threat_x: float, threat_y: float, arena_w: float, arena_h: float):
        # Reverse the vector to move away instead of towards the threat.
        dx = self.x - threat_x
        dy = self.y - threat_y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        else:
            self.x += random.uniform(-1.0, 1.0) * self.speed
            self.y += random.uniform(-1.0, 1.0) * self.speed
        self.clamp_position(arena_w, arena_h)

    def clamp_position(self, arena_w: float, arena_h: float):
        self.x = max(0.0, min(self.x, arena_w))
        self.y = max(0.0, min(self.y, arena_h))

    def take_damage(self, amount: float):
        # Keeping alive=False lets the simulation remove defeated agents.
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    @abstractmethod
    def update(self, arena: "SimulationPlayground"):
        pass


class Human(Agent):
    """Unarmed civilian: flee from a nearby zombie, otherwise wander randomly."""

    def __init__(self, x: float, y: float, speed: float = 2.4):
        super().__init__(x=x, y=y, speed=speed)
        self.perception_radius = 25.0

    def get_nearest_zombie(self, arena: "SimulationPlayground"):
        # A parallel worker may already have located this tick's target.
        # Reuse that snapshot result instead of repeating the distance search.
        if hasattr(self, "_parallel_target"):
            target = self._parallel_target
            if target is not None and target.alive:
                distance = self.distance_to(target)
                if distance < self.perception_radius:
                    return target, distance
            return None, self.perception_radius

        candidates = arena.zombie_grid.query_radius(self.x, self.y, self.perception_radius)
        nearest = None
        min_dist = self.perception_radius
        for zombie in candidates:
            if not zombie.alive:
                continue
            dist = self.distance_to(zombie)
            if dist < min_dist:
                min_dist = dist
                nearest = zombie
        return nearest, min_dist

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        nearest_zombie, min_dist = self.get_nearest_zombie(arena)

        # Civilians move away from danger; otherwise they wander slowly.
        if nearest_zombie:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height)
        else:
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.clamp_position(arena.width, arena.height)


class ArmedHuman(Human):
    """Civilian with ranged attacks and a cooldown between shots.

    The attack radius is larger than the usual fleeing/perception radius.
    A shot changes zombie health only in the main process, avoiding races.
    """

    def __init__(self, x: float, y: float, speed: float = 2.1):
        super().__init__(x=x, y=y, speed=speed)
        self.attack_range = 35.0
        self.shot_damage = 45.0
        self.cooldown_max = 6
        self.cooldown_timer = random.randint(0, self.cooldown_max)

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # Use the precomputed parallel result when enabled, otherwise retain
        # the original serial nearest-target search as a fallback.
        if hasattr(self, "_parallel_target"):
            nearest_zombie = self._parallel_target
            min_dist = (self.distance_to(nearest_zombie)
                        if nearest_zombie and nearest_zombie.alive else self.attack_range)
            if min_dist >= self.attack_range:
                nearest_zombie = None
        else:
            candidates = arena.zombie_grid.query_radius(self.x, self.y, self.attack_range)
            nearest_zombie = None
            min_dist = self.attack_range
            for zombie in candidates:
                if not zombie.alive:
                    continue
                dist = self.distance_to(zombie)
                if dist < min_dist:
                    min_dist = dist
                    nearest_zombie = zombie

        # Damage and cooldown updates happen sequentially in main process.
        if nearest_zombie and self.cooldown_timer == 0:
            nearest_zombie.take_damage(self.shot_damage)
            self.cooldown_timer = self.cooldown_max

        if nearest_zombie and min_dist <= self.perception_radius:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height)
        else:
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.clamp_position(arena.width, arena.height)


class Zombie(Agent):
    """Zombie: lose a little health per tick, chase civilians and bite.

    When an attack defeats a human, the arena queues a new zombie. Queuing
    avoids modifying the zombie list while it is being iterated over.
    """

    def __init__(self, x: float, y: float, speed: float = 2.1, bite_damage: float = 40.0):
        super().__init__(x=x, y=y, speed=speed)
        self.bite_range = 3.2
        self.bite_damage = bite_damage
        self.hunt_radius = 120

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        # Starvation/decay applies even when no human is nearby.
        self.take_damage(0.075)
        if not self.alive:
            return

        # Workers search against the start-of-tick world snapshot.
        # An agent might have died since then, so recheck alive here.
        if hasattr(self, "_parallel_target"):
            nearest_human = self._parallel_target
            min_dist = (self.distance_to(nearest_human)
                        if nearest_human and nearest_human.alive else self.hunt_radius)
            if min_dist >= self.hunt_radius:
                nearest_human = None
        else:
            candidates = arena.human_grid.query_radius(self.x, self.y, self.hunt_radius)
            nearest_human = None
            min_dist = self.hunt_radius
            for human in candidates:
                if not human.alive:
                    continue
                dist = self.distance_to(human)
                if dist < min_dist:
                    min_dist = dist
                    nearest_human = human

        if nearest_human:
            self.move_towards(nearest_human.x, nearest_human.y, arena.width, arena.height)
            if min_dist <= self.bite_range:
                self.bite(nearest_human, arena)
        else:
            self.x += random.uniform(-1.0, 1.0) * self.speed
            self.y += random.uniform(-1.0, 1.0) * self.speed
            self.clamp_position(arena.width, arena.height)

    def bite(self, human: Human, arena: "SimulationPlayground"):
        # Bites, healing, and conversions mutate shared state only here.
        human.take_damage(self.bite_damage)
        self.health = min(self.max_health, self.health + 18.0)
        if not human.alive:
            arena.convert_human_to_zombie(human)


class SimulationPlayground:
    """Own the live agents and coordinate one complete simulation tick.

    Multiprocessing is restricted to proximity calculations. Workers operate
    on *copies* of positions, not shared mutable Agent instances. After
    gathering results, the parent applies attacks and movement in order.
    This preserves clear ownership of game state and Tkinter UI objects.
    """

    def __init__(self, width: float = 1200.0, height: float = 1200.0,
                 use_multiprocessing: bool = True, workers: int | None = None):
        self.width = width
        self.height = height
        self.humans: list[Human] = []
        self.zombies: list[Zombie] = []
        self.new_zombies: list[Zombie] = []
        self.tick_count = 0

        # Keep one reusable process pool, rather than spawning every tick.
        # Set workers=1 or use_multiprocessing=False to run in serial mode.
        self.use_multiprocessing = use_multiprocessing
        self.workers = max(1, workers or min(4, os.cpu_count() or 1))
        self._executor = None

        self.human_grid = SpatialGrid(cell_size=80.0)
        self.zombie_grid = SpatialGrid(cell_size=80.0)

    def populate(self, num_regular: int, num_armed: int, num_zombies: int):
        """Place the three agent types at random arena coordinates."""
        for _ in range(num_regular):
            self.humans.append(Human(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_armed):
            self.humans.append(ArmedHuman(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_zombies):
            self.zombies.append(Zombie(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))

    def convert_human_to_zombie(self, human: Human):
        """Queue a conversion for the end of the current simulation step."""
        self.new_zombies.append(Zombie(x=human.x, y=human.y))

    def rebuild_spatial_grids(self):
        """Rebuild both spatial indexes from current agent positions."""
        self.human_grid.clear()
        self.zombie_grid.clear()
        for h in self.humans:
            self.human_grid.insert(h)
        for z in self.zombies:
            self.zombie_grid.insert(z)

    def _parallel_target_search(self):
        """Dispatch nearby-target queries to multiple worker processes.

        Each worker gets independent, read-only target data. Chunking the
        queries reduces the number of tasks and serialization overhead.
        Returned indexes are mapped to original live agents in the parent.
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)

        cell_size = self.human_grid.cell_size
        humans_snapshot, human_buckets = _snapshot_grid(self.humans, cell_size)
        zombies_snapshot, zombie_buckets = _snapshot_grid(self.zombies, cell_size)

        human_queries = [
            (h.x, h.y, h.attack_range if isinstance(h, ArmedHuman) else h.perception_radius)
            for h in self.humans
        ]
        zombie_queries = [(z.x, z.y, z.hunt_radius) for z in self.zombies]

        tasks = []
        assignments = []
        for agents, queries, targets, buckets in (
            (self.humans, human_queries, zombies_snapshot, zombie_buckets),
            (self.zombies, zombie_queries, humans_snapshot, human_buckets),
        ):
            chunk_size = max(128, (len(queries) + self.workers * 2 - 1) // (self.workers * 2))
            for start in range(0, len(queries), chunk_size):
                tasks.append((queries[start:start + chunk_size], targets, buckets, cell_size))
                assignments.append((agents, start, min(len(queries), start + chunk_size)))

        # executor.map keeps results ordered to match our assignments.
        for (agents, start, end), indexes in zip(
            assignments, self._executor.map(_nearest_targets, tasks)
        ):
            target_agents = self.zombies if agents is self.humans else self.humans
            for agent, index in zip(agents[start:end], indexes):
                agent._parallel_target = target_agents[index] if index >= 0 else None

    def close(self):
        """Cleanly stop child processes, including when the window closes."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def step(self):
        """Advance one tick: snapshot/search, act, convert, then clean up."""
        self.tick_count += 1
        self.new_zombies.clear()

        self.rebuild_spatial_grids()

        # Parallel work only decides who is close to whom. The actual
        # event processing remains sequential to avoid concurrent writes.
        if self.use_multiprocessing and self.workers > 1:
            self._parallel_target_search()

        # Humans act first, then zombies, as in the original game.
        for human in self.humans:
            human.update(self)

        for zombie in self.zombies:
            zombie.update(self)

        # Remove transient target links before refreshing the next snapshot.
        for agent in (*self.humans, *self.zombies):
            if hasattr(agent, "_parallel_target"):
                del agent._parallel_target

        # Append conversions after iteration and remove dead entities.
        self.zombies.extend(self.new_zombies)
        self.humans = [h for h in self.humans if h.alive]
        self.zombies = [z for z in self.zombies if z.alive]


class SimulationGUI:
    """Draw the simulation and population chart using Tkinter's event loop.

    Tkinter widgets are never passed to subprocesses. root.after() calls
    update_loop periodically instead of blocking the GUI with a while loop.
    """

    def __init__(self, root, sim: SimulationPlayground, scale: float = 0.52):
        self.root = root
        self.sim = sim
        self.scale = scale
        self.arena_px = int(sim.width * scale)  # ~624 px
        self.graph_w = 360
        self.graph_h = self.arena_px

        self.root.title("Zombie Game Simulation - 10,000 Agents")
        self.root.configure(bg="#1E1E1E")

        self.status_label = tk.Label(
            root, text="", font=("Consolas", 11, "bold"), fg="#FFFFFF", bg="#1E1E1E", pady=6
        )
        self.status_label.pack()

        panes_frame = tk.Frame(root, bg="#1E1E1E")
        panes_frame.pack(padx=10, pady=5)

        self.arena_canvas = tk.Canvas(
            panes_frame, width=self.arena_px, height=self.arena_px, bg="#101010", highlightthickness=1
        )
        self.arena_canvas.grid(row=0, column=0, padx=6)

        self.graph_canvas = tk.Canvas(
            panes_frame, width=self.graph_w, height=self.graph_h, bg="#161616", highlightthickness=1
        )
        self.graph_canvas.grid(row=0, column=1, padx=6)

        self.human_history = [len(sim.humans)]
        self.zombie_history = [len(sim.zombies)]
        self.total_initial = len(sim.humans) + len(sim.zombies)

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def on_close(self):
        """Shut down worker processes before destroying the GUI window."""
        self.running = False
        self.sim.close()
        self.root.destroy()

    def update_loop(self):
        """Run the simulation, record history, and redraw on each GUI tick."""
        if self.running:
            self.sim.step()
            self.human_history.append(len(self.sim.humans))
            self.zombie_history.append(len(self.sim.zombies))

            self.draw_arena()
            self.draw_graph()
            self.update_header()

            if not self.sim.humans or not self.sim.zombies:
                self.running = False
                self.sim.close()

        self.root.after(20, self.update_loop)

    def draw_arena(self):
        """Render each live agent as a small colored square."""
        self.arena_canvas.delete("all")

        # Fast 2-pixel point particle rendering for crowd density
        s = self.scale
        for human in self.sim.humans:
            px = human.x * s
            py = human.y * s
            color = "#FFD700" if isinstance(human, ArmedHuman) else "#00D2FF"
            self.arena_canvas.create_rectangle(px, py, px + 2.0, py + 2.0, fill=color, outline="")

        for zombie in self.sim.zombies:
            px = zombie.x * s
            py = zombie.y * s
            self.arena_canvas.create_rectangle(px, py, px + 2.5, py + 2.5, fill="#44FF44", outline="")

    def draw_graph(self):
        """Plot civilian and zombie counts recorded after each tick."""
        self.graph_canvas.delete("all")
        margin = 35
        gw = self.graph_w - margin * 2
        gh = self.graph_h - margin * 2

        self.graph_canvas.create_line(margin, margin, margin, margin + gh, fill="#555555")
        self.graph_canvas.create_line(margin, margin + gh, margin + gw, margin + gh, fill="#555555")
        self.graph_canvas.create_text(
            margin + gw // 2, margin - 15, text="Live Population Dynamics", fill="#FFFFFF", font=("Consolas", 10, "bold")
        )

        pts = len(self.human_history)
        if pts < 2:
            return

        max_pop = max(self.total_initial, max(self.human_history), max(self.zombie_history))
        dx = gw / max(pts - 1, 30)

        # Stride sampling keeps graph rendering instant even over long runs
        human_coords, zombie_coords = [], []
        step_stride = max(1, pts // 250)
        for i in range(0, pts, step_stride):
            px = margin + i * dx
            hy = (margin + gh) - (self.human_history[i] / max_pop) * gh
            zy = (margin + gh) - (self.zombie_history[i] / max_pop) * gh
            human_coords.extend([px, hy])
            zombie_coords.extend([px, zy])

        if len(human_coords) >= 4:
            self.graph_canvas.create_line(*human_coords, fill="#00D2FF", width=2)
            self.graph_canvas.create_line(*zombie_coords, fill="#44FF44", width=2)

    def update_header(self):
        """Display counts and the game-over result at the top of the window."""
        unarmed = sum(1 for h in self.sim.humans if not isinstance(h, ArmedHuman))
        armed = sum(1 for h in self.sim.humans if isinstance(h, ArmedHuman))
        zombies = len(self.sim.zombies)

        status = f"Step: {self.sim.tick_count:03d} | Civilians: {unarmed} | Armed: {armed} | Zombies: {zombies} | Total: {unarmed+armed+zombies}"
        if not self.sim.humans:
            status += "  [EXTINCTION: Zombies Win!]"
        elif not self.sim.zombies:
            status += "  [CONTAINED: Humans Win!]"

        self.status_label.config(text=status)


# This guard is essential on Windows: multiprocessing's spawn method imports
# the module in each child process. Without the guard, every child could start
# another Tkinter window and recursively launch more processes.
if __name__ == "__main__":
    # 10,000 Agents Setup
    playground = SimulationPlayground(
        width=1200.0, height=1200.0, use_multiprocessing=True, workers=4
    )

    # 7,500 Civilians, 500 Armed, 2,000 Zombies = 10,000 Total
    playground.populate(num_regular=7500, num_armed=500, num_zombies=2000)

    app = tk.Tk()
    gui = SimulationGUI(app, playground, scale=0.52)
    app.mainloop()