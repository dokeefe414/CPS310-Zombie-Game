"""
Zombie Crowd Simulation - Swarm Intelligence & PostgreSQL Database Persistence.
Integrates Linear Thompson Sampling swarm policy, dynamic human injury drag,
accessible red color palette, boundary repulsion margins, and PostgreSQL batch persistence.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import os
import random
import time
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from db_manager import DatabaseManager
from models import LinearThompsonSwarmLearner
from forecasting import SlidingWindowAdaptiveKalman, ForecastTracker


def _snapshot_grid(agents, cell_size):
    targets = [(agent.x, agent.y, agent.alive) for agent in agents]
    buckets = defaultdict(list)
    for index, (x, y, _) in enumerate(targets):
        buckets[(int(x // cell_size), int(y // cell_size))].append(index)
    return targets, dict(buckets)


def _nearest_targets(task):
    queries, targets, buckets, cell_size = task
    results = []

    for x, y, radius in queries:
        best_index = -1
        best_distance = radius

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
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list["Agent"]] = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def insert(self, agent: "Agent"):
        cx = int(agent.x // self.cell_size)
        cy = int(agent.y // self.cell_size)
        self.grid[(cx, cy)].append(agent)

    def query_radius(self, x: float, y: float, radius: float) -> list["Agent"]:
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
    def __init__(self, x: float, y: float, base_speed: float, max_health: float = 100.0):
        self.x = x
        self.y = y
        self.base_speed = base_speed
        self.health = max_health
        self.max_health = max_health
        self.alive = True

    def distance_to(self, other: "Agent") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def apply_boundary_repulsion(self, arena_w: float, arena_h: float, margin: float = 80.0) -> tuple[float, float]:
        """Calculates steering vector away from walls to prevent corner hoarding."""
        fx, fy = 0.0, 0.0
        if self.x < margin:
            fx += (margin - self.x) / margin
        elif self.x > arena_w - margin:
            fx -= (self.x - (arena_w - margin)) / margin

        if self.y < margin:
            fy += (margin - self.y) / margin
        elif self.y > arena_h - margin:
            fy -= (self.y - (arena_h - margin)) / margin

        return fx, fy

    def move_towards(self, target_x: float, target_y: float, arena_w: float, arena_h: float, speed: float):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        
        dir_x = (dx / dist) if dist > 0 else 0.0
        dir_y = (dy / dist) if dist > 0 else 0.0

        # Blend target direction with wall repulsion
        r_x, r_y = self.apply_boundary_repulsion(arena_w, arena_h)
        total_x = dir_x + 1.2 * r_x
        total_y = dir_y + 1.2 * r_y
        total_len = math.hypot(total_x, total_y)

        if total_len > 0:
            self.x += (total_x / total_len) * speed
            self.y += (total_y / total_len) * speed

        self.x = max(15.0, min(self.x, arena_w - 15.0))
        self.y = max(15.0, min(self.y, arena_h - 15.0))

    def move_away_from(self, threat_x: float, threat_y: float, arena_w: float, arena_h: float, speed: float):
        dx = self.x - threat_x
        dy = self.y - threat_y
        dist = math.hypot(dx, dy)

        dir_x = (dx / dist) if dist > 0 else random.uniform(-1.0, 1.0)
        dir_y = (dy / dist) if dist > 0 else random.uniform(-1.0, 1.0)

        # Blend flee vector with wall avoidance so humans don't trap themselves in corners
        r_x, r_y = self.apply_boundary_repulsion(arena_w, arena_h)
        total_x = dir_x + 1.8 * r_x
        total_y = dir_y + 1.8 * r_y
        total_len = math.hypot(total_x, total_y)

        if total_len > 0:
            self.x += (total_x / total_len) * speed
            self.y += (total_y / total_len) * speed

        self.x = max(15.0, min(self.x, arena_w - 15.0))
        self.y = max(15.0, min(self.y, arena_h - 15.0))

    def take_damage(self, amount: float):
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    @abstractmethod
    def update(self, arena: "SimulationPlayground"):
        pass


class Human(Agent):
    def __init__(self, x: float, y: float, base_speed: float = 2.4):
        super().__init__(x=x, y=y, base_speed=base_speed, max_health=100.0)
        self.perception_radius = 28.0

    def get_current_speed(self) -> float:
        health_ratio = max(0.0, self.health / self.max_health)
        speed_factor = 0.65 + (0.35 * health_ratio)
        return self.base_speed * speed_factor

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        arena.deposit_scent(self.x, self.y, amount=1.2)

        current_speed = self.get_current_speed()
        nearest_zombie = None
        min_dist = self.perception_radius

        if hasattr(self, "_parallel_target"):
            target = self._parallel_target
            if target is not None and target.alive:
                dist = self.distance_to(target)
                if dist < self.perception_radius:
                    nearest_zombie = target
                    min_dist = dist
        else:
            for zombie in arena.zombie_grid.query_radius(self.x, self.y, self.perception_radius):
                if not zombie.alive:
                    continue
                dist = self.distance_to(zombie)
                if dist < min_dist:
                    min_dist = dist
                    nearest_zombie = zombie

        if nearest_zombie:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height, current_speed)
        else:
            r_x, r_y = self.apply_boundary_repulsion(arena.width, arena.height)
            self.x += (random.uniform(-1.0, 1.0) * (current_speed * 0.4)) + (r_x * 0.8)
            self.y += (random.uniform(-1.0, 1.0) * (current_speed * 0.4)) + (r_y * 0.8)
            self.x = max(15.0, min(self.x, arena.width - 15.0))
            self.y = max(15.0, min(self.y, arena.height - 15.0))


class ArmedHuman(Human):
    def __init__(self, x: float, y: float, base_speed: float = 2.1):
        super().__init__(x=x, y=y, base_speed=base_speed)
        self.attack_range = 35.0
        self.shot_damage = 42.0
        self.cooldown_max = 7
        self.cooldown_timer = random.randint(0, self.cooldown_max)

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        arena.deposit_scent(self.x, self.y, amount=1.2)
        current_speed = self.get_current_speed()
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        nearest_zombie = None
        min_dist = self.attack_range

        if hasattr(self, "_parallel_target"):
            target = self._parallel_target
            if target is not None and target.alive:
                dist = self.distance_to(target)
                if dist <= self.attack_range:
                    nearest_zombie = target
                    min_dist = dist
        else:
            for zombie in arena.zombie_grid.query_radius(self.x, self.y, self.attack_range):
                if not zombie.alive:
                    continue
                dist = self.distance_to(zombie)
                if dist < min_dist:
                    min_dist = dist
                    nearest_zombie = zombie

        if nearest_zombie and self.cooldown_timer == 0:
            nearest_zombie.take_damage(self.shot_damage)
            if hasattr(nearest_zombie, "last_action") and hasattr(nearest_zombie, "last_context"):
                arena.swarm_learner.update(nearest_zombie.last_action, nearest_zombie.last_context, -1.0)
            self.cooldown_timer = self.cooldown_max

        if nearest_zombie and min_dist <= self.perception_radius:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height, current_speed)
        else:
            r_x, r_y = self.apply_boundary_repulsion(arena.width, arena.height)
            self.x += (random.uniform(-1.0, 1.0) * (current_speed * 0.4)) + (r_x * 0.8)
            self.y += (random.uniform(-1.0, 1.0) * (current_speed * 0.4)) + (r_y * 0.8)
            self.x = max(15.0, min(self.x, arena.width - 15.0))
            self.y = max(15.0, min(self.y, arena.height - 15.0))


class Zombie(Agent):
    def __init__(self, x: float, y: float, base_speed: float = 2.28, bite_damage: float = 52.0):
        super().__init__(x=x, y=y, base_speed=base_speed, max_health=130.0)
        self.bite_range = 3.8
        self.bite_damage = bite_damage
        self.hunt_radius = 140.0
        self.last_action = 0
        self.last_context = np.zeros(4, dtype=np.float64)

    def extract_context(self, arena: "SimulationPlayground", nearest_human: Human, min_dist: float) -> np.ndarray:
        norm_dist = min_dist / self.hunt_radius if nearest_human else 1.0
        local_agents = arena.human_grid.query_radius(self.x, self.y, 50.0)
        armed_cnt = sum(1 for a in local_agents if isinstance(a, ArmedHuman))
        armed_density = (armed_cnt / len(local_agents)) if local_agents else 0.0
        local_zombies = len(arena.zombie_grid.query_radius(self.x, self.y, 50.0))
        norm_allies = min(1.0, local_zombies / 6.0)
        return np.array([norm_dist, armed_density, norm_allies, 1.0], dtype=np.float64)

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        self.take_damage(0.012)
        if not self.alive:
            return

        nearest_human = None
        min_dist = self.hunt_radius

        if hasattr(self, "_parallel_target"):
            target = self._parallel_target
            if target is not None and target.alive:
                dist = self.distance_to(target)
                if dist <= self.hunt_radius:
                    nearest_human = target
                    min_dist = dist
        else:
            for human in arena.human_grid.query_radius(self.x, self.y, self.hunt_radius):
                if not human.alive:
                    continue
                dist = self.distance_to(human)
                if dist < min_dist:
                    min_dist = dist
                    nearest_human = human

        # --- PURSUIT VIA THOMPSON SAMPLING ---
        if nearest_human:
            ctx = self.extract_context(arena, nearest_human, min_dist)
            action = (0 if arena.strategy_mode == "rush_only" else
                      arena.swarm_learner.select_action(ctx))
            self.last_action = action
            self.last_context = ctx

            target_x, target_y = nearest_human.x, nearest_human.y

            if action == 1:
                angle = math.atan2(target_y - self.y, target_x - self.x) + 0.65
                target_x = self.x + math.cos(angle) * min_dist
                target_y = self.y + math.sin(angle) * min_dist
            elif action == 2:
                allies = arena.zombie_grid.query_radius(self.x, self.y, 60.0)
                if len(allies) > 1:
                    avg_x = sum(a.x for a in allies) / len(allies)
                    avg_y = sum(a.y for a in allies) / len(allies)
                    target_x = 0.6 * target_x + 0.4 * avg_x
                    target_y = 0.6 * target_y + 0.4 * avg_y

            self.move_towards(target_x, target_y, arena.width, arena.height, self.base_speed)

            if self.distance_to(nearest_human) <= self.bite_range:
                self.bite(nearest_human, arena)

        # --- ACTIVE HUNTING / SCENT RECOVERY ---
        else:
            tx, ty = arena.get_strongest_scent_direction(self.x, self.y)
            if tx is not None and ty is not None:
                self.move_towards(tx, ty, arena.width, arena.height, self.base_speed)
            else:
                # If no scent nearby, step toward map center if near walls, or sweep open arena
                r_x, r_y = self.apply_boundary_repulsion(arena.width, arena.height, margin=100.0)
                if abs(r_x) > 0.05 or abs(r_y) > 0.05:
                    self.move_towards(arena.width * 0.5, arena.height * 0.5, arena.width, arena.height, self.base_speed)
                else:
                    self.x += random.uniform(-1.0, 1.0) * self.base_speed
                    self.y += random.uniform(-1.0, 1.0) * self.base_speed
                    self.x = max(15.0, min(self.x, arena.width - 15.0))
                    self.y = max(15.0, min(self.y, arena.height - 15.0))

    def bite(self, human: Human, arena: "SimulationPlayground"):
        human.take_damage(self.bite_damage)
        self.health = min(self.max_health, self.health + 25.0)

        reward = 1.2
        if not human.alive:
            reward += 3.0
            arena.convert_human_to_zombie(human)

        arena.swarm_learner.update(self.last_action, self.last_context, reward)


class SimulationPlayground:
    def __init__(self, width: float = 1200.0, height: float = 1200.0, use_multiprocessing: bool = True, workers: int | None = None):
        self.width = width
        self.height = height
        self.humans: list[Human] = []
        self.zombies: list[Zombie] = []
        self.new_zombies: list[Zombie] = []
        self.tick_count = 0
        # Research-only intervention; default preserves Owen's Thompson Sampling.
        self.strategy_mode = "thompson"

        self.initial_regular = 0
        self.initial_armed = 0
        self.initial_zombies = 0

        self.use_multiprocessing = use_multiprocessing
        self.workers = max(1, workers or min(4, os.cpu_count() or 1))
        self._executor = None

        self.human_grid = SpatialGrid(cell_size=80.0)
        self.zombie_grid = SpatialGrid(cell_size=80.0)

        # 24x24 Coarse Pheromone Scent Map (50x50 units per cell)
        self.scent_cell_size = 50.0
        self.scent_cols = int(self.width // self.scent_cell_size)
        self.scent_rows = int(self.height // self.scent_cell_size)
        self.scent_map = np.zeros((self.scent_rows, self.scent_cols), dtype=np.float32)

        self.swarm_learner = LinearThompsonSwarmLearner(n_features=4, v_squared=0.25, resample_interval=5)

    def deposit_scent(self, x: float, y: float, amount: float = 1.2):
        c = int(min(max(x // self.scent_cell_size, 0), self.scent_cols - 1))
        r = int(min(max(y // self.scent_cell_size, 0), self.scent_rows - 1))
        self.scent_map[r, c] = min(40.0, self.scent_map[r, c] + amount)

    def diffuse_and_decay_scent(self):
        """Simulates 2D scent dissipation so trails lead smoothly toward targets."""
        # Decay
        self.scent_map *= 0.94
        # 4-way box blur diffusion kernel
        diffused = (
            np.roll(self.scent_map, 1, axis=0)
            + np.roll(self.scent_map, -1, axis=0)
            + np.roll(self.scent_map, 1, axis=1)
            + np.roll(self.scent_map, -1, axis=1)
        ) * 0.08
        self.scent_map = (self.scent_map * 0.68) + diffused

    def get_strongest_scent_direction(self, x: float, y: float) -> tuple[float | None, float | None]:
        c = int(min(max(x // self.scent_cell_size, 0), self.scent_cols - 1))
        r = int(min(max(y // self.scent_cell_size, 0), self.scent_rows - 1))

        best_val = self.scent_map[r, c]
        best_target = None

        # Sample 5x5 neighborhood around zombie to find directional scent trail
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.scent_rows and 0 <= nc < self.scent_cols:
                    val = self.scent_map[nr, nc]
                    if val > best_val:
                        best_val = val
                        best_target = (
                            (nc + 0.5) * self.scent_cell_size,
                            (nr + 0.5) * self.scent_cell_size,
                        )

        if best_target and best_val > 0.15:
            return best_target
        return None, None

    def populate(self, num_regular: int, num_armed: int, num_zombies: int):
        self.initial_regular = num_regular
        self.initial_armed = num_armed
        self.initial_zombies = num_zombies

        for _ in range(num_regular):
            self.humans.append(Human(x=random.uniform(50, self.width - 50), y=random.uniform(50, self.height - 50)))
        for _ in range(num_armed):
            self.humans.append(ArmedHuman(x=random.uniform(50, self.width - 50), y=random.uniform(50, self.height - 50)))
        for _ in range(num_zombies):
            self.zombies.append(Zombie(x=random.uniform(50, self.width - 50), y=random.uniform(50, self.height - 50)))

    def convert_human_to_zombie(self, human: Human):
        self.new_zombies.append(Zombie(x=human.x, y=human.y))

    def rebuild_spatial_grids(self):
        self.human_grid.clear()
        self.zombie_grid.clear()
        for h in self.humans:
            self.human_grid.insert(h)
        for z in self.zombies:
            self.zombie_grid.insert(z)

    def _parallel_target_search(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)

        cell_size = self.human_grid.cell_size
        humans_snapshot, human_buckets = _snapshot_grid(self.humans, cell_size)
        zombies_snapshot, zombie_buckets = _snapshot_grid(self.zombies, cell_size)

        human_queries = [(h.x, h.y, h.attack_range if isinstance(h, ArmedHuman) else h.perception_radius) for h in self.humans]
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

        for (agents, start, end), indexes in zip(assignments, self._executor.map(_nearest_targets, tasks)):
            target_agents = self.zombies if agents is self.humans else self.humans
            for agent, index in zip(agents[start:end], indexes):
                agent._parallel_target = target_agents[index] if index >= 0 else None

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def step(self):
        self.tick_count += 1
        self.new_zombies.clear()

        self.diffuse_and_decay_scent()
        self.swarm_learner.step()
        self.rebuild_spatial_grids()

        if self.use_multiprocessing and self.workers > 1:
            self._parallel_target_search()

        for human in self.humans:
            human.update(self)

        for zombie in self.zombies:
            zombie.update(self)

        for agent in (*self.humans, *self.zombies):
            if hasattr(agent, "_parallel_target"):
                del agent._parallel_target

        self.zombies.extend(self.new_zombies)
        self.humans = [h for h in self.humans if h.alive]
        self.zombies = [z for z in self.zombies if z.alive]


class SimulationGUI:
    def __init__(self, root, sim: SimulationPlayground, scale: float = 0.52, db_manager: DatabaseManager = None,
                 forecast_window: int = 50, render_every: int = 4):
        self.root = root
        self.sim = sim
        self.scale = scale
        self.db_manager = db_manager
        self.arena_px = int(sim.width * scale)
        self.graph_w = 400
        self.graph_h = self.arena_px
        self.render_every = max(1, render_every)
        self.step_times = []
        self.render_times = []

        self.root.title("Zombie Game Simulation - Swarm Learning & PostgreSQL Persistence")
        self.root.configure(bg="#1E1E1E")

        self.status_label = tk.Label(root, text="", font=("Consolas", 11, "bold"), fg="#FFFFFF", bg="#1E1E1E", pady=6)
        self.status_label.pack()

        panes_frame = tk.Frame(root, bg="#1E1E1E")
        panes_frame.pack(padx=10, pady=5)

        self.arena_canvas = tk.Canvas(panes_frame, width=self.arena_px, height=self.arena_px, bg="#101010", highlightthickness=1)
        self.arena_canvas.grid(row=0, column=0, padx=6)

        self.graph_canvas = tk.Canvas(panes_frame, width=self.graph_w, height=self.graph_h, bg="#161616", highlightthickness=1)
        self.graph_canvas.grid(row=0, column=1, padx=6)

        self.human_history = [len(sim.humans)]
        self.zombie_history = [len(sim.zombies)]
        self.total_initial = len(sim.humans) + len(sim.zombies)

        # Forecasting observes the simulation; it never changes agent behavior.
        self.window_var = tk.IntVar(value=forecast_window)
        self.forecaster = SlidingWindowAdaptiveKalman(window=forecast_window, capacity=self.total_initial)
        self.forecaster.update(self.zombie_history[0])
        self.forecast_horizon = 10
        self.forecast_tracker = ForecastTracker(horizon=self.forecast_horizon)
        self.zombie_forecast = self.forecaster.predict(self.forecast_horizon)

        controls = tk.Frame(root, bg="#1E1E1E")
        controls.pack()
        tk.Label(controls, text="Sliding window:", fg="white", bg="#1E1E1E").pack(side="left")
        for window in (20, 50, 100):
            tk.Radiobutton(controls, text=str(window), variable=self.window_var, value=window,
                           command=self.change_window, fg="white", bg="#1E1E1E",
                           selectcolor="#444444").pack(side="left")
        tk.Label(controls, text="   Cyan: humans   Red: zombies   Orange: forecast",
                 fg="white", bg="#1E1E1E").pack(side="left")

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def change_window(self):
        """Replay observed counts when window changes; don't reset the simulation."""
        self.forecaster = SlidingWindowAdaptiveKalman(
            window=self.window_var.get(), capacity=self.total_initial
        )
        self.forecast_tracker = ForecastTracker(horizon=self.forecast_horizon)
        for tick, count in enumerate(self.zombie_history):
            self.forecaster.update(count)
            prediction = self.forecaster.predict(self.forecast_horizon)[-1]
            self.forecast_tracker.record(tick, count, prediction, count)
        self.zombie_forecast = self.forecaster.predict(self.forecast_horizon)
        self.draw_graph()
        self.update_header()

    def on_close(self):
        self.running = False
        self.sim.close()

        if self.db_manager and len(self.human_history) > 1:
            metadata = {
                "initial_regular": self.sim.initial_regular,
                "initial_armed": self.sim.initial_armed,
                "initial_zombies": self.sim.initial_zombies,
                "total_ticks": self.sim.tick_count,
                "outcome": "TERMINATED_EARLY (Manual Exit)",
            }
            history_records = [
                (tick, self.human_history[tick], self.zombie_history[tick])
                for tick in range(len(self.human_history))
            ]
            self.db_manager.save_run(metadata, history_records)

        self.root.destroy()

    def update_loop(self):
        if self.running:
            started = time.perf_counter()
            self.sim.step()
            self.step_times.append(time.perf_counter() - started)
            if len(self.step_times) > 100:
                self.step_times.pop(0)
            self.human_history.append(len(self.sim.humans))
            self.zombie_history.append(len(self.sim.zombies))
            self.forecaster.update(self.zombie_history[-1])
            self.zombie_forecast = self.forecaster.predict(self.forecast_horizon)
            self.forecast_tracker.record(
                self.sim.tick_count, self.zombie_history[-1],
                self.zombie_forecast[-1], self.zombie_history[-1]
            )

            # Physics and forecasting run every tick; expensive canvas redraw is throttled.
            if self.sim.tick_count % self.render_every == 0 or not self.sim.humans or not self.sim.zombies:
                render_start = time.perf_counter()
                self.draw_arena()
                self.draw_graph()
                self.update_header()
                self.render_times.append(time.perf_counter() - render_start)
                if len(self.render_times) > 100:
                    self.render_times.pop(0)

            if not self.sim.humans or not self.sim.zombies:
                self.running = False
                self.sim.close()

                outcome = "EXTINCTION (Zombies Won)" if not self.sim.humans else "CONTAINED (Humans Won)"
                metadata = {
                    "initial_regular": self.sim.initial_regular,
                    "initial_armed": self.sim.initial_armed,
                    "initial_zombies": self.sim.initial_zombies,
                    "total_ticks": self.sim.tick_count,
                    "outcome": outcome,
                }
                history_records = [
                    (tick, self.human_history[tick], self.zombie_history[tick])
                    for tick in range(len(self.human_history))
                ]
                if self.db_manager:
                    self.db_manager.save_run(metadata, history_records)

        self.root.after(20, self.update_loop)

    def draw_arena(self):
        self.arena_canvas.delete("all")
        s = self.scale

        for human in self.sim.humans:
            px = human.x * s
            py = human.y * s
            color = "#FFD700" if isinstance(human, ArmedHuman) else "#00D2FF"
            self.arena_canvas.create_rectangle(px, py, px + 2.0, py + 2.0, fill=color, outline="")

        for zombie in self.sim.zombies:
            px = zombie.x * s
            py = zombie.y * s
            self.arena_canvas.create_rectangle(px, py, px + 2.5, py + 2.5, fill="#FF3333", outline="")

    def draw_graph(self):
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
        dx = gw / max(pts - 1 + self.forecast_horizon, 40)

        human_coords, zombie_coords = [], []
        step_stride = max(1, (pts - 1) // 250)
        for i in sorted(set(range(0, pts, step_stride)) | {pts - 1}):
            px = margin + i * dx
            hy = (margin + gh) - (self.human_history[i] / max_pop) * gh
            zy = (margin + gh) - (self.zombie_history[i] / max_pop) * gh
            human_coords.extend([px, hy])
            zombie_coords.extend([px, zy])

        if len(human_coords) >= 4:
            self.graph_canvas.create_line(*human_coords, fill="#00D2FF", width=2)
            self.graph_canvas.create_line(*zombie_coords, fill="#FF3333", width=2)

        # Dashed future trajectory: latest observation through the next 10 ticks.
        forecast_coords = [margin + (pts - 1) * dx,
                           margin + gh - (self.zombie_history[-1] / max_pop) * gh]
        for horizon, prediction in enumerate(self.zombie_forecast, 1):
            forecast_coords.extend([
                margin + (pts - 1 + horizon) * dx,
                margin + gh - (prediction / max_pop) * gh,
            ])
        self.graph_canvas.create_line(*forecast_coords, fill="#FFAA33", width=2, dash=(4, 3))
        self.graph_canvas.create_text(margin + gw // 2, margin + gh + 15,
                                      text=f"Ticks 0 to {self.sim.tick_count + self.forecast_horizon}",
                                      fill="#BBBBBB", font=("Consolas", 9))

    def update_header(self):
        unarmed = sum(1 for h in self.sim.humans if not isinstance(h, ArmedHuman))
        armed = sum(1 for h in self.sim.humans if isinstance(h, ArmedHuman))
        zombies = len(self.sim.zombies)

        predicted = round(self.zombie_forecast[-1])
        metrics = self.forecast_tracker.metrics()
        score = (f" | 10-tick MAE: {metrics['mae']:.1f}"
                 if metrics['n'] else " | Forecast warming up")
        status = (f"Step: {self.sim.tick_count:03d} | Civilians: {unarmed} | "
                  f"Armed: {armed} | Zombies: {zombies} | "
                  f"Zombies +10: {predicted}{score}")
        if not self.sim.humans:
            status += "  [EXTINCTION (ZOMBIES WON)]"
        elif not self.sim.zombies:
            status += "  [CONTAINED (HUMANS WON)]"

        if self.step_times:
            status += f" | Step: {1000 * sum(self.step_times) / len(self.step_times):.0f}ms"
        if self.render_times:
            status += f" | Draw: {1000 * sum(self.render_times) / len(self.render_times):.0f}ms"
        self.status_label.config(text=status)


if __name__ == "__main__":
    db = DatabaseManager(
        dbname="zombie_db",
        user="zombie_admin",
        password="zombiepass123",
        host="127.0.0.1",
        port=5432,
    )

    playground = SimulationPlayground(
        width=1200.0,
        height=1200.0,
        use_multiprocessing=True,
        workers=1,
    )

    # Start at a responsive 1,000 agents; headless experiments can test 5k/10k.
    playground.populate(num_regular=750, num_armed=50, num_zombies=200)

    app = tk.Tk()
    gui = SimulationGUI(app, playground, scale=0.52, db_manager=db)
    app.mainloop()