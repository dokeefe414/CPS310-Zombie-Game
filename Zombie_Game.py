from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random
import tkinter as tk


class SpatialGrid:
    """Partitions 2D Euclidean space into fixed buckets for O(1) neighbor lookups."""

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
    """Abstract base class representing an entity in 2D bounded space."""

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
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        self.clamp_position(arena_w, arena_h)

    def move_away_from(self, threat_x: float, threat_y: float, arena_w: float, arena_h: float):
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
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    @abstractmethod
    def update(self, arena: "SimulationPlayground"):
        pass


class Human(Agent):
    """Regular civilian human: flees or wanders."""

    def __init__(self, x: float, y: float, speed: float = 2.4):
        super().__init__(x=x, y=y, speed=speed)
        self.perception_radius = 25.0

    def get_nearest_zombie(self, arena: "SimulationPlayground"):
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

        if nearest_zombie:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height)
        else:
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.clamp_position(arena.width, arena.height)


class ArmedHuman(Human):
    """Specialized human capable of firing ranged shots at zombies."""

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
    """Zombie chases humans, bites, and decays via starvation."""

    def __init__(self, x: float, y: float, speed: float = 2.1, bite_damage: float = 40.0):
        super().__init__(x=x, y=y, speed=speed)
        self.bite_range = 3.2
        self.bite_damage = bite_damage
        self.hunt_radius = 120

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        self.take_damage(0.075)
        if not self.alive:
            return

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
        human.take_damage(self.bite_damage)
        self.health = min(self.max_health, self.health + 18.0)
        if not human.alive:
            arena.convert_human_to_zombie(human)


class SimulationPlayground:
    """Bounded Euclidean environment and coordination loop with spatial bucketing."""

    def __init__(self, width: float = 1200.0, height: float = 1200.0):
        self.width = width
        self.height = height
        self.humans: list[Human] = []
        self.zombies: list[Zombie] = []
        self.new_zombies: list[Zombie] = []
        self.tick_count = 0

        self.human_grid = SpatialGrid(cell_size=80.0)
        self.zombie_grid = SpatialGrid(cell_size=80.0)

    def populate(self, num_regular: int, num_armed: int, num_zombies: int):
        for _ in range(num_regular):
            self.humans.append(Human(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_armed):
            self.humans.append(ArmedHuman(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_zombies):
            self.zombies.append(Zombie(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))

    def convert_human_to_zombie(self, human: Human):
        self.new_zombies.append(Zombie(x=human.x, y=human.y))

    def rebuild_spatial_grids(self):
        self.human_grid.clear()
        self.zombie_grid.clear()
        for h in self.humans:
            self.human_grid.insert(h)
        for z in self.zombies:
            self.zombie_grid.insert(z)

    def step(self):
        self.tick_count += 1
        self.new_zombies.clear()

        self.rebuild_spatial_grids()

        for human in self.humans:
            human.update(self)

        for zombie in self.zombies:
            zombie.update(self)

        self.zombies.extend(self.new_zombies)
        self.humans = [h for h in self.humans if h.alive]
        self.zombies = [z for z in self.zombies if z.alive]


class SimulationGUI:
    """Tkinter interface optimized for high-density particle crowd rendering."""

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
        self.update_loop()

    def update_loop(self):
        if self.running:
            self.sim.step()
            self.human_history.append(len(self.sim.humans))
            self.zombie_history.append(len(self.sim.zombies))

            self.draw_arena()
            self.draw_graph()
            self.update_header()

            if not self.sim.humans or not self.sim.zombies:
                self.running = False

        self.root.after(20, self.update_loop)

    def draw_arena(self):
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
        unarmed = sum(1 for h in self.sim.humans if not isinstance(h, ArmedHuman))
        armed = sum(1 for h in self.sim.humans if isinstance(h, ArmedHuman))
        zombies = len(self.sim.zombies)

        status = f"Step: {self.sim.tick_count:03d} | Civilians: {unarmed} | Armed: {armed} | Zombies: {zombies} | Total: {unarmed+armed+zombies}"
        if not self.sim.humans:
            status += "  [EXTINCTION: Zombies Win!]"
        elif not self.sim.zombies:
            status += "  [CONTAINED: Humans Win!]"

        self.status_label.config(text=status)


if __name__ == "__main__":
    # 10,000 Agents Setup
    playground = SimulationPlayground(width=1200.0, height=1200.0)

    # 7,500 Civilians, 500 Armed, 2,000 Zombies = 10,000 Total
    playground.populate(num_regular=7500, num_armed=500, num_zombies=2000)

    app = tk.Tk()
    gui = SimulationGUI(app, playground, scale=0.52)
    app.mainloop()