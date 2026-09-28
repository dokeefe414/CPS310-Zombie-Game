from abc import ABC, abstractmethod
import math
import random
import tkinter as tk


class Agent(ABC):
    """Abstract base class representing an entity in 2D bounded Euclidean space."""

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
    """Regular civilian human: flees from zombies or wanders calmly."""

    def __init__(self, x: float, y: float, speed: float = 2.2):
        super().__init__(x=x, y=y, speed=speed)
        self.perception_radius = 28.0

    def get_nearest_zombie(self, arena: "SimulationPlayground"):
        nearest = None
        min_dist = float("inf")
        for zombie in arena.zombies:
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

        if nearest_zombie and min_dist <= self.perception_radius:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height)
        else:
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.5)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.5)
            self.clamp_position(arena.width, arena.height)


class ArmedHuman(Human):
    """Specialized human capable of firing ranged shots at encroaching zombies."""

    def __init__(self, x: float, y: float, speed: float = 2.0):
        super().__init__(x=x, y=y, speed=speed)
        self.attack_range = 35.0
        self.shot_damage = 40.0
        self.cooldown_max = 8
        self.cooldown_timer = random.randint(0, self.cooldown_max)

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        nearest_zombie, min_dist = self.get_nearest_zombie(arena)

        # Ranged attack engagement
        if nearest_zombie and min_dist <= self.attack_range and self.cooldown_timer == 0:
            self.fire_weapon(nearest_zombie, arena)
            self.cooldown_timer = self.cooldown_max

        # Kite/Flee logic
        if nearest_zombie and min_dist <= self.perception_radius:
            self.move_away_from(nearest_zombie.x, nearest_zombie.y, arena.width, arena.height)
        else:
            self.x += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.y += random.uniform(-1.0, 1.0) * (self.speed * 0.4)
            self.clamp_position(arena.width, arena.height)

    def fire_weapon(self, target: "Zombie", arena: "SimulationPlayground"):
        target.take_damage(self.shot_damage)
        arena.bullet_tracers.append(((self.x, self.y), (target.x, target.y)))


class Zombie(Agent):
    """Zombie chases humans, bites in close combat, and experiences starvation decay."""

    def __init__(self, x: float, y: float, speed: float = 1.65, bite_damage: float = 35.0):
        super().__init__(x=x, y=y, speed=speed)
        self.bite_range = 3.0
        self.bite_damage = bite_damage

    def update(self, arena: "SimulationPlayground"):
        if not self.alive:
            return

        # Gradual hunger decay
        self.take_damage(0.2)
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
            self.move_towards(nearest_human.x, nearest_human.y, arena.width, arena.height)
            if self.distance_to(nearest_human) <= self.bite_range:
                self.bite(nearest_human, arena)
        else:
            self.x += random.uniform(-1.0, 1.0) * self.speed
            self.y += random.uniform(-1.0, 1.0) * self.speed
            self.clamp_position(arena.width, arena.height)

    def bite(self, human: Human, arena: "SimulationPlayground"):
        human.take_damage(self.bite_damage)
        self.health = min(self.max_health, self.health + 15.0)
        if not human.alive:
            arena.convert_human_to_zombie(human)


class SimulationPlayground:
    """Bounded Euclidean environment and coordination loop."""

    def __init__(self, width: float = 120.0, height: float = 120.0):
        self.width = width
        self.height = height
        self.humans: list[Human] = []
        self.zombies: list[Zombie] = []
        self.new_zombies: list[Zombie] = []
        self.bullet_tracers: list[tuple] = []
        self.tick_count = 0

    def populate(self, num_regular: int, num_armed: int, num_zombies: int):
        for _ in range(num_regular):
            self.humans.append(Human(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_armed):
            self.humans.append(ArmedHuman(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))
        for _ in range(num_zombies):
            self.zombies.append(Zombie(x=random.uniform(0, self.width), y=random.uniform(0, self.height)))

    def convert_human_to_zombie(self, human: Human):
        self.new_zombies.append(Zombie(x=human.x, y=human.y))

    def step(self):
        self.tick_count += 1
        self.new_zombies.clear()
        self.bullet_tracers.clear()

        for human in list(self.humans):
            human.update(self)

        for zombie in list(self.zombies):
            zombie.update(self)

        self.zombies.extend(self.new_zombies)
        self.humans = [h for h in self.humans if h.alive]
        self.zombies = [z for z in self.zombies if z.alive]


class SimulationGUI:
    """Tkinter interface containing visual arena and dynamic population graph."""

    def __init__(self, root, sim: SimulationPlayground, scale: float = 5.0):
        self.root = root
        self.sim = sim
        self.scale = scale
        self.arena_px = int(sim.width * scale)
        self.graph_w = 400
        self.graph_h = self.arena_px

        self.root.title("Zombie Game Simulation - Stage 1")
        self.root.configure(bg="#1E1E1E")

        # Top status bar
        self.status_label = tk.Label(
            root, text="", font=("Consolas", 12, "bold"), fg="#FFFFFF", bg="#1E1E1E", pady=6
        )
        self.status_label.pack()

        # Canvas container frame
        panes_frame = tk.Frame(root, bg="#1E1E1E")
        panes_frame.pack(padx=10, pady=5)

        # Left: Arena Canvas
        self.arena_canvas = tk.Canvas(
            panes_frame, width=self.arena_px, height=self.arena_px, bg="#121212", highlightthickness=1
        )
        self.arena_canvas.grid(row=0, column=0, padx=6)

        # Right: Graph Canvas
        self.graph_canvas = tk.Canvas(
            panes_frame, width=self.graph_w, height=self.graph_h, bg="#181818", highlightthickness=1
        )
        self.graph_canvas.grid(row=0, column=1, padx=6)

        # Population history tracking
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

        self.root.after(40, self.update_loop)

    def draw_health_bar(self, cx: float, cy: float, health: float, max_health: float, radius: float):
        """Draws a mini proportional health bar directly above the agent."""
        bar_w = 12.0
        bar_h = 2.5
        bx1 = cx - bar_w / 2.0
        by1 = cy - radius - 5.0
        bx2 = cx + bar_w / 2.0
        by2 = by1 + bar_h

        # Background track (dark red/gray)
        self.arena_canvas.create_rectangle(bx1, by1, bx2, by2, fill="#331111", outline="")

        # Dynamic health fill
        pct = max(0.0, min(1.0, health / max_health))
        fill_x2 = bx1 + bar_w * pct

        # Color changes based on current health status
        if pct > 0.5:
            bar_color = "#2ECC71"  # Healthy Green
        elif pct > 0.2:
            bar_color = "#F39C12"  # Damaged Orange
        else:
            bar_color = "#E74C3C"  # Critical Red

        if pct > 0:
            self.arena_canvas.create_rectangle(bx1, by1, fill_x2, by2, fill=bar_color, outline="")

    def draw_arena(self):
        self.arena_canvas.delete("all")

        # Draw bullet lines
        for (sx, sy), (tx, ty) in self.sim.bullet_tracers:
            self.arena_canvas.create_line(
                sx * self.scale, sy * self.scale,
                tx * self.scale, ty * self.scale,
                fill="#FFDD00", width=2, dash=(4, 2)
            )

        # Draw Humans + Health Bars
        r = 4.0
        for human in self.sim.humans:
            cx, cy = human.x * self.scale, human.y * self.scale
            color = "#FFD700" if isinstance(human, ArmedHuman) else "#00D2FF"
            self.arena_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")
            self.draw_health_bar(cx, cy, human.health, human.max_health, r)

        # Draw Zombies + Health Bars
        zr = 4.5
        for zombie in self.sim.zombies:
            cx, cy = zombie.x * self.scale, zombie.y * self.scale
            self.arena_canvas.create_oval(cx - zr, cy - zr, cx + zr, cy + zr, fill="#44FF44", outline="")
            self.draw_health_bar(cx, cy, zombie.health, zombie.max_health, zr)

    def draw_graph(self):
        self.graph_canvas.delete("all")
        margin = 35
        gw = self.graph_w - margin * 2
        gh = self.graph_h - margin * 2

        # Axes & Labels
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

        human_coords, zombie_coords = [], []
        for i in range(pts):
            px = margin + i * dx
            hy = (margin + gh) - (self.human_history[i] / max_pop) * gh
            zy = (margin + gh) - (self.zombie_history[i] / max_pop) * gh
            human_coords.extend([px, hy])
            zombie_coords.extend([px, zy])

        self.graph_canvas.create_line(*human_coords, fill="#00D2FF", width=2)
        self.graph_canvas.create_line(*zombie_coords, fill="#44FF44", width=2)

        # Legend
        self.graph_canvas.create_rectangle(margin + 10, margin + 10, margin + 20, margin + 20, fill="#00D2FF", outline="")
        self.graph_canvas.create_text(margin + 26, margin + 15, text="Civilians", anchor="w", fill="#FFFFFF", font=("Consolas", 9))
        self.graph_canvas.create_rectangle(margin + 10, margin + 28, margin + 20, margin + 38, fill="#FFD700", outline="")
        self.graph_canvas.create_text(margin + 26, margin + 33, text="Armed", anchor="w", fill="#FFFFFF", font=("Consolas", 9))
        self.graph_canvas.create_rectangle(margin + 10, margin + 46, margin + 20, margin + 56, fill="#44FF44", outline="")
        self.graph_canvas.create_text(margin + 26, margin + 51, text="Zombies", anchor="w", fill="#FFFFFF", font=("Consolas", 9))

    def update_header(self):
        unarmed = sum(1 for h in self.sim.humans if not isinstance(h, ArmedHuman))
        armed = sum(1 for h in self.sim.humans if isinstance(h, ArmedHuman))
        zombies = len(self.sim.zombies)

        status = f"Step: {self.sim.tick_count:03d} | Civilians: {unarmed} | Armed: {armed} | Zombies: {zombies}"
        if not self.sim.humans:
            status += "  [EXTINCTION: Zombies Win!]"
        elif not self.sim.zombies:
            status += "  [CONTAINED: Humans Win!]"

        self.status_label.config(text=status)


if __name__ == "__main__":
    playground = SimulationPlayground(width=120.0, height=120.0)
    playground.populate(num_regular=25, num_armed=8, num_zombies=6)

    app = tk.Tk()
    gui = SimulationGUI(app, playground, scale=4.5)
    app.mainloop()