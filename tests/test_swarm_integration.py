"""Integration regression tests: protect the existing swarm policy.

Run with:
    py -m unittest discover -s tests -v

These tests deliberately do not open Tkinter, require PostgreSQL, or edit
Owen's models.py. They check the simulation/forecasting integration boundary.
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from forecasting import SlidingWindowAdaptiveKalman
from experiment_runner import run_simulation
from Zombie_Game import Human, SimulationGUI, SimulationPlayground, Zombie


class SwarmIntegrationTests(unittest.TestCase):
    def make_arena(self):
        arena = SimulationPlayground(use_multiprocessing=False, workers=1)
        human = Human(x=600.0, y=600.0)
        zombie = Zombie(x=610.0, y=600.0)
        arena.humans.append(human)
        arena.zombies.append(zombie)
        return arena, human, zombie

    def test_thompson_is_default_and_each_action_still_reachable(self):
        """Normal simulation must call Owen's learner for all three actions."""
        for action in (0, 1, 2):  # rush, flank, swarm_rally
            with self.subTest(action=action):
                arena, _, zombie = self.make_arena()
                try:
                    self.assertEqual(arena.strategy_mode, "thompson")
                    self.assertEqual(arena.swarm_learner.actions,
                                     ["rush", "flank", "swarm_rally"])
                    with patch.object(arena.swarm_learner, "select_action",
                                      return_value=action) as select:
                        arena.step()
                    select.assert_called_once()
                    self.assertEqual(zombie.last_action, action)
                    context = select.call_args.args[0]
                    self.assertEqual(context.shape, (4,))
                    self.assertTrue(np.all(np.isfinite(context)))
                    self.assertEqual(arena.tick_count, 1)
                finally:
                    arena.close()

    def test_research_intervention_is_opt_in_and_reversible(self):
        """Rush-only experiments bypass selection; returning to default restores it."""
        arena, _, zombie = self.make_arena()
        try:
            arena.strategy_mode = "rush_only"
            with patch.object(arena.swarm_learner, "select_action") as select:
                arena.step()
            select.assert_not_called()
            self.assertEqual(zombie.last_action, 0)

            arena.strategy_mode = "thompson"
            with patch.object(arena.swarm_learner, "select_action",
                              return_value=2) as select:
                arena.step()
            select.assert_called_once()
            self.assertEqual(zombie.last_action, 2)
        finally:
            arena.close()

    def test_swarm_posterior_updates_on_bite_and_conversion(self):
        """Original game-theory reward/posterior path remains operational."""
        arena, human, zombie = self.make_arena()
        try:
            human.health = 1.0
            zombie.last_action = 1
            zombie.last_context = np.array([0.2, 0.0, 0.5, 1.0])
            before = arena.swarm_learner.A_inv[1].copy()
            zombie.bite(human, arena)
            self.assertFalse(human.alive)
            self.assertEqual(len(arena.new_zombies), 1)
            self.assertFalse(np.allclose(before, arena.swarm_learner.A_inv[1]))
            self.assertTrue(np.all(np.isfinite(arena.swarm_learner.A_inv[1])))
        finally:
            arena.close()

    def test_thompson_clock_refreshes_on_normal_ticks(self):
        """Simulation ticks still advance Owen's posterior-sampling clock."""
        arena = SimulationPlayground(use_multiprocessing=False, workers=1)
        try:
            with patch.object(arena.swarm_learner, "sample_thetas",
                              wraps=arena.swarm_learner.sample_thetas) as resample:
                for _ in range(5):
                    arena.step()
            self.assertEqual(arena.swarm_learner._step_counter, 5)
            resample.assert_called_once()
        finally:
            arena.close()

    def test_window_replay_does_not_modify_simulation_or_learner(self):
        """Switching forecast window replays observations, not simulation ticks."""
        arena, _, _ = self.make_arena()
        try:
            gui = object.__new__(SimulationGUI)  # avoid needing a display
            gui.window_var = Mock()
            gui.window_var.get.return_value = 20
            gui.total_initial = 1000
            gui.forecast_horizon = 10
            gui.zombie_history = [200, 200, 201, 202, 203, 202, 201]
            gui.draw_graph = Mock()
            gui.update_header = Mock()
            before_count = arena.tick_count
            before_a = {a: x.copy() for a, x in arena.swarm_learner.A_inv.items()}
            before_b = {a: x.copy() for a, x in arena.swarm_learner.b.items()}

            gui.change_window()

            self.assertIsInstance(gui.forecaster, SlidingWindowAdaptiveKalman)
            self.assertEqual(gui.forecaster.window, 20)
            self.assertEqual(gui.forecaster.observations, len(gui.zombie_history))
            self.assertEqual(len(gui.zombie_forecast), 10)
            self.assertEqual(arena.tick_count, before_count)
            for action in before_a:
                np.testing.assert_array_equal(arena.swarm_learner.A_inv[action],
                                              before_a[action])
                np.testing.assert_array_equal(arena.swarm_learner.b[action],
                                              before_b[action])
            gui.draw_graph.assert_called_once()
            gui.update_header.assert_called_once()
        finally:
            arena.close()

    def test_same_seed_has_identical_pre_intervention_population(self):
        """Intervention must not affect behavior before its requested tick."""
        control, _ = run_simulation(agents=60, ticks=8, seed=19)
        shifted, _ = run_simulation(agents=60, ticks=8, seed=19,
                                    shift_tick=5)
        self.assertGreaterEqual(len(control), 5)
        self.assertEqual(control[:5], shifted[:5])
        self.assertEqual(shifted[5]["strategy"], "rush_only")


if __name__ == "__main__":
    unittest.main()
