"""
Multi-Agent Swarm Intelligence Model.
Implements Contextual Linear Thompson Sampling with Sherman-Morrison rank-1 updates
for decentralized action coordination in multi-agent environments.
"""

import numpy as np


class LinearThompsonSwarmLearner:
    """
    Decentralized Swarm Learning Policy using Contextual Linear Thompson Sampling.
    Maintains Bayesian posterior distributions over linear contextual reward weights
    with numerical stability protections and Sherman-Morrison rank-1 updates.
    """

    def __init__(self, n_features: int = 4, v_squared: float = 0.25, resample_interval: int = 5):
        self.n_features = n_features
        self.v_squared = v_squared
        self.actions = ["rush", "flank", "swarm_rally"]
        self.n_actions = len(self.actions)

        # Precision matrix inverses (A_inv = Covariance / v^2) per action
        self.A_inv = {a: np.identity(n_features, dtype=np.float64) for a in range(self.n_actions)}
        # Feature-weighted accumulated reward vectors
        self.b = {a: np.zeros((n_features, 1), dtype=np.float64) for a in range(self.n_actions)}

        self.sampled_thetas = {}
        self.resample_interval = resample_interval
        self._step_counter = 0
        self.sample_thetas()

    def sample_thetas(self):
        """Draws parameter vectors θ_a ~ N(μ_a, v^2 * A_a^-1) from current posteriors."""
        for a in range(self.n_actions):
            A_inv = self.A_inv[a]
            # Symmetrize covariance matrix to guarantee positive semi-definiteness
            sym_cov = 0.5 * (A_inv + A_inv.T) * self.v_squared
            sym_cov += np.eye(self.n_features) * 1e-6

            mu_a = (A_inv @ self.b[a]).flatten()
            self.sampled_thetas[a] = np.random.multivariate_normal(mu_a, sym_cov).reshape(-1, 1)

    def select_action(self, context: np.ndarray) -> int:
        """Evaluates context vector against sampled posterior parameters."""
        x = context.reshape(-1, 1)
        best_action = 0
        max_score = -float("inf")

        for a in range(self.n_actions):
            score = (self.sampled_thetas[a].T @ x).item()
            if score > max_score:
                max_score = score
                best_action = a

        return best_action

    def step(self):
        """Advances internal decision clock to refresh Thompson posterior samples."""
        self._step_counter += 1
        if self._step_counter % self.resample_interval == 0:
            self.sample_thetas()

    def update(self, action: int, context: np.ndarray, reward: float):
        """Applies exact recursive Bayesian update via Sherman-Morrison rank-1 formulation."""
        x = context.reshape(-1, 1)
        self.b[action] += reward * x

        A_inv = self.A_inv[action]
        numerator = A_inv @ x @ x.T @ A_inv
        denominator = 1.0 + (x.T @ A_inv @ x).item()

        self.A_inv[action] = A_inv - (numerator / denominator)