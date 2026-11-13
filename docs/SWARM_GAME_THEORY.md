# Swarm Game Theory: Contextual Thompson Sampling Explained

This is an explanatory companion to the existing `models.py` and
`Zombie_Game.py` implementation. **It does not change Owen's algorithm.**
The model file itself remains the reference for exact implementation behavior.

Read with [Architecture](ARCHITECTURE.md), [Integration](INTEGRATION.md), and
[Forecast mathematics](FORECAST_MATH.md).

## 1. What problem does the learner solve?

Each zombie that sees a nearby human must choose a pursuit action.
Three discrete actions are available, represented by integer IDs:

| Action ID | Name | Behavior in `Zombie.update()` |
| --- | --- | --- |
| `0` | `rush` | Head toward the selected human |
| `1` | `flank` | Rotate the target direction by +0.65 radians before moving |
| `2` | `swarm_rally` | Blend target position with the center of nearby zombies when more than one ally is in range |

The learner uses the local situation, or **context**, to choose an action.
The game supplies rewards when actions lead to damage dealt or consequences
from armed civilians. This is a contextual multi-armed bandit approach,
not multi-step perfect-information minimax or an adversarial search tree.

## 2. Where the contextual feature vector comes from

`Zombie.extract_context(arena, nearest_human, min_dist)` builds a vector
with exactly four floating-point features:

```python
context = [
    min_dist / hunt_radius,
    nearby_armed_humans / nearby_humans,
    min(1.0, nearby_zombies / 6.0),
    1.0,
]
```

The code uses `hunt_radius=140.0`, a 50-unit radius when measuring local
armed-human proportion and zombie allies, and a constant final bias feature.

- The **distance feature** helps distinguish a close target from a distant one.
- The **armed-human fraction** approximates local threat.
- The **ally-density feature** indicates the local presence of other zombies.
- The **bias** allows the linear model to learn a base preference for a
  strategy independent of the other features.

There is no learned time-series forecast in the context vector.
`SlidingWindowAdaptiveKalman` output is **not** an input to the policy.

## 3. What Thompson Sampling means in this code

For each action `a`, the model tracks:

- `A_inv[a]`: a 4x4 inverse design/precision-like matrix, initialized to
  the identity.
- `b[a]`: a 4x1 reward-weighted sum of context features, initialized to zero.
- `sampled_thetas[a]`: one sampled 4x1 vector of action weights.

A posterior mean for an action is computed as:

```text
mu_a = A_inv[a] @ b[a]
```

The sampled action weights follow approximately

```text
theta_a ~ Normal(mu_a, v_squared * symmetric(A_inv[a]) + epsilon * I)
```

where the implementation defaults to `v_squared = 0.25` and adds
`epsilon = 1e-6` to the diagonal for numerical protection.

Given the context column vector `x`, action `a` receives score

```text
score_a = theta_a.T @ x
```

The learner returns the action index with the **highest sampled score**.
Because weights are sampled from a distribution rather than always using
the posterior mean, the method trades off **exploration** and
**exploitation**.

### Sampling frequency

`LinearThompsonSwarmLearner.step()` increments its internal clock
once per simulation tick; with `resample_interval=5`, fresh
`sampled_thetas` are drawn at ticks 5, 10, 15, and so on. Between these
ticks, the policy uses the same sampled weight vectors even though the
underlying posterior may update after rewards.

The learner instance is **shared by the arena's zombies**. Although the
class calls itself a decentralized swarm policy, the actual current
`SimulationPlayground` creates one `self.swarm_learner`, not an
independent posterior per individual zombie. That distinction matters
when describing the code to a professor.

## 4. How it learns: rank-one posterior updates

For the selected action and context `x`, the game calls
`update(action, x, reward)`. The implementation performs

```text
b_a(new) = b_a(old) + reward * x

A_inv(new) = A_inv(old) -
    (A_inv(old) @ x @ x.T @ A_inv(old)) / (1 + x.T @ A_inv(old) @ x)
```

This second expression is the **Sherman–Morrison** rank-one update.
It updates the inverse matrix without recomputing a full inverse each time.

The update is per-action: choosing `flank` changes `flank`'s stored
posterior statistics, not those of `rush` or `swarm_rally`.

This describes the exact implemented online linear estimator. It should
not be presented as a guarantee that the reward assumptions perfectly
model the actual game or that the resulting strategy is optimal.

## 5. Where rewards come from

Rewards are passed by game interactions in `Zombie_Game.py`:

| Event | Effect on model |
| --- | --- |
| Zombie bite | `+1.2` reward for `zombie.last_action` |
| Bite kills human | Additional `+3.0` reward, total `+4.2` for that bite |
| Armed human shoots zombie | `-1.0` reward/penalty associated with the zombie's last action and context |

Each zombie stores its own `last_action` and `last_context`, but
the **posterior statistics are stored centrally** in `arena.swarm_learner`.
The updates come from interactions, not from forecast prediction error.

The negative shooting feedback may use a zombie's last stored action/context,
which can be from a prior tick; the program does not maintain a separate
causal credit-assignment log. Thus, the reward represents a heuristic
feedback signal rather than a rigorous experimental measure of every
action's causal value.

## 6. Step-by-step example (illustrative, not a recorded run)

1. A zombie is close to a human; the context is
   `[0.20, 0.00, 0.50, 1.00]`.
2. The learner computes scores for all three sampled action weight vectors.
3. Suppose `flank` has the highest score; the returned integer is `1`.
4. The zombie records `last_action=1` and moves toward a rotated target.
5. If it bites the human but does not kill them, the game calls
   `update(1, context, 1.2)`.
6. `b[1]` and `A_inv[1]` change. On a later scheduled resampling tick,
   the updated posterior can affect future action choices.

A different random seed or posterior state can result in a different action.

## 7. How it interacts with the forecasting specialization

The **normal GUI** always begins with
`SimulationPlayground.strategy_mode = "thompson"`. Forecasting
observes the number of surviving zombies after each tick and predicts
future population counts.

The optional research run uses

```powershell
py experiment_runner.py --agents 1000 --ticks 300 --seeds 1 2 3 4 5 --shift-tick 150 --output results/shift_v1
```

At tick 150 the runner changes `strategy_mode` to `"rush_only"`.
For zombies that have a target, the game then chooses action `0`
without calling `select_action(context)`. The existing
`swarm_learner.step()` and reward `update()` calls still execute.
Therefore, the intervention **bypasses action selection**, but does
**not** disable all learning-state updates.

A policy change can alter population trajectories and the difficulty
of forecasting them. This does **not** demonstrate that the forecasting
algorithm drives zombie decisions.

## 8. How to verify the learner was preserved

Run all regression tests:

```powershell
py -m unittest discover -s tests -v
```

`tests/test_swarm_integration.py` checks, among other things:

- The default mode selects through `LinearThompsonSwarmLearner`.
- Each of the three actions remains available and dispatches correctly.
- The `rush_only` research mode is opt-in and reversible.
- Bite and human-to-zombie conversion still trigger policy updates.
- The resampling clock advances on simulation ticks.
- Changing the forecasting window does not advance the game or mutate
  the learner's posterior.

**Limitations:** This suite does not prove optimal policy performance,
exhaustively test stochastic trajectories, or verify a separate learning
model per zombie. `models.py` is intentionally not modified as part of
forecasting integration.

## 9. Questions an instructor may ask

**What is the difference between a policy and a forecast?**
A policy chooses what an agent should do; a forecast predicts an
observable quantity. The Thompson learner chooses pursuit behavior,
while the adaptive filter estimates future zombie population.

**Is this supervised learning?**
Not in the ordinary offline labeled-dataset sense. The bandit
learner updates action-level statistics online from contextual rewards.

**Why use Sherman–Morrison?**
It efficiently maintains the inverse matrix after each rank-one
context update without explicitly inverting the full matrix again.

**Does the strategy switch remove Owen's research?**
No. In ordinary gameplay the original policy is selected by default,
its source file is unchanged, and integration tests protect its
action-selection and reward-update pathways.

**Does an unchanged `models.py` guarantee the entire game theory system
is unchanged?**
No. Game-theory outcomes also depend on simulation mechanics, contexts,
rewards, and the order of updates in `Zombie_Game.py`, which is why
integration tests and manual validation matter.
