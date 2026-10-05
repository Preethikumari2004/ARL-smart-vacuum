"""
trainer.py
----------
Training loop that runs Q-learning for a configurable number of episodes
and records per-episode metrics.

Usage:
    session = TrainingSession(env_config, hyperparams)
    results = session.train()
    # results is a dict with lists of per-episode metrics
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.environment.vacuum_env import VacuumEnv
from backend.agent.q_learning_agent import QLearningAgent
from backend.utils.helpers import compute_rolling_average
from backend.config import DEFAULT_HYPERPARAMS


class TrainingSession:
    """
    Manages a full Q-learning training run.

    Parameters
    ----------
    env_config   : dict  — grid configuration (size, vacuum_start, etc.)
    hyperparams  : dict  — override any value from DEFAULT_HYPERPARAMS
    """

    def __init__(self, env_config: dict, hyperparams: dict = None):
        self.env_config  = env_config
        self.hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}

        # Create the agent
        self.agent = QLearningAgent(
            learning_rate   = self.hp["learning_rate"],
            discount_factor = self.hp["discount_factor"],
            epsilon         = self.hp["epsilon"],
            epsilon_min     = self.hp["epsilon_min"],
            epsilon_decay   = self.hp["epsilon_decay"],
        )

        # Per-episode metric lists (populated during training)
        self.episode_rewards     = []
        self.episode_steps       = []
        self.episode_collisions  = []
        self.episode_completion  = []   # % of dirt cleaned
        self.episode_epsilons    = []
        self.episode_td_errors   = []   # mean TD-error per episode

        self.trained = False

    # ------------------------------------------------------------------
    def train(self, progress_callback=None) -> dict:
        """
        Run the training loop.

        Parameters
        ----------
        progress_callback : optional callable(episode, total, metrics_dict)
            Called every 100 episodes so callers can report live progress.

        Returns
        -------
        dict with full training results and the trained agent.
        """
        n_episodes = self.hp["episodes"]
        max_steps  = self.hp["max_steps"]

        best_reward  = float("-inf")
        best_episode = 0

        for episode in range(1, n_episodes + 1):
            # Create a fresh environment for each episode
            env   = VacuumEnv(self.env_config, max_steps=max_steps)
            state = env.reset()

            ep_reward    = 0.0
            ep_steps     = 0
            ep_collisions = 0
            ep_td_errors  = []

            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Q-Learning update
                td_error = self.agent.update(state, action, reward, next_state)
                ep_td_errors.append(abs(td_error))

                state         = next_state
                ep_reward    += reward
                ep_steps     += 1
                ep_collisions += 1 if info["collision"] else 0

            # Decay epsilon after each episode
            self.agent.decay_epsilon()

            # Record metrics
            ep_reward = round(ep_reward, 4)
            self.episode_rewards.append(ep_reward)
            self.episode_steps.append(ep_steps)
            self.episode_collisions.append(ep_collisions)
            self.episode_completion.append(env.get_completion_pct())
            self.episode_epsilons.append(round(self.agent.epsilon, 6))
            mean_td = round(sum(ep_td_errors) / max(len(ep_td_errors), 1), 4)
            self.episode_td_errors.append(mean_td)

            if ep_reward > best_reward:
                best_reward  = ep_reward
                best_episode = episode

            # Optional progress callback (every 100 episodes)
            if progress_callback and episode % 100 == 0:
                progress_callback(episode, n_episodes, {
                    "episode":          episode,
                    "reward":           ep_reward,
                    "epsilon":          self.agent.epsilon,
                    "avg_reward":       self._avg_last(self.episode_rewards, 100),
                    "best_reward":      best_reward,
                    "q_table_size":     len(self.agent.q_table),
                })

        # Compute rolling averages for charts
        rolling_rewards = compute_rolling_average(self.episode_rewards, window=100)
        rolling_steps   = compute_rolling_average(self.episode_steps,   window=100)

        self.trained = True

        return {
            "episodes":            n_episodes,
            "hyperparams":         self.hp,
            "best_reward":         best_reward,
            "best_episode":        best_episode,
            "final_epsilon":       round(self.agent.epsilon, 6),
            "q_table_size":        len(self.agent.q_table),
            "total_q_updates":     self.agent.total_updates,
            "episode_rewards":     self.episode_rewards,
            "episode_steps":       self.episode_steps,
            "episode_collisions":  self.episode_collisions,
            "episode_completion":  self.episode_completion,
            "episode_epsilons":    self.episode_epsilons,
            "episode_td_errors":   self.episode_td_errors,
            "rolling_avg_rewards": rolling_rewards,
            "rolling_avg_steps":   rolling_steps,
        }

    # ------------------------------------------------------------------
    def get_agent(self) -> QLearningAgent:
        """Return the trained agent."""
        return self.agent

    # ------------------------------------------------------------------
    @staticmethod
    def _avg_last(lst: list, n: int) -> float:
        if not lst:
            return 0.0
        window = lst[-n:]
        return round(sum(window) / len(window), 4)
