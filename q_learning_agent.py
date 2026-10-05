"""
q_learning_agent.py
-------------------
Implements the Q-Learning agent.

Algorithm:
    Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') - Q(s, a) ]

Key concepts:
    α  (learning_rate)   — how much new info overrides old info
    γ  (discount_factor) — how much future rewards are valued
    ε  (epsilon)         — exploration probability (decays over time)

The Q-table is stored as a nested Python dict:
    { state_tuple : { action_int : float } }

States are tuples: (row, col, frozenset_of_dirty_cells)
"""

import random
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.config import ACTION_COUNT, ACTIONS, DEFAULT_HYPERPARAMS


class QLearningAgent:
    """
    Tabular Q-Learning agent for the vacuum cleaning environment.

    Parameters
    ----------
    learning_rate   : α — step size for Q-value updates
    discount_factor : γ — importance of future rewards
    epsilon         : initial exploration probability
    epsilon_min     : floor for epsilon after decay
    epsilon_decay   : multiplicative decay factor applied each episode
    action_count    : number of available actions (default 5)
    """

    def __init__(
        self,
        learning_rate:   float = DEFAULT_HYPERPARAMS["learning_rate"],
        discount_factor: float = DEFAULT_HYPERPARAMS["discount_factor"],
        epsilon:         float = DEFAULT_HYPERPARAMS["epsilon"],
        epsilon_min:     float = DEFAULT_HYPERPARAMS["epsilon_min"],
        epsilon_decay:   float = DEFAULT_HYPERPARAMS["epsilon_decay"],
        action_count:    int   = ACTION_COUNT,
    ):
        self.alpha         = learning_rate
        self.gamma         = discount_factor
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_count  = action_count

        # Q-table: dict of { state -> list of Q-values per action }
        # Using a list (indexed by action) for speed; converted to dict for display
        self.q_table: dict = {}

        # Statistics
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Q-Table Access
    # ------------------------------------------------------------------
    def _ensure_state(self, state):
        """Initialise Q-values for a new state to 0.0."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_count

    def get_q_value(self, state, action: int) -> float:
        """Return Q(state, action), defaulting to 0.0 for unseen states."""
        self._ensure_state(state)
        return self.q_table[state][action]

    def get_all_q_values(self, state) -> list:
        """Return all Q-values for a state as a list indexed by action."""
        self._ensure_state(state)
        return list(self.q_table[state])

    def get_best_action(self, state) -> int:
        """Return the action with the highest Q-value for a given state."""
        self._ensure_state(state)
        values = self.q_table[state]
        # Break ties randomly for cleaner exploration
        max_val = max(values)
        best_actions = [a for a, v in enumerate(values) if v == max_val]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Action Selection  (ε-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state) -> int:
        """
        Select an action using the ε-greedy strategy.

        With probability ε  → explore: pick a random action
        With probability 1-ε → exploit: pick the best known action

        Returns
        -------
        action : int
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        return self.get_best_action(state)

    def select_greedy_action(self, state) -> int:
        """
        Always select the best known action (no exploration).
        Used when evaluating the trained agent.
        """
        return self.get_best_action(state)

    # ------------------------------------------------------------------
    # Q-Value Update
    # ------------------------------------------------------------------
    def update(self, state, action: int, reward: float, next_state) -> float:
        """
        Apply the Q-learning update rule:
            Q(s,a) ← Q(s,a) + α [ r + γ·max_a'Q(s',a') − Q(s,a) ]

        Parameters
        ----------
        state      : current state tuple
        action     : action taken (int)
        reward     : reward received (float)
        next_state : resulting state tuple

        Returns
        -------
        td_error : float — the temporal difference error (useful for logging)
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])

        # Q-learning update (Bellman equation)
        td_target = reward + self.gamma * max_next_q
        td_error  = td_target - current_q
        new_q     = current_q + self.alpha * td_error

        self.q_table[state][action] = new_q
        self.total_updates += 1

        return td_error

    # ------------------------------------------------------------------
    # Epsilon Decay
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """
        Apply epsilon decay after each episode.
        Epsilon will not fall below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_json(self) -> dict:
        """
        Serialise the Q-table and agent parameters to a JSON-compatible dict.

        States (tuples with frozensets) are serialised as strings.
        """
        serialised_table = {}
        for state, values in self.q_table.items():
            key = self._state_to_key(state)
            serialised_table[key] = values

        return {
            "hyperparams": {
                "learning_rate":   self.alpha,
                "discount_factor": self.gamma,
                "epsilon":         self.epsilon,
                "epsilon_min":     self.epsilon_min,
                "epsilon_decay":   self.epsilon_decay,
                "action_count":    self.action_count,
            },
            "total_updates":   self.total_updates,
            "q_table_size":    len(self.q_table),
            "q_table":         serialised_table,
        }

    @classmethod
    def from_json(cls, data: dict) -> "QLearningAgent":
        """Reconstruct an agent from a previously serialised dict."""
        hp = data["hyperparams"]
        agent = cls(
            learning_rate   = hp["learning_rate"],
            discount_factor = hp["discount_factor"],
            epsilon         = hp["epsilon"],
            epsilon_min     = hp["epsilon_min"],
            epsilon_decay   = hp["epsilon_decay"],
            action_count    = hp.get("action_count", ACTION_COUNT),
        )
        agent.total_updates = data.get("total_updates", 0)

        for key, values in data["q_table"].items():
            state = cls._key_to_state(key)
            agent.q_table[state] = values

        return agent

    # ------------------------------------------------------------------
    # State key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _state_to_key(state) -> str:
        """Convert state tuple to a JSON-safe string key."""
        row, col, dirty_fs = state
        dirty_list = sorted(list(dirty_fs))
        return json.dumps([row, col, dirty_list], separators=(',', ':'))

    @staticmethod
    def _key_to_state(key: str):
        """Reconstruct a state tuple from a string key."""
        data = json.loads(key)
        row, col, dirty_list = data
        dirty_fs = frozenset(tuple(d) for d in dirty_list)
        return (row, col, dirty_fs)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_state_info(self, state) -> dict:
        """Return a human-readable dict of Q-values for a given state."""
        self._ensure_state(state)
        values = self.q_table[state]
        best   = self.get_best_action(state)
        return {
            "state":       str(state),
            "q_values":    {ACTIONS[i]: round(v, 4) for i, v in enumerate(values)},
            "best_action": ACTIONS[best],
            "best_action_id": best,
        }

    def reset(self):
        """Clear the Q-table and reset all parameters to defaults."""
        self.q_table       = {}
        self.total_updates = 0

    def __repr__(self):
        return (
            f"QLearningAgent("
            f"α={self.alpha}, γ={self.gamma}, ε={self.epsilon:.4f}, "
            f"states_known={len(self.q_table)}, updates={self.total_updates})"
        )
