"""
evaluator.py
------------
Functions to run the random agent and trained agent and record
full step-by-step traces for visualization in the frontend.

Each trace step contains:
    {
        "step":          int,
        "vacuum_pos":    [row, col],
        "dirty_cells":   [[r,c], ...],
        "action":        int,
        "action_name":   str,
        "reward":        float,
        "cumulative_reward": float,
        "done":          bool,
        "collision":     bool,
        "cleaned":       bool,
        "completion_pct": float,
        "grid":          [[cell_type, ...], ...]
    }
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.environment.vacuum_env import VacuumEnv
from backend.agent.q_learning_agent import QLearningAgent
from backend.config import ACTION_COUNT, ACTIONS


def run_random_agent(env_config: dict, max_steps: int = 200) -> dict:
    """
    Run a completely random agent on the environment.

    The agent picks actions uniformly at random — this is the baseline
    (before any training).

    Returns
    -------
    dict with:
        steps_trace     : list of step dicts
        total_reward    : float
        steps           : int
        collisions      : int
        completion_pct  : float
        cleaned_cells   : int
        total_dirty     : int
    """
    env   = VacuumEnv(env_config, max_steps=max_steps)
    state = env.reset()

    trace          = []
    cumulative_rew = 0.0
    done           = False

    # Initial frame (before any action)
    trace.append(_make_frame(0, env, action=None, reward=0.0,
                              cumulative=0.0, collision=False, cleaned=False))

    step = 0
    while not done:
        step  += 1
        action = random.randint(0, ACTION_COUNT - 1)
        next_state, reward, done, info = env.step(action)

        cumulative_rew += reward
        trace.append(_make_frame(
            step, env, action=action, reward=reward,
            cumulative=cumulative_rew,
            collision=info["collision"],
            cleaned=info["cleaned"],
        ))
        state = next_state

    total_dirty   = len(env.initial_dirty)
    cleaned_cells = total_dirty - len(env.dirty_cells)

    return {
        "agent_type":    "random",
        "steps_trace":   trace,
        "total_reward":  round(cumulative_rew, 4),
        "steps":         env.steps,
        "collisions":    env.collisions,
        "completion_pct": env.get_completion_pct(),
        "cleaned_cells": cleaned_cells,
        "total_dirty":   total_dirty,
    }


def run_trained_agent(agent: QLearningAgent, env_config: dict,
                      max_steps: int = 200) -> dict:
    """
    Run the trained Q-learning agent using its learned policy (greedy).

    Parameters
    ----------
    agent       : QLearningAgent — must have been trained
    env_config  : dict — same grid config used during training
    max_steps   : int — safety limit on steps

    Returns
    -------
    dict with steps_trace and summary metrics.
    """
    env   = VacuumEnv(env_config, max_steps=max_steps)
    state = env.reset()

    trace          = []
    cumulative_rew = 0.0
    done           = False

    # Initial frame
    trace.append(_make_frame(0, env, action=None, reward=0.0,
                              cumulative=0.0, collision=False, cleaned=False))

    step = 0
    while not done:
        step += 1
        # Greedy action — pure exploitation of learned Q-values
        action = agent.select_greedy_action(state)
        next_state, reward, done, info = env.step(action)

        cumulative_rew += reward
        trace.append(_make_frame(
            step, env, action=action, reward=reward,
            cumulative=cumulative_rew,
            collision=info["collision"],
            cleaned=info["cleaned"],
        ))
        state = next_state

    total_dirty   = len(env.initial_dirty)
    cleaned_cells = total_dirty - len(env.dirty_cells)

    return {
        "agent_type":    "trained",
        "steps_trace":   trace,
        "total_reward":  round(cumulative_rew, 4),
        "steps":         env.steps,
        "collisions":    env.collisions,
        "completion_pct": env.get_completion_pct(),
        "cleaned_cells": cleaned_cells,
        "total_dirty":   total_dirty,
    }


def build_comparison(random_result: dict, trained_result: dict) -> dict:
    """
    Build a structured comparison between random and trained agent results.
    """
    def eff(result: dict) -> float:
        """
        Cleaning efficiency = completion % / max(steps, 1).
        Higher is better (more dirt cleaned per step).
        """
        return round(result["completion_pct"] / max(result["steps"], 1), 4)

    return {
        "random": {
            "total_reward":   random_result["total_reward"],
            "steps":          random_result["steps"],
            "completion_pct": random_result["completion_pct"],
            "collisions":     random_result["collisions"],
            "efficiency":     eff(random_result),
        },
        "trained": {
            "total_reward":   trained_result["total_reward"],
            "steps":          trained_result["steps"],
            "completion_pct": trained_result["completion_pct"],
            "collisions":     trained_result["collisions"],
            "efficiency":     eff(trained_result),
        },
    }


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------
def _make_frame(step: int, env: VacuumEnv, action, reward: float,
                cumulative: float, collision: bool, cleaned: bool) -> dict:
    """Create one step frame for the animation trace."""
    return {
        "step":              step,
        "vacuum_pos":        list(env.vacuum_pos),
        "dirty_cells":       [list(p) for p in sorted(env.dirty_cells)],
        "action":            action,
        "action_name":       ACTIONS.get(action, "START") if action is not None else "START",
        "reward":            round(reward, 4),
        "cumulative_reward": round(cumulative, 4),
        "collision":         collision,
        "cleaned":           cleaned,
        "done":              env.done,
        "completion_pct":    env.get_completion_pct(),
        "collisions_so_far": env.collisions,
        "grid":              env.render(),
    }
