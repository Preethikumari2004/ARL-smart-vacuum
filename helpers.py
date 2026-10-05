"""
helpers.py
----------
Utility functions for state serialisation, Q-table I/O, and validation.
"""

import json
import os


def state_to_str(state) -> str:
    """
    Convert a state tuple (row, col, frozenset_dirty) to a compact string.
    Used as a dict key or display label.
    """
    row, col, dirty_fs = state
    dirty_sorted = sorted(list(dirty_fs))
    return f"({row},{col})|dirty={dirty_sorted}"


def validate_grid_config(config: dict) -> list:
    """
    Validate a grid configuration dict.

    Returns a list of error strings.  Empty list means valid.

    Parameters
    ----------
    config : dict with keys size, vacuum_start, dirty_cells, obstacles
    """
    errors = []

    size = config.get("size")
    if not isinstance(size, int) or size < 2:
        errors.append("Grid size must be an integer ≥ 2.")
        return errors  # Cannot validate positions without a valid size

    def in_bounds(pos):
        return (isinstance(pos, (list, tuple)) and len(pos) == 2
                and 0 <= pos[0] < size and 0 <= pos[1] < size)

    vacuum_start = config.get("vacuum_start", [])
    if not in_bounds(vacuum_start):
        errors.append(f"vacuum_start {vacuum_start} is out of bounds for a {size}×{size} grid.")

    obstacles = config.get("obstacles", [])
    obstacle_set = set()
    for obs in obstacles:
        if not in_bounds(obs):
            errors.append(f"Obstacle {obs} is out of bounds.")
        else:
            obstacle_set.add(tuple(obs))

    if tuple(vacuum_start) in obstacle_set:
        errors.append("Vacuum start position overlaps an obstacle.")

    dirty_cells = config.get("dirty_cells", [])
    for dc in dirty_cells:
        if not in_bounds(dc):
            errors.append(f"Dirty cell {dc} is out of bounds.")
        elif tuple(dc) in obstacle_set:
            errors.append(f"Dirty cell {dc} overlaps an obstacle.")

    if len(dirty_cells) == 0:
        errors.append("No dirty cells defined — nothing to clean.")

    return errors


def validate_hyperparams(params: dict) -> list:
    """
    Validate hyperparameter values.

    Returns a list of error strings.  Empty list means valid.
    """
    errors = []

    lr = params.get("learning_rate", 0)
    if not (0 < lr <= 1):
        errors.append(f"learning_rate must be in (0, 1], got {lr}.")

    gamma = params.get("discount_factor", 0)
    if not (0 <= gamma < 1):
        errors.append(f"discount_factor must be in [0, 1), got {gamma}.")

    eps = params.get("epsilon", 0)
    if not (0 <= eps <= 1):
        errors.append(f"epsilon must be in [0, 1], got {eps}.")

    eps_min = params.get("epsilon_min", 0)
    if not (0 <= eps_min <= 1):
        errors.append(f"epsilon_min must be in [0, 1], got {eps_min}.")

    eps_decay = params.get("epsilon_decay", 0)
    if not (0 < eps_decay < 1):
        errors.append(f"epsilon_decay must be in (0, 1), got {eps_decay}.")

    episodes = params.get("episodes", 0)
    if not isinstance(episodes, int) or episodes < 1:
        errors.append(f"episodes must be a positive integer, got {episodes}.")

    max_steps = params.get("max_steps", 0)
    if not isinstance(max_steps, int) or max_steps < 1:
        errors.append(f"max_steps must be a positive integer, got {max_steps}.")

    return errors


def compute_rolling_average(values: list, window: int = 100) -> list:
    """
    Compute a rolling (moving) average over a list of floats.

    Returns a list of the same length.  Values before the first full
    window are averaged over whatever data is available.
    """
    averages = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        avg   = sum(values[start: i + 1]) / (i - start + 1)
        averages.append(round(avg, 4))
    return averages


def save_q_table(agent, filepath: str):
    """
    Save the agent's Q-table to a JSON file.

    Parameters
    ----------
    agent    : QLearningAgent instance
    filepath : absolute path to the output .json file
    """
    data = agent.to_json()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_q_table(agent_cls, filepath: str):
    """
    Load an agent's Q-table from a JSON file.

    Parameters
    ----------
    agent_cls : QLearningAgent class (not instance)
    filepath  : path to the .json file

    Returns
    -------
    QLearningAgent instance
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Q-table file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return agent_cls.from_json(data)
