"""
config.py
---------
Central configuration for the Smart Vacuum RL project.
All reward values, default hyperparameters, and grid settings live here.
Change these values to experiment with different learning behaviours.
"""

# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------
REWARDS = {
    "clean_dirty":        10,    # Agent cleans a dirty cell
    "complete_room":      20,    # Agent cleans ALL dirty cells
    "normal_move":        -1,    # Any movement step (encourages efficiency)
    "hit_wall":           -5,    # Agent tries to move outside the grid
    "hit_obstacle":       -5,    # Agent tries to enter an obstacle cell
    "clean_already_clean": -3,   # Agent tries to CLEAN a cell that is already clean
}

# ---------------------------------------------------------------------------
# Default Hyperparameters for Q-Learning
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    "episodes":        5000,
    "learning_rate":   0.1,      # alpha (α)
    "discount_factor": 0.9,      # gamma (γ)
    "epsilon":         1.0,      # initial exploration rate
    "epsilon_min":     0.01,     # minimum exploration rate
    "epsilon_decay":   0.995,    # multiplicative decay per episode
    "max_steps":       200,      # max steps per episode before forced termination
}

# ---------------------------------------------------------------------------
# Default Grid Configuration
# ---------------------------------------------------------------------------
DEFAULT_GRID = {
    "size":          5,           # NxN grid
    "vacuum_start":  [0, 0],      # [row, col]
    "dirty_cells":   [[0, 2], [1, 3], [2, 1], [3, 4], [4, 2]],
    "obstacles":     [[1, 1], [2, 3]],
}

# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------
ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "CLEAN",
}
ACTION_COUNT = len(ACTIONS)

# Action index lookup
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_CLEAN = 4

# Action deltas for movement (row_delta, col_delta)
ACTION_DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
    ACTION_CLEAN: ( 0,  0),   # CLEAN does not move
}

# ---------------------------------------------------------------------------
# Preset Room Configurations
# ---------------------------------------------------------------------------
PRESET_ROOMS = {
    "simple": {
        "name":        "Simple Room",
        "description": "3 dirty cells, 1 obstacle — great for first training runs.",
        "size":        5,
        "vacuum_start": [0, 0],
        "dirty_cells": [[0, 2], [2, 4], [4, 4]],
        "obstacles":   [[2, 2]],
    },
    "medium": {
        "name":        "Medium Room",
        "description": "5 dirty cells, 3 obstacles — moderate challenge.",
        "size":        5,
        "vacuum_start": [0, 0],
        "dirty_cells": [[0, 2], [1, 3], [2, 0], [3, 4], [4, 1]],
        "obstacles":   [[1, 1], [2, 3], [3, 2]],
    },
    "complex": {
        "name":        "Complex Room",
        "description": "7 dirty cells, 5 obstacles — real challenge for the agent.",
        "size":        6,
        "vacuum_start": [0, 0],
        "dirty_cells": [[0, 3], [1, 5], [2, 1], [2, 4], [3, 0], [4, 3], [5, 5]],
        "obstacles":   [[1, 2], [2, 2], [3, 3], [4, 1], [4, 4]],
    },
}
