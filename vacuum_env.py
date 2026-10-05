"""
vacuum_env.py
-------------
The grid-based vacuum cleaning environment for Q-Learning.

State representation:
    A tuple: (vacuum_row, vacuum_col, dirty_cells_frozenset)
    Example: (0, 2, frozenset({(1,3), (4,0)}))

    This state fully captures:
      - Where the vacuum is
      - Which cells are still dirty
    Obstacles are fixed and part of the environment definition,
    not the dynamic state.

Actions:
    0 = UP
    1 = DOWN
    2 = LEFT
    3 = RIGHT
    4 = CLEAN
"""

import copy
import sys
import os

# Allow running this file standalone for quick testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.config import (
    REWARDS, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_CLEAN, ACTION_DELTAS, ACTION_COUNT, ACTIONS
)


class VacuumEnv:
    """
    Custom grid-based vacuum cleaning environment.

    The agent (vacuum) operates on an N×N grid.  Each cell is one of:
        - Clean floor
        - Dirty floor  (must be cleaned)
        - Obstacle      (impassable)
        - Vacuum        (current position)

    One episode ends when:
        (a) all dirty cells have been cleaned, OR
        (b) the agent exceeds max_steps
    """

    # Cell type constants (used internally and for rendering)
    CELL_CLEAN    = "clean"
    CELL_DIRTY    = "dirty"
    CELL_OBSTACLE = "obstacle"
    CELL_VACUUM   = "vacuum"

    def __init__(self, grid_config: dict, max_steps: int = 200):
        """
        Initialise the environment from a grid configuration dict.

        Expected keys in grid_config:
            size         (int)        : grid side length N
            vacuum_start ([row, col]) : starting position
            dirty_cells  ([[r,c], …]) : initially dirty cells
            obstacles    ([[r,c], …]) : obstacle positions
        """
        self.size          = grid_config["size"]
        self.start_pos     = tuple(grid_config["vacuum_start"])
        self.initial_dirty = frozenset(tuple(c) for c in grid_config["dirty_cells"])
        self.obstacles     = frozenset(tuple(c) for c in grid_config["obstacles"])
        self.max_steps     = max_steps

        # Validate config on creation
        self._validate_config()

        # Runtime state (populated by reset())
        self.vacuum_pos  = self.start_pos
        self.dirty_cells = set(self.initial_dirty)
        self.steps       = 0
        self.total_reward = 0.0
        self.collisions  = 0
        self.done        = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_config(self):
        """Raise ValueError for obviously invalid configurations."""
        n = self.size
        if n < 2:
            raise ValueError("Grid size must be at least 2.")

        def in_bounds(pos):
            return 0 <= pos[0] < n and 0 <= pos[1] < n

        if not in_bounds(self.start_pos):
            raise ValueError(f"Vacuum start {self.start_pos} is outside the grid.")
        if self.start_pos in self.obstacles:
            raise ValueError("Vacuum start position overlaps an obstacle.")

        for pos in self.initial_dirty:
            if not in_bounds(pos):
                raise ValueError(f"Dirty cell {pos} is outside the grid.")
            if pos in self.obstacles:
                raise ValueError(f"Dirty cell {pos} overlaps an obstacle.")

        for pos in self.obstacles:
            if not in_bounds(pos):
                raise ValueError(f"Obstacle {pos} is outside the grid.")

    # ------------------------------------------------------------------
    # Core RL Interface
    # ------------------------------------------------------------------
    def reset(self):
        """
        Reset the environment to its initial state.
        Returns the initial state tuple.
        """
        self.vacuum_pos   = self.start_pos
        self.dirty_cells  = set(self.initial_dirty)
        self.steps        = 0
        self.total_reward = 0.0
        self.collisions   = 0
        self.done         = False
        return self.get_state()

    def step(self, action: int):
        """
        Apply an action and return (next_state, reward, done, info).

        Parameters
        ----------
        action : int
            One of 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=CLEAN

        Returns
        -------
        next_state : tuple
        reward     : float
        done       : bool
        info       : dict  (extra diagnostic information)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        reward = 0.0
        collision = False
        cleaned   = False

        if action == ACTION_CLEAN:
            # ---- CLEAN action ----------------------------------------
            if self.vacuum_pos in self.dirty_cells:
                self.dirty_cells.remove(self.vacuum_pos)
                reward  += REWARDS["clean_dirty"]
                cleaned  = True

                # Bonus if room is now completely clean
                if len(self.dirty_cells) == 0:
                    reward += REWARDS["complete_room"]
            else:
                # Trying to clean an already-clean (or obstacle-free) cell
                reward += REWARDS["clean_already_clean"]

        else:
            # ---- Movement action -------------------------------------
            dr, dc = ACTION_DELTAS[action]
            new_row = self.vacuum_pos[0] + dr
            new_col = self.vacuum_pos[1] + dc
            new_pos = (new_row, new_col)

            if not self._in_bounds(new_pos):
                # Hit a wall
                reward    += REWARDS["hit_wall"]
                collision  = True
            elif new_pos in self.obstacles:
                # Hit an obstacle
                reward    += REWARDS["hit_obstacle"]
                collision  = True
            else:
                # Valid move
                self.vacuum_pos = new_pos
                reward += REWARDS["normal_move"]

        if collision:
            self.collisions += 1

        self.steps        += 1
        self.total_reward += reward

        # Check termination
        all_clean = len(self.dirty_cells) == 0
        max_steps_reached = self.steps >= self.max_steps
        self.done = all_clean or max_steps_reached

        info = {
            "cleaned":          cleaned,
            "collision":        collision,
            "dirty_remaining":  len(self.dirty_cells),
            "total_dirty":      len(self.initial_dirty),
            "completion_pct":   self.get_completion_pct(),
            "steps":            self.steps,
        }

        return self.get_state(), reward, self.done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_state(self):
        """
        Return the current state as a hashable tuple:
            (row, col, frozenset_of_dirty_cells)
        """
        return (self.vacuum_pos[0], self.vacuum_pos[1], frozenset(self.dirty_cells))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_completion_pct(self) -> float:
        """Return the percentage of dirty cells that have been cleaned."""
        total = len(self.initial_dirty)
        if total == 0:
            return 100.0
        cleaned = total - len(self.dirty_cells)
        return round(cleaned / total * 100.0, 2)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> list:
        """
        Return the grid as a 2-D list of cell-type strings.
        Each element is one of: 'clean', 'dirty', 'obstacle', 'vacuum'.
        Useful for building the JSON response to the frontend.
        """
        grid = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.vacuum_pos:
                    row.append(self.CELL_VACUUM)
                elif pos in self.obstacles:
                    row.append(self.CELL_OBSTACLE)
                elif pos in self.dirty_cells:
                    row.append(self.CELL_DIRTY)
                else:
                    row.append(self.CELL_CLEAN)
            grid.append(row)
        return grid

    def to_dict(self) -> dict:
        """Serialise current state to a JSON-friendly dict."""
        return {
            "vacuum_pos":     list(self.vacuum_pos),
            "dirty_cells":    [list(p) for p in sorted(self.dirty_cells)],
            "obstacles":      [list(p) for p in sorted(self.obstacles)],
            "size":           self.size,
            "steps":          self.steps,
            "total_reward":   self.total_reward,
            "collisions":     self.collisions,
            "done":           self.done,
            "completion_pct": self.get_completion_pct(),
            "grid":           self.render(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _in_bounds(self, pos) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def get_config(self) -> dict:
        """Return the environment configuration as a serialisable dict."""
        return {
            "size":          self.size,
            "vacuum_start":  list(self.start_pos),
            "dirty_cells":   [list(p) for p in sorted(self.initial_dirty)],
            "obstacles":     [list(p) for p in sorted(self.obstacles)],
        }

    def __repr__(self):
        grid = self.render()
        symbols = {
            self.CELL_VACUUM:    "🧹",
            self.CELL_DIRTY:     "🟫",
            self.CELL_OBSTACLE:  "⬛",
            self.CELL_CLEAN:     "⬜",
        }
        lines = []
        for row in grid:
            lines.append(" ".join(symbols[c] for c in row))
        return "\n".join(lines)
