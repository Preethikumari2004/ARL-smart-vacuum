"""
room_generator.py
-----------------
Provides preset room configurations and a random room generator.

The random generator guarantees:
  - Vacuum does not start on an obstacle.
  - Dirty cells do not overlap obstacles or the vacuum start.
  - All positions are inside the grid.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.config import PRESET_ROOMS


def get_preset(name: str) -> dict:
    """
    Return a preset room configuration by name.

    Parameters
    ----------
    name : str
        One of 'simple', 'medium', 'complex'

    Returns
    -------
    dict with keys: size, vacuum_start, dirty_cells, obstacles
    """
    name = name.lower()
    if name not in PRESET_ROOMS:
        available = list(PRESET_ROOMS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    preset = PRESET_ROOMS[name]
    # Return a deep copy so callers cannot mutate the config
    return {
        "size":          preset["size"],
        "vacuum_start":  list(preset["vacuum_start"]),
        "dirty_cells":   [list(c) for c in preset["dirty_cells"]],
        "obstacles":     [list(c) for c in preset["obstacles"]],
        "name":          preset["name"],
        "description":   preset["description"],
    }


def list_presets() -> list:
    """Return metadata for all available presets."""
    return [
        {
            "key":         k,
            "name":        v["name"],
            "description": v["description"],
            "size":        v["size"],
        }
        for k, v in PRESET_ROOMS.items()
    ]


def generate_random_room(
    size: int = 5,
    n_dirty: int = 5,
    n_obstacles: int = 3,
    seed: int = None,
) -> dict:
    """
    Generate a random valid room configuration.

    Parameters
    ----------
    size        : int  — grid side length (N×N), minimum 3
    n_dirty     : int  — number of dirty cells to place
    n_obstacles : int  — number of obstacles to place
    seed        : int  — optional random seed for reproducibility

    Returns
    -------
    dict with keys: size, vacuum_start, dirty_cells, obstacles

    Raises
    ------
    ValueError  — if the grid is too small to fit all elements
    """
    if size < 3:
        raise ValueError("Grid size must be at least 3 for random generation.")

    total_cells = size * size
    if 1 + n_dirty + n_obstacles > total_cells:
        raise ValueError(
            f"Cannot fit vacuum + {n_dirty} dirty + {n_obstacles} obstacles "
            f"in a {size}×{size} grid ({total_cells} cells)."
        )

    rng = random.Random(seed)

    all_cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(all_cells)

    # Pick vacuum start first
    vacuum_start = list(all_cells[0])
    used = {all_cells[0]}
    idx = 1

    # Pick obstacles (not on vacuum start)
    obstacles = []
    while len(obstacles) < n_obstacles and idx < len(all_cells):
        pos = all_cells[idx]
        idx += 1
        if pos not in used:
            obstacles.append(list(pos))
            used.add(pos)

    # Pick dirty cells (not on vacuum or obstacles)
    dirty_cells = []
    while len(dirty_cells) < n_dirty and idx < len(all_cells):
        pos = all_cells[idx]
        idx += 1
        if pos not in used:
            dirty_cells.append(list(pos))
            used.add(pos)

    if len(dirty_cells) < n_dirty:
        raise ValueError(
            f"Could only place {len(dirty_cells)} dirty cells (requested {n_dirty})."
        )

    return {
        "size":          size,
        "vacuum_start":  vacuum_start,
        "dirty_cells":   dirty_cells,
        "obstacles":     obstacles,
    }
