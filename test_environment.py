"""
test_environment.py
-------------------
Tests for VacuumEnv: movement, collision, cleaning, episode termination.
Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from backend.environment.vacuum_env import VacuumEnv
from backend.config import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_CLEAN


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def simple_config():
    return {
        "size":         5,
        "vacuum_start": [2, 2],
        "dirty_cells":  [[0, 0], [4, 4]],
        "obstacles":    [[1, 1], [3, 3]],
    }

@pytest.fixture
def env(simple_config):
    e = VacuumEnv(simple_config)
    e.reset()
    return e


# ── Init & Reset ──────────────────────────────────────────────────────────
def test_initial_position(env):
    assert env.vacuum_pos == (2, 2)

def test_reset_restores_dirty_cells(env):
    env.step(ACTION_CLEAN)  # at (2,2) which is clean → negative reward
    env.reset()
    assert env.dirty_cells == {(0, 0), (4, 4)}

def test_reset_returns_state(env):
    state = env.reset()
    r, c, dirty = state
    assert (r, c) == (2, 2)
    assert dirty == frozenset({(0,0), (4,4)})


# ── Movement ─────────────────────────────────────────────────────────────
def test_move_up(env):
    state, reward, done, info = env.step(ACTION_UP)
    assert env.vacuum_pos == (1, 2)
    assert reward == -1        # normal move

def test_move_down(env):
    env.step(ACTION_DOWN)
    assert env.vacuum_pos == (3, 2)

def test_move_left(env):
    env.step(ACTION_LEFT)
    assert env.vacuum_pos == (2, 1)

def test_move_right(env):
    env.step(ACTION_RIGHT)
    assert env.vacuum_pos == (2, 3)


# ── Wall Collision ────────────────────────────────────────────────────────
def test_wall_top():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[4,4]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    _, reward, _, info = env.step(ACTION_UP)
    assert env.vacuum_pos == (0, 0)   # didn't move
    assert reward == -5
    assert info["collision"] is True
    assert env.collisions == 1

def test_wall_left():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[4,4]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    _, reward, _, info = env.step(ACTION_LEFT)
    assert env.vacuum_pos == (0, 0)
    assert reward == -5

def test_wall_bottom():
    config = {"size": 5, "vacuum_start": [4,4], "dirty_cells": [[0,0]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    env.step(ACTION_DOWN)
    assert env.vacuum_pos == (4, 4)

def test_wall_right():
    config = {"size": 5, "vacuum_start": [4,4], "dirty_cells": [[0,0]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    env.step(ACTION_RIGHT)
    assert env.vacuum_pos == (4, 4)


# ── Obstacle Collision ────────────────────────────────────────────────────
def test_obstacle_collision(env):
    # env starts at (2,2); obstacle is at (1,1) and (3,3)
    # Move up → (1,2) — no obstacle
    env.step(ACTION_UP)   # now at (1,2)
    # Move left → (1,1) — obstacle!
    _, reward, _, info = env.step(ACTION_LEFT)
    assert env.vacuum_pos == (1, 2)
    assert reward == -5
    assert info["collision"] is True


# ── Cleaning ──────────────────────────────────────────────────────────────
def test_clean_dirty_cell():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[0,0]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    _, reward, done, info = env.step(ACTION_CLEAN)
    assert reward == 10 + 20    # clean_dirty + complete_room (only 1 dirty cell)
    assert info["cleaned"] is True
    assert (0,0) not in env.dirty_cells

def test_clean_already_clean(env):
    # (2,2) is not dirty
    _, reward, _, _ = env.step(ACTION_CLEAN)
    assert reward == -3

def test_clean_all_ends_episode():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[0,0]], "obstacles": []}
    env = VacuumEnv(config)
    env.reset()
    _, _, done, _ = env.step(ACTION_CLEAN)
    assert done is True


# ── Max Steps ─────────────────────────────────────────────────────────────
def test_max_steps_termination():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[4,4]], "obstacles": []}
    env = VacuumEnv(config, max_steps=3)
    env.reset()
    for _ in range(3):
        _, _, done, _ = env.step(ACTION_RIGHT)
    assert done is True


# ── Render & Config ───────────────────────────────────────────────────────
def test_render_returns_correct_size(env):
    grid = env.render()
    assert len(grid) == 5
    assert all(len(row) == 5 for row in grid)

def test_render_vacuum_position(env):
    grid = env.render()
    assert grid[2][2] == VacuumEnv.CELL_VACUUM

def test_render_obstacle(env):
    grid = env.render()
    assert grid[1][1] == VacuumEnv.CELL_OBSTACLE

def test_completion_pct(env):
    assert env.get_completion_pct() == 0.0
    # Move to (0,0) and clean
    env.step(ACTION_UP); env.step(ACTION_UP)   # row 0
    env.step(ACTION_LEFT); env.step(ACTION_LEFT)  # col 0
    env.step(ACTION_CLEAN)
    assert env.get_completion_pct() == 50.0


# ── Validation ────────────────────────────────────────────────────────────
def test_invalid_start_outside_grid():
    config = {"size": 5, "vacuum_start": [10,0], "dirty_cells": [[0,0]], "obstacles": []}
    with pytest.raises(ValueError):
        VacuumEnv(config)

def test_dirty_on_obstacle_raises():
    config = {"size": 5, "vacuum_start": [0,0], "dirty_cells": [[1,1]], "obstacles": [[1,1]]}
    with pytest.raises(ValueError):
        VacuumEnv(config)

def test_vacuum_on_obstacle_raises():
    config = {"size": 5, "vacuum_start": [1,1], "dirty_cells": [[0,0]], "obstacles": [[1,1]]}
    with pytest.raises(ValueError):
        VacuumEnv(config)
