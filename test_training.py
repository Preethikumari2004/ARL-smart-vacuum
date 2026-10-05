"""
test_training.py
----------------
Tests for the TrainingSession: runs complete training and verifies
that metrics are populated and the Q-table is updated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from backend.training.trainer import TrainingSession


@pytest.fixture
def small_config():
    """A small 3x3 room for fast training tests."""
    return {
        "size":         3,
        "vacuum_start": [0, 0],
        "dirty_cells":  [[2, 2], [1, 1]],
        "obstacles":    [],
    }

@pytest.fixture
def tiny_hp():
    return {
        "episodes":        50,
        "learning_rate":   0.1,
        "discount_factor": 0.9,
        "epsilon":         1.0,
        "epsilon_min":     0.01,
        "epsilon_decay":   0.9,
        "max_steps":       30,
    }


# ── Training runs ─────────────────────────────────────────────────────────
def test_training_completes(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert results is not None

def test_correct_episode_count(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert len(results["episode_rewards"]) == tiny_hp["episodes"]
    assert len(results["episode_steps"])   == tiny_hp["episodes"]

def test_q_table_grows_during_training(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert results["q_table_size"] > 0

def test_q_table_updates_occur(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert results["total_q_updates"] > 0

def test_epsilon_decays(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert results["final_epsilon"] < tiny_hp["epsilon"]

def test_best_reward_in_results(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert "best_reward"  in results
    assert "best_episode" in results

def test_rolling_avg_length_matches_episodes(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    assert len(results["rolling_avg_rewards"]) == tiny_hp["episodes"]

def test_episode_completion_values_in_range(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    results = session.train()
    for pct in results["episode_completion"]:
        assert 0.0 <= pct <= 100.0

def test_agent_accessible_after_training(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    session.train()
    agent = session.get_agent()
    assert agent is not None
    assert len(agent.q_table) > 0

def test_trained_flag(small_config, tiny_hp):
    session = TrainingSession(small_config, tiny_hp)
    assert session.trained is False
    session.train()
    assert session.trained is True
