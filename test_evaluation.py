"""
test_evaluation.py
------------------
Tests for evaluator: random agent and trained agent produce valid results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from backend.evaluation.evaluator import run_random_agent, run_trained_agent, build_comparison
from backend.training.trainer     import TrainingSession
from backend.agent.q_learning_agent import QLearningAgent


@pytest.fixture
def env_config():
    return {
        "size":         4,
        "vacuum_start": [0, 0],
        "dirty_cells":  [[3, 3], [1, 2]],
        "obstacles":    [[2, 2]],
    }

@pytest.fixture
def trained_agent(env_config):
    session = TrainingSession(env_config, {
        "episodes": 100, "learning_rate": 0.1, "discount_factor": 0.9,
        "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.9,
        "max_steps": 40,
    })
    session.train()
    return session.get_agent()


# ── Random Agent ──────────────────────────────────────────────────────────
def test_random_agent_returns_result(env_config):
    res = run_random_agent(env_config, max_steps=50)
    assert res is not None

def test_random_agent_has_trace(env_config):
    res = run_random_agent(env_config, max_steps=50)
    assert "steps_trace" in res
    assert len(res["steps_trace"]) > 0

def test_random_agent_steps_within_limit(env_config):
    res = run_random_agent(env_config, max_steps=50)
    assert res["steps"] <= 50

def test_random_agent_completion_in_range(env_config):
    res = run_random_agent(env_config, max_steps=50)
    assert 0.0 <= res["completion_pct"] <= 100.0

def test_random_agent_type_label(env_config):
    res = run_random_agent(env_config, max_steps=50)
    assert res["agent_type"] == "random"

def test_random_trace_has_correct_fields(env_config):
    res = run_random_agent(env_config, max_steps=20)
    frame = res["steps_trace"][1]   # step 1 (index 0 is START)
    required = {"step","vacuum_pos","dirty_cells","action","action_name",
                "reward","cumulative_reward","collision","cleaned","done",
                "completion_pct","collisions_so_far","grid"}
    assert required.issubset(frame.keys())

def test_random_trace_start_frame(env_config):
    res = run_random_agent(env_config, max_steps=20)
    frame0 = res["steps_trace"][0]
    assert frame0["step"] == 0
    assert frame0["action_name"] == "START"


# ── Trained Agent ─────────────────────────────────────────────────────────
def test_trained_agent_returns_result(env_config, trained_agent):
    res = run_trained_agent(trained_agent, env_config, max_steps=50)
    assert res is not None

def test_trained_agent_type_label(env_config, trained_agent):
    res = run_trained_agent(trained_agent, env_config, max_steps=50)
    assert res["agent_type"] == "trained"

def test_trained_agent_actions_from_q_table(env_config, trained_agent):
    """Each action in the trained trace should be a known action index."""
    res = run_trained_agent(trained_agent, env_config, max_steps=50)
    for frame in res["steps_trace"][1:]:
        assert frame["action"] in [0, 1, 2, 3, 4]

def test_trained_agent_has_trace(env_config, trained_agent):
    res = run_trained_agent(trained_agent, env_config, max_steps=50)
    assert len(res["steps_trace"]) > 0


# ── Comparison ────────────────────────────────────────────────────────────
def test_comparison_structure(env_config, trained_agent):
    rand   = run_random_agent(env_config, max_steps=50)
    trained = run_trained_agent(trained_agent, env_config, max_steps=50)
    comp   = build_comparison(rand, trained)

    assert "random"  in comp
    assert "trained" in comp
    for key in ("total_reward","steps","completion_pct","collisions","efficiency"):
        assert key in comp["random"]
        assert key in comp["trained"]

def test_comparison_efficiency_nonnegative(env_config, trained_agent):
    rand    = run_random_agent(env_config, max_steps=50)
    trained = run_trained_agent(trained_agent, env_config, max_steps=50)
    comp    = build_comparison(rand, trained)
    assert comp["random"]["efficiency"]  >= 0
    assert comp["trained"]["efficiency"] >= 0
