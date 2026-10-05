"""
test_q_learning.py
------------------
Tests for QLearningAgent: initialisation, action selection, Q-update, epsilon decay.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from backend.agent.q_learning_agent import QLearningAgent
from backend.config import ACTION_COUNT, ACTIONS


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def agent():
    return QLearningAgent(
        learning_rate   = 0.1,
        discount_factor = 0.9,
        epsilon         = 1.0,
        epsilon_min     = 0.01,
        epsilon_decay   = 0.9,
    )

def make_state(row, col, dirty=None):
    return (row, col, frozenset(dirty or []))


# ── Initialisation ────────────────────────────────────────────────────────
def test_agent_init(agent):
    assert agent.alpha  == 0.1
    assert agent.gamma  == 0.9
    assert agent.epsilon == 1.0
    assert len(agent.q_table) == 0

def test_q_table_starts_empty(agent):
    assert agent.q_table == {}

def test_get_q_value_unseen_state_returns_zero(agent):
    s = make_state(0, 0, [(1, 1)])
    assert agent.get_q_value(s, 0) == 0.0

def test_get_all_q_values_length(agent):
    s = make_state(0, 0)
    vals = agent.get_all_q_values(s)
    assert len(vals) == ACTION_COUNT
    assert all(v == 0.0 for v in vals)


# ── Action Selection ──────────────────────────────────────────────────────
def test_select_action_with_full_epsilon_is_random(agent):
    """With epsilon=1.0, all actions should appear random over many calls."""
    s       = make_state(0, 0)
    actions = set(agent.select_action(s) for _ in range(200))
    assert len(actions) > 1    # should see multiple distinct actions

def test_select_greedy_action_all_zero_returns_int(agent):
    s = make_state(1, 1)
    a = agent.select_greedy_action(s)
    assert isinstance(a, int)
    assert 0 <= a < ACTION_COUNT

def test_select_greedy_picks_max(agent):
    """After manually setting Q-values, greedy should pick the highest."""
    s = make_state(0, 0)
    agent._ensure_state(s)
    agent.q_table[s][2] = 99.0     # action 2 (LEFT) is best
    assert agent.select_greedy_action(s) == 2


# ── Q-Value Update ────────────────────────────────────────────────────────
def test_q_update_changes_value(agent):
    s  = make_state(0, 0)
    s2 = make_state(0, 1)
    agent.update(s, 0, 10.0, s2)
    # Q(s,0) should be non-zero now
    assert agent.get_q_value(s, 0) != 0.0

def test_q_update_bellman_calculation(agent):
    """
    Manually verify the Bellman update:
    Q(s,a) = Q(s,a) + α [r + γ·max_Q(s') - Q(s,a)]
    With Q(s,a)=0, Q(s')=[0,…], r=10:
    new = 0 + 0.1 * [10 + 0.9*0 - 0] = 1.0
    """
    s  = make_state(0, 0)
    s2 = make_state(0, 1)
    agent.update(s, 0, 10.0, s2)
    assert abs(agent.get_q_value(s, 0) - 1.0) < 1e-9

def test_q_update_with_nonzero_next_state(agent):
    """Verify γ·max_Q(s') is incorporated."""
    s  = make_state(0, 0)
    s2 = make_state(0, 1)
    agent._ensure_state(s2)
    agent.q_table[s2][3] = 100.0   # max_Q(s') = 100
    # new = 0 + 0.1 * [5 + 0.9*100 - 0] = 0.1 * 95 = 9.5
    agent.update(s, 1, 5.0, s2)
    assert abs(agent.get_q_value(s, 1) - 9.5) < 1e-9

def test_total_updates_increments(agent):
    s  = make_state(0, 0)
    s2 = make_state(1, 0)
    for _ in range(5):
        agent.update(s, 0, 1.0, s2)
    assert agent.total_updates == 5


# ── Epsilon Decay ─────────────────────────────────────────────────────────
def test_epsilon_decays(agent):
    initial = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial

def test_epsilon_does_not_go_below_min(agent):
    agent.epsilon = 0.015
    agent.decay_epsilon()
    assert agent.epsilon >= agent.epsilon_min

def test_epsilon_floor(agent):
    agent.epsilon = 0.001  # below epsilon_min
    agent.decay_epsilon()
    assert agent.epsilon == agent.epsilon_min


# ── Serialisation ─────────────────────────────────────────────────────────
def test_serialise_and_restore(agent):
    s  = make_state(1, 2, [(0,0),(3,4)])
    s2 = make_state(1, 3)
    agent.update(s, 4, 15.0, s2)

    data      = agent.to_json()
    restored  = QLearningAgent.from_json(data)

    assert len(restored.q_table) == len(agent.q_table)
    assert abs(restored.get_q_value(s, 4) - agent.get_q_value(s, 4)) < 1e-9
    assert restored.alpha == agent.alpha
    assert restored.epsilon == agent.epsilon

def test_state_key_roundtrip():
    s    = (2, 3, frozenset([(0,0),(1,2)]))
    key  = QLearningAgent._state_to_key(s)
    back = QLearningAgent._key_to_state(key)
    assert back == s


# ── Reset ─────────────────────────────────────────────────────────────────
def test_reset_clears_q_table(agent):
    s = make_state(0, 0)
    agent._ensure_state(s)
    agent.reset()
    assert len(agent.q_table) == 0
    assert agent.total_updates == 0

def test_get_state_info(agent):
    s = make_state(2, 2, [(1,1)])
    info = agent.get_state_info(s)
    assert "q_values" in info
    assert "best_action" in info
    assert "best_action_id" in info
    assert len(info["q_values"]) == ACTION_COUNT
