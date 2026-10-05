"""
app.py
------
FastAPI application for the Smart Vacuum RL project.

Endpoints:
    GET  /api/health
    GET  /api/environment/presets
    POST /api/environment/create
    POST /api/environment/random
    POST /api/train
    GET  /api/training/results
    POST /api/test/random
    POST /api/test/trained
    GET  /api/q-table/state
    GET  /api/q-table/policy
    POST /api/q-table/save
    POST /api/q-table/load
    POST /api/reset

The frontend is served as static files from /frontend/.
"""

import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
QTABLE_FILE  = os.path.join(BASE_DIR, "saved_q_table.json")

sys.path.insert(0, BASE_DIR)

from backend.environment.vacuum_env   import VacuumEnv
from backend.environment.room_generator import get_preset, list_presets, generate_random_room
from backend.agent.q_learning_agent  import QLearningAgent
from backend.training.trainer        import TrainingSession
from backend.evaluation.evaluator    import run_random_agent, run_trained_agent, build_comparison
from backend.utils.helpers           import validate_grid_config, validate_hyperparams, save_q_table, load_q_table
from backend.config                  import DEFAULT_HYPERPARAMS, DEFAULT_GRID, ACTIONS, REWARDS

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Vacuum RL",
    description="Q-Learning powered vacuum cleaner — academic RL project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state ────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "env_config":       DEFAULT_GRID,
    "agent":            None,           # QLearningAgent after training
    "training_results": None,           # last TrainingSession results dict
    "random_result":    None,
    "trained_result":   None,
}


# ── Pydantic request models ────────────────────────────────────────────────
class GridConfig(BaseModel):
    size:          int              = Field(5,  ge=2, le=10)
    vacuum_start:  List[int]        = Field([0, 0])
    dirty_cells:   List[List[int]]  = Field([[0,2],[2,4],[4,4]])
    obstacles:     List[List[int]]  = Field([[2,2]])


class RandomRoomRequest(BaseModel):
    size:        int = Field(5,  ge=3, le=10)
    n_dirty:     int = Field(5,  ge=1, le=30)
    n_obstacles: int = Field(3,  ge=0, le=20)
    seed:        Optional[int] = None


class TrainRequest(BaseModel):
    env_config:      Optional[Dict[str, Any]] = None
    episodes:        int   = Field(5000, ge=1,    le=50000)
    learning_rate:   float = Field(0.1,  ge=0.001, le=1.0)
    discount_factor: float = Field(0.9,  ge=0.0,   le=0.999)
    epsilon:         float = Field(1.0,  ge=0.0,   le=1.0)
    epsilon_min:     float = Field(0.01, ge=0.001, le=1.0)
    epsilon_decay:   float = Field(0.995,ge=0.001, le=0.9999)
    max_steps:       int   = Field(200,  ge=10,    le=2000)


class TestRequest(BaseModel):
    env_config: Optional[Dict[str, Any]] = None
    max_steps:  int = Field(200, ge=10, le=2000)


class StateQueryRequest(BaseModel):
    row:         int
    col:         int
    dirty_cells: List[List[int]] = []


class SaveRequest(BaseModel):
    filepath: Optional[str] = None


class LoadRequest(BaseModel):
    filepath: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":  "ok",
        "message": "Smart Vacuum RL backend is running.",
        "trained": _state["agent"] is not None,
        "rewards": REWARDS,
    }


# ── Environment ────────────────────────────────────────────────────────────
@app.get("/api/environment/presets")
def get_presets():
    return {"presets": list_presets()}


@app.post("/api/environment/create")
def create_environment(config: GridConfig):
    cfg = config.model_dump()
    errors = validate_grid_config(cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    _state["env_config"] = cfg
    env = VacuumEnv(cfg)
    return {"message": "Environment created.", "env": env.to_dict(), "config": cfg}


@app.post("/api/environment/random")
def create_random_environment(req: RandomRoomRequest):
    try:
        cfg = generate_random_room(
            size=req.size,
            n_dirty=req.n_dirty,
            n_obstacles=req.n_obstacles,
            seed=req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _state["env_config"] = cfg
    env = VacuumEnv(cfg)
    return {"message": "Random room generated.", "env": env.to_dict(), "config": cfg}


@app.get("/api/environment/preset/{name}")
def load_preset(name: str):
    try:
        cfg = get_preset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Strip display-only fields before storing
    env_cfg = {k: cfg[k] for k in ("size", "vacuum_start", "dirty_cells", "obstacles")}
    _state["env_config"] = env_cfg
    env = VacuumEnv(env_cfg)
    return {
        "message": f"Preset '{name}' loaded.",
        "preset":  cfg,
        "env":     env.to_dict(),
        "config":  env_cfg,
    }


@app.get("/api/environment/current")
def get_current_environment():
    cfg = _state["env_config"]
    env = VacuumEnv(cfg)
    return {"env": env.to_dict(), "config": cfg}


# ── Training ───────────────────────────────────────────────────────────────
@app.post("/api/train")
def train_agent(req: TrainRequest):
    env_cfg = req.env_config or _state["env_config"]

    # Validate grid
    errors = validate_grid_config(env_cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    hyperparams = {
        "episodes":        req.episodes,
        "learning_rate":   req.learning_rate,
        "discount_factor": req.discount_factor,
        "epsilon":         req.epsilon,
        "epsilon_min":     req.epsilon_min,
        "epsilon_decay":   req.epsilon_decay,
        "max_steps":       req.max_steps,
    }
    hp_errors = validate_hyperparams(hyperparams)
    if hp_errors:
        raise HTTPException(status_code=400, detail={"errors": hp_errors})

    # Run training
    session = TrainingSession(env_cfg, hyperparams)
    results = session.train()

    # Store agent in memory
    _state["agent"]            = session.get_agent()
    _state["training_results"] = results
    _state["env_config"]       = env_cfg

    # Return a summary (full episode lists are available via /api/training/results)
    return {
        "message":         "Training complete.",
        "episodes":        results["episodes"],
        "best_reward":     results["best_reward"],
        "best_episode":    results["best_episode"],
        "final_epsilon":   results["final_epsilon"],
        "q_table_size":    results["q_table_size"],
        "total_q_updates": results["total_q_updates"],
        "hyperparams":     results["hyperparams"],
    }


@app.get("/api/training/results")
def get_training_results():
    if _state["training_results"] is None:
        raise HTTPException(status_code=404, detail="No training results yet. Run /api/train first.")
    return _state["training_results"]


# ── Evaluation ─────────────────────────────────────────────────────────────
@app.post("/api/test/random")
def test_random(req: TestRequest):
    env_cfg = req.env_config or _state["env_config"]
    errors  = validate_grid_config(env_cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    result = run_random_agent(env_cfg, max_steps=req.max_steps)
    _state["random_result"] = result
    return result


@app.post("/api/test/trained")
def test_trained(req: TestRequest):
    if _state["agent"] is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet. Run /api/train first.")

    env_cfg = req.env_config or _state["env_config"]
    errors  = validate_grid_config(env_cfg)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    result = run_trained_agent(_state["agent"], env_cfg, max_steps=req.max_steps)
    _state["trained_result"] = result
    return result


@app.get("/api/metrics/comparison")
def get_comparison():
    if _state["random_result"] is None or _state["trained_result"] is None:
        raise HTTPException(
            status_code=404,
            detail="Run both /api/test/random and /api/test/trained first."
        )
    return build_comparison(_state["random_result"], _state["trained_result"])


# ── Q-Table ────────────────────────────────────────────────────────────────
@app.post("/api/q-table/state")
def get_q_values_for_state(req: StateQueryRequest):
    if _state["agent"] is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet.")

    agent = _state["agent"]
    dirty_fs = frozenset(tuple(d) for d in req.dirty_cells)
    state    = (req.row, req.col, dirty_fs)

    return agent.get_state_info(state)


@app.get("/api/q-table/summary")
def get_q_table_summary():
    if _state["agent"] is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet.")

    agent = _state["agent"]
    return {
        "q_table_size":    len(agent.q_table),
        "total_updates":   agent.total_updates,
        "current_epsilon": round(agent.epsilon, 6),
        "actions":         ACTIONS,
        "hyperparams": {
            "learning_rate":   agent.alpha,
            "discount_factor": agent.gamma,
            "epsilon_min":     agent.epsilon_min,
            "epsilon_decay":   agent.epsilon_decay,
        },
    }


@app.get("/api/q-table/policy")
def get_policy():
    """
    Return the best action for each known state.
    For large Q-tables, returns a capped list of 500 entries.
    """
    if _state["agent"] is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet.")

    agent   = _state["agent"]
    env_cfg = _state["env_config"]
    policy  = []

    for state, values in list(agent.q_table.items())[:500]:
        row, col, dirty_fs = state
        best_action = agent.get_best_action(state)
        policy.append({
            "row":         row,
            "col":         col,
            "dirty_cells": sorted([list(d) for d in dirty_fs]),
            "best_action": ACTIONS[best_action],
            "best_action_id": best_action,
            "q_values":    {ACTIONS[i]: round(v, 4) for i, v in enumerate(values)},
        })

    return {
        "total_states":   len(agent.q_table),
        "showing_states": len(policy),
        "policy":         policy,
    }


# ── Save / Load ────────────────────────────────────────────────────────────
@app.post("/api/q-table/save")
def save_agent(req: SaveRequest):
    if _state["agent"] is None:
        raise HTTPException(status_code=400, detail="No trained agent to save.")

    filepath = req.filepath or QTABLE_FILE
    try:
        save_q_table(_state["agent"], filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

    return {"message": f"Q-table saved to {filepath}", "filepath": filepath}


@app.post("/api/q-table/load")
def load_agent(req: LoadRequest):
    filepath = req.filepath or QTABLE_FILE
    try:
        agent = load_q_table(QLearningAgent, filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

    _state["agent"] = agent
    return {
        "message":      f"Q-table loaded from {filepath}",
        "q_table_size": len(agent.q_table),
        "epsilon":      round(agent.epsilon, 6),
    }


# ── Reset ──────────────────────────────────────────────────────────────────
@app.post("/api/reset")
def reset_all():
    _state["agent"]            = None
    _state["training_results"] = None
    _state["random_result"]    = None
    _state["trained_result"]   = None
    _state["env_config"]       = DEFAULT_GRID
    return {"message": "All state cleared. Environment reset to default."}


@app.post("/api/reset/environment")
def reset_environment():
    _state["env_config"] = DEFAULT_GRID
    env = VacuumEnv(DEFAULT_GRID)
    return {"message": "Environment reset to default.", "env": env.to_dict()}


@app.post("/api/reset/training")
def reset_training():
    _state["agent"]            = None
    _state["training_results"] = None
    _state["random_result"]    = None
    _state["trained_result"]   = None
    return {"message": "Training state cleared. Agent reset."}


# ── Config ─────────────────────────────────────────────────────────────────
@app.get("/api/config/defaults")
def get_defaults():
    return {
        "hyperparams": DEFAULT_HYPERPARAMS,
        "rewards":     REWARDS,
        "grid":        DEFAULT_GRID,
        "actions":     ACTIONS,
    }


# ── Frontend static files ──────────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
