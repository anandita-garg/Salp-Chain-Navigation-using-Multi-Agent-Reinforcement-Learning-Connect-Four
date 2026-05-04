"""
demo.py  —  Salp Chain MARL · Trained Model Demo
=================================================
Imports FastEnv DIRECTLY from each algorithm's own training script,
loads the saved checkpoints from the repo, and runs 1 episode per
algorithm to prove they work.

Algorithms demonstrated (in order):
  1. IQL   Static   — Q-table, checkpoint_iql_static.pkl
  2. IQL   Parallel — Q-table, checkpoint_iql_parallel.pkl  (or iql_parallel.pkl)
  3. MADDPG Static  — Actor nets, maddpg_static_checkpoint.pt
  4. MADDPG Dynamic — Actor nets, maddpg_dynamic_checkpoint.pt
  5. MAPPO  Static  — Actor nets, mappo_checkpoint_mappo_static.pt

Run from the repository root:
    python demo.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import pickle
import sys
import types
from typing import List

import numpy as np
import torch
import torch.nn as nn

# ─── colours for terminal output ─────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YLW  = "\033[93m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"

REPO = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# SHARED PHYSICS CONSTANTS  (same across all training scripts)
# =============================================================================
N_SALPS    = 5
GOAL_RADIUS = 30
ACTION_DIM  = 2
LOCAL_OBS_DIM = 10

# IQL discrete actions (9-way: coast + 8 cardinal)
ACTION_DIRS = [
    (0.0, 0.0),
    (0.0,-1.0),(1.0,-1.0),(1.0,0.0),(1.0,1.0),
    (0.0, 1.0),(-1.0,1.0),(-1.0,0.0),(-1.0,-1.0),
]
N_ACTIONS  = len(ACTION_DIRS)
ANGLE_BINS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UTILITY: load a Python module from its file path
# =============================================================================
def _load_module(name: str, filepath: str):
    """Import a .py file as a module without executing __main__ blocks."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    # Must register before exec so that @dataclasses.dataclass and similar
    # decorators that call sys.modules[cls.__module__] can resolve correctly.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# UTILITY: run one episode with a generic action function
# =============================================================================
def run_episode(env, get_actions_fn):
    """
    Step `env` for one full episode using `get_actions_fn(obs_list) -> actions`.
    Returns (success, steps, total_team_reward, min_dist_ever).
    """
    env.reset()
    obs = env.get_obs()
    total_reward = 0.0
    min_dist     = float("inf")
    done         = False

    while not done:
        actions = get_actions_fn(obs)
        env.step(actions)
        obs = env.get_obs()
        rewards, team_r, dist, done, success, timeout, *_ = env.reward()
        total_reward += team_r
        if dist < min_dist:
            min_dist = dist

    return success, env.episode_steps, total_reward, min_dist


# =============================================================================
# UTILITY: print a result row
# =============================================================================
def print_result(algo: str, success: bool, steps: int,
                 reward: float, min_dist: float):
    tag  = f"{GRN}SUCCESS{RST}" if success else f"{RED}TIMEOUT{RST}"
    line = (f"  {BLD}{algo:<22}{RST}  {tag}  "
            f"steps={steps:4d}  reward={reward:8.1f}  "
            f"min_dist={min_dist:6.1f}")
    print(line)


# =============================================================================
# 1 & 2.  IQL  (static + parallel)  — Q-table agents
# =============================================================================

def iql_get_obs_key(obs: np.ndarray) -> tuple:
    """
    Discretise a single salp's LOCAL_OBS_DIM observation into a hashable key.
    Matches the discretisation used in both IQL training scripts.
    """
    # obs = [x/W, y/H, gx/W, gy/H, dist_goal, sin_ang, cos_ang,
    #         d1/W, sin_a1, cos_a1]
    x_bin   = int(obs[0] * 5)
    y_bin   = int(obs[1] * 5)
    ang     = math.atan2(float(obs[5]), float(obs[6]))
    ang_bin = int((ang + math.pi) / (2 * math.pi) * ANGLE_BINS) % ANGLE_BINS
    dist_b  = min(int(obs[4] * 5), 4)
    obs_b   = min(int(obs[7] * 5), 4)
    return (x_bin, y_bin, ang_bin, dist_b, obs_b)


def iql_get_actions(q_tables: list, obs_list: list) -> list:
    """Greedy (epsilon=0) Q-table policy for all 5 salps."""
    actions = []
    for i in range(N_SALPS):
        key = iql_get_obs_key(obs_list[i])
        q   = q_tables[i].get(key, [0.0] * N_ACTIONS)
        a   = int(np.argmax(q))
        actions.append(list(ACTION_DIRS[a]))
    return actions


def demo_iql(label: str, script_path: str, checkpoint_path: str):
    """Run one IQL episode using the real FastEnv from the training script."""
    if not os.path.exists(script_path):
        print(f"  {YLW}SKIP{RST}  {label} — script not found: {script_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"  {YLW}SKIP{RST}  {label} — checkpoint not found: {checkpoint_path}")
        return

    # Load the real training module and borrow its FastEnv
    mod = _load_module(f"iql_{label}", script_path)
    env = mod.FastEnv()
    env.reset()

    # Load Q-tables
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    # Checkpoints are saved in different shapes across the two scripts;
    # handle both: dict with "q_tables" key, or a plain list.
    if isinstance(ckpt, dict):
        q_tables = ckpt.get("q_tables", ckpt.get("qtables", ckpt))
    else:
        q_tables = ckpt          # plain list of dicts

    success, steps, reward, min_dist = run_episode(
        env,
        lambda obs: iql_get_actions(q_tables, obs)
    )
    print_result(label, success, steps, reward, min_dist)


# =============================================================================
# 3 & 4.  MADDPG  (static + dynamic)  — Actor networks
# =============================================================================

class MaddpgActor(nn.Module):
    """
    MADDPG actor: matches both static and dynamic training scripts.
    Architecture confirmed from salp_maddpg_static.py: 2-hidden-layer MLP,
    hidden=256, tanh activations, tanh output.
    """
    def __init__(self, obs_dim: int = LOCAL_OBS_DIM,
                 action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self.net(t).squeeze(0).cpu().numpy()


def load_maddpg_actors(checkpoint_path: str) -> List[MaddpgActor]:
    data   = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MaddpgActor().to(DEVICE) for _ in range(N_SALPS)]
    # Checkpoint key is "actors" — list of state_dicts
    actor_states = data["actors"]
    for i, actor in enumerate(actors):
        actor.load_state_dict(actor_states[i])
        actor.eval()
    return actors


def demo_maddpg(label: str, script_path: str, checkpoint_path: str):
    if not os.path.exists(script_path):
        print(f"  {YLW}SKIP{RST}  {label} — script not found: {script_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"  {YLW}SKIP{RST}  {label} — checkpoint not found: {checkpoint_path}")
        return

    mod = _load_module(f"maddpg_{label}", script_path)
    env = mod.FastEnv()
    env.reset()

    actors = load_maddpg_actors(checkpoint_path)

    def get_actions(obs_list):
        return [actors[i].act(obs_list[i]) for i in range(N_SALPS)]

    success, steps, reward, min_dist = run_episode(env, get_actions)
    print_result(label, success, steps, reward, min_dist)


# =============================================================================
# 5.  MAPPO  Static  — Actor networks (Gaussian policy, mean + log_std)
# =============================================================================

class MappoActor(nn.Module):
    """
    MAPPO actor: matches salp_mappo_static.py exactly.
    hidden=64, tanh trunk, mean_head + shared log_std parameter.
    """
    LOG_STD_MIN = -4.0
    LOG_STD_MAX =  1.0

    def __init__(self, obs_dim: int = LOCAL_OBS_DIM,
                 action_dim: int = ACTION_DIM, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std   = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        h    = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        return mean

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self(t).squeeze(0).cpu().numpy()


def load_mappo_actors(checkpoint_path: str) -> List[MappoActor]:
    data   = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MappoActor().to(DEVICE) for _ in range(N_SALPS)]
    saved  = data["actors"]
    share  = data.get("share_params", False)
    for i, actor in enumerate(actors):
        idx = 0 if share else min(i, len(saved) - 1)
        actor.load_state_dict(saved[idx])
        actor.eval()
    return actors


def demo_mappo(label: str, script_path: str, checkpoint_path: str):
    if not os.path.exists(script_path):
        print(f"  {YLW}SKIP{RST}  {label} — script not found: {script_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"  {YLW}SKIP{RST}  {label} — checkpoint not found: {checkpoint_path}")
        return

    mod = _load_module(f"mappo_{label}", script_path)
    env = mod.FastEnv()
    env.reset()

    actors = load_mappo_actors(checkpoint_path)

    def get_actions(obs_list):
        return [actors[i].act(obs_list[i]) for i in range(N_SALPS)]

    success, steps, reward, min_dist = run_episode(env, get_actions)
    print_result(label, success, steps, reward, min_dist)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{BLD}{CYN}{'═'*62}{RST}")
    print(f"{BLD}{CYN}   Salp Chain Navigation · MARL Demo{RST}")
    print(f"{BLD}{CYN}   1 episode per algorithm | trained checkpoints{RST}")
    print(f"{BLD}{CYN}{'═'*62}{RST}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Repo   : {REPO}\n")
    print(f"  {'Algorithm':<22}  {'Result':<12}  {'Details'}")
    print(f"  {'-'*58}")

    # ── IQL Static ───────────────────────────────────────────────────────────
    demo_iql(
        label          = "IQL  Static",
        script_path    = os.path.join(REPO, "salp_iql_static_final.py"),
        checkpoint_path= os.path.join(REPO, "checkpoint_iql_static.pkl"),
    )

    # ── IQL Parallel ─────────────────────────────────────────────────────────
    # Two checkpoint filenames exist in the repo; try both.
    iql_par_ckpt = os.path.join(REPO, "checkpoint_iql_parallel.pkl")
    if not os.path.exists(iql_par_ckpt):
        iql_par_ckpt = os.path.join(REPO, "iql_parallel.pkl")
    demo_iql(
        label          = "IQL  Parallel",
        script_path    = os.path.join(REPO, "salp_iql_parallel.py"),
        checkpoint_path= iql_par_ckpt,
    )

    # ── MADDPG Static ────────────────────────────────────────────────────────
    demo_maddpg(
        label          = "MADDPG Static",
        script_path    = os.path.join(REPO, "salp_maddpg_static.py"),
        checkpoint_path= os.path.join(REPO, "maddpg_static_checkpoint.pt"),
    )

    # ── MADDPG Dynamic ───────────────────────────────────────────────────────
    demo_maddpg(
        label          = "MADDPG Dynamic",
        script_path    = os.path.join(REPO, "salp_maddpg_parallel.py"),
        checkpoint_path= os.path.join(REPO, "maddpg_dynamic_checkpoint.pt"),
    )

    # ── MAPPO Static ─────────────────────────────────────────────────────────
    demo_mappo(
        label          = "MAPPO  Static",
        script_path    = os.path.join(REPO, "salp_mappo_static.py"),
        checkpoint_path= os.path.join(REPO, "mappo_checkpoint_mappo_static.pt"),
    )

    print(f"\n  {'-'*58}")
    print(f"  {GRN}Demo complete.{RST}\n")


if __name__ == "__main__":
    main()
