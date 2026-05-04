"""
Salp Chain MADDPG — MAXIMUM PARALLELISM EDITION
================================================
Changes from original:
  - New reward function: smooth distance shaping, stronger time penalty,
    larger collision penalty, time-dependent success bonus, explicit timeout penalty
  - Detailed logging: per-salp, per-env, per-timestep AND per-episode logs
    * timestep_log.csv  — every step: episode, env, step, salp, reward, collision,
                          wall_hit, dist_to_goal, obs_x, obs_y
    * episode_log.csv   — every episode: all original fields + per-salp breakdowns
"""

from __future__ import annotations

import math
import os
import pickle
import queue
import random
import sys
import threading
import time
from collections import deque
from multiprocessing import Process, Queue as MPQueue
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
HEADLESS         = True
NUM_WORKERS      = max(1, os.cpu_count() or 1)
ENVS_PER_WORKER  = 2
TOTAL_ENVS       = NUM_WORKERS * ENVS_PER_WORKER

# Physics
N_SALPS     = 5
LINK_LEN    = 68.0
SPRING_K    = 1.0
DRAG        = 0.955
MAX_SPEED   = 10.0
THRUST_MAG  = 0.60
GOAL_RADIUS = 30

# Env geometry
W_DEFAULT, H_DEFAULT, MARGIN_DEFAULT = 1400, 800, 70
LAND_REBUILD_INTERVAL = 1
OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX = 0.5, 1.5

# MADDPG hypers
LR_ACTOR         = 1e-4
LR_CRITIC        = 3e-4
GAMMA            = 0.97
TAU              = 0.005
BATCH_SIZE       = 256
BUFFER_CAPACITY  = 200_000
WARMUP_STEPS     = 10_000

# OU noise
OU_MU, OU_THETA, OU_SIGMA = 0.0, 0.15, 0.20
OU_SIGMA_MIN, OU_SIGMA_DECAY = 0.02, 0.9999

# Network dims
LOCAL_OBS_DIM    = 10
GLOBAL_STATE_DIM = LOCAL_OBS_DIM * N_SALPS
ACTION_DIM       = 2
CRITIC_INPUT_DIM = GLOBAL_STATE_DIM + ACTION_DIM * N_SALPS

# Training
MAX_EPISODES         = 10_000
UPDATE_EVERY         = 4
UPDATES_PER_STEP     = 4
CHECKPOINT_INTERVAL  = 500

# Logging — flush timestep log every N rows to avoid RAM buildup
TIMESTEP_LOG_FLUSH_INTERVAL = 5_000

# Paths
SAVE_DIR          = "salp_saves_parallel"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH   = os.path.join(SAVE_DIR, "maddpg_checkpoint.pt")
EPISODE_LOG_PATH  = os.path.join(SAVE_DIR, "episode_log.csv")
TIMESTEP_LOG_PATH = os.path.join(SAVE_DIR, "timestep_log.csv")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COMPILE = hasattr(torch, "compile") and DEVICE.type == "cuda"
print(f"Device: {DEVICE}  |  torch.compile: {USE_COMPILE}"
      f"  |  Workers: {NUM_WORKERS}  |  Envs/worker: {ENVS_PER_WORKER}"
      f"  |  Total envs: {TOTAL_ENVS}")


# =============================================================================
# GEOMETRY
# =============================================================================

def point_in_poly(px: float, py: float, poly: np.ndarray) -> bool:
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cond   = (yi > py) != (yj > py)
    x_int  = (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
    return bool(np.sum(cond & (px < x_int)) % 2 == 1)


def closest_on_poly(px: float, py: float, poly: np.ndarray) -> Tuple[float, float]:
    a   = poly
    b   = np.roll(poly, -1, axis=0)
    ab  = b - a
    ab2 = np.einsum("vi,vi->v", ab, ab)
    ap  = np.array([px - a[:, 0], py - a[:, 1]], dtype=np.float32).T
    t   = np.einsum("vi,vi->v", ap, ab) / (ab2 + 1e-12)
    t   = np.clip(t, 0.0, 1.0)
    cp  = a + t[:, None] * ab
    dx, dy = cp[:, 0] - px, cp[:, 1] - py
    d2  = dx * dx + dy * dy
    best = int(np.argmin(d2))
    return float(cp[best, 0]), float(cp[best, 1])


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    return poly.mean(axis=0)


def generate_land_polygon(cx, cy, base_r, points=18, jitter=0.2):
    ao     = random.uniform(0, 2 * math.pi)
    angles = ao + np.arange(points) * (2 * math.pi / points)
    rs     = base_r * (1 + np.random.uniform(-jitter, jitter, points))
    xs     = cx + np.cos(angles) * rs
    ys     = cy + np.sin(angles) * rs
    return np.stack([xs, ys], axis=1).astype(np.float32)


def segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)


# =============================================================================
# NETWORKS
# =============================================================================

class Actor(nn.Module):
    def __init__(self, obs_dim=LOCAL_OBS_DIM, action_dim=ACTION_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, input_dim=CRITIC_INPUT_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, global_obs, all_actions):
        return self.net(torch.cat([global_obs, all_actions], dim=-1)).squeeze(-1)


# =============================================================================
# OU NOISE
# =============================================================================

class OUNoise:
    def __init__(self, size, mu=OU_MU, theta=OU_THETA, sigma=OU_SIGMA):
        self.mu    = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(*self.state.shape).astype(np.float32)
        self.state += dx
        return self.state.copy()


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self.size     = 0
        self.pos      = 0

        self.obs  = np.zeros((capacity, N_SALPS, LOCAL_OBS_DIM), dtype=np.float32)
        self.acts = np.zeros((capacity, N_SALPS, ACTION_DIM),    dtype=np.float32)
        self.rews = np.zeros((capacity, N_SALPS),                dtype=np.float32)
        self.nobs = np.zeros((capacity, N_SALPS, LOCAL_OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity,),                        dtype=np.float32)

        self._prefetch_result = None
        self._prefetch_thread = None

    def push(self, local_obs, actions, rewards, next_obs, done):
        p = self.pos
        self.obs[p]  = np.stack(local_obs)   if isinstance(local_obs, list)  else local_obs
        self.acts[p] = np.stack(actions)      if isinstance(actions,   list)  else actions
        self.rews[p] = np.array(rewards,      dtype=np.float32)
        self.nobs[p] = np.stack(next_obs)     if isinstance(next_obs,  list)  else next_obs
        self.done[p] = float(done)
        self.pos  = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_tensors(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        def t(arr):
            return torch.from_numpy(arr[idx]).to(DEVICE, non_blocking=True)
        return t(self.obs), t(self.acts), t(self.rews), t(self.nobs), t(self.done)

    def sample(self, batch_size=BATCH_SIZE):
        return self._sample_tensors(batch_size)

    def prefetch(self, batch_size=BATCH_SIZE):
        if self.size < batch_size:
            return
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        def _work():
            self._prefetch_result = self._sample_tensors(batch_size)
        self._prefetch_thread = threading.Thread(target=_work, daemon=True)
        self._prefetch_thread.start()

    def get_prefetched(self, batch_size=BATCH_SIZE):
        if self._prefetch_thread:
            self._prefetch_thread.join()
        if self._prefetch_result is not None:
            result = self._prefetch_result
            self._prefetch_result = None
            return result
        return self._sample_tensors(batch_size)

    def __len__(self):
        return self.size


# =============================================================================
# MADDPG AGENT
# =============================================================================

def soft_update(target, source, tau=TAU):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


class MADDPG:
    def __init__(self):
        self.actors         = [Actor().to(DEVICE)  for _ in range(N_SALPS)]
        self.critics        = [Critic().to(DEVICE) for _ in range(N_SALPS)]
        self.target_actors  = [Actor().to(DEVICE)  for _ in range(N_SALPS)]
        self.target_critics = [Critic().to(DEVICE) for _ in range(N_SALPS)]

        for i in range(N_SALPS):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        adam_kwargs = {"fused": True} if DEVICE.type == "cuda" else {}
        self.actor_opts  = [optim.Adam(a.parameters(), lr=LR_ACTOR,  **adam_kwargs) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=LR_CRITIC, **adam_kwargs) for c in self.critics]

        if USE_COMPILE:
            self.actors         = [torch.compile(a, mode="reduce-overhead") for a in self.actors]
            self.critics        = [torch.compile(c, mode="reduce-overhead") for c in self.critics]
            self.target_actors  = [torch.compile(a, mode="reduce-overhead") for a in self.target_actors]
            self.target_critics = [torch.compile(c, mode="reduce-overhead") for c in self.target_critics]

        # Per-env per-agent noise instances to avoid shared noise across envs
        # Shape: noises[env_idx][agent_idx]
        self.noises      = [[OUNoise(ACTION_DIM) for _ in range(N_SALPS)]
                            for _ in range(TOTAL_ENVS)]
        self.noise_sigma = OU_SIGMA

    def act_batch(self, all_env_obs: List[List[np.ndarray]], explore: bool = True) -> List[List[np.ndarray]]:
        """
        all_env_obs: (n_envs, N_SALPS, LOCAL_OBS_DIM)
        Returns:     (n_envs, N_SALPS, ACTION_DIM)
        Fixed: each env gets independent noise sample (was shared before)
        """
        n_envs = len(all_env_obs)
        # agent_actions[agent_i] = (n_envs, ACTION_DIM)
        agent_actions = []
        for i in range(N_SALPS):
            obs_np = np.stack([all_env_obs[e][i] for e in range(n_envs)])
            obs_t  = torch.from_numpy(obs_np).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                a = self.actors[i](obs_t).cpu().numpy()  # (n_envs, ACTION_DIM)
            if explore:
                # Each env gets its own independent noise sample
                noise = np.stack([self.noises[e][i].sample() for e in range(n_envs)])
                a += noise
            agent_actions.append(np.clip(a, -1.0, 1.0))

        return [
            [agent_actions[agent_i][env_i] for agent_i in range(N_SALPS)]
            for env_i in range(n_envs)
        ]

    def update(self, buffer: ReplayBuffer):
        if len(buffer) < BATCH_SIZE:
            return

        obs_b, act_b, rew_b, nobs_b, done_b = buffer.get_prefetched()
        B = obs_b.shape[0]

        global_obs  = obs_b.reshape(B, -1)
        global_nobs = nobs_b.reshape(B, -1)
        all_actions = act_b.reshape(B, -1)

        with torch.no_grad():
            tgt_acts = torch.stack([self.target_actors[i](nobs_b[:, i]) for i in range(N_SALPS)], dim=1)
            tgt_acts_flat = tgt_acts.reshape(B, -1)

        for i in range(N_SALPS):
            # Critic update
            with torch.no_grad():
                tgt_q = rew_b[:, i] + GAMMA * (1.0 - done_b) * \
                        self.target_critics[i](global_nobs, tgt_acts_flat)

            cur_q       = self.critics[i](global_obs, all_actions)
            critic_loss = F.mse_loss(cur_q, tgt_q)

            self.critic_opts[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_opts[i].step()

            # Actor update — use current policy for all agents (fixed)
            acts_for_actor = act_b.clone()
            for j in range(N_SALPS):
                if j == i:
                    acts_for_actor[:, j] = self.actors[i](obs_b[:, i])  # differentiable
                else:
                    with torch.no_grad():
                        acts_for_actor[:, j] = self.actors[j](obs_b[:, j])  # current, no grad
            actor_loss = -self.critics[i](global_obs, acts_for_actor.reshape(B, -1)).mean()

            self.actor_opts[i].zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_opts[i].step()

        for i in range(N_SALPS):
            soft_update(self.target_actors[i],  self.actors[i])
            soft_update(self.target_critics[i], self.critics[i])

        buffer.prefetch()

    def decay_noise(self):
        self.noise_sigma = max(OU_SIGMA_MIN, self.noise_sigma * OU_SIGMA_DECAY)
        for env_noises in self.noises:
            for n in env_noises:
                n.sigma = self.noise_sigma

    def reset_noise(self, env_idx: int = None):
        """Reset noise for a specific env (on episode end) or all envs."""
        if env_idx is not None:
            for n in self.noises[env_idx]:
                n.reset()
        else:
            for env_noises in self.noises:
                for n in env_noises:
                    n.reset()

    def save(self, path=CHECKPOINT_PATH):
        def state(m):
            return (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
        torch.save({
            "actors":        [state(a) for a in self.actors],
            "critics":       [state(c) for c in self.critics],
            "target_actors": [state(a) for a in self.target_actors],
            "target_critics":[state(c) for c in self.target_critics],
            "actor_opts":    [o.state_dict() for o in self.actor_opts],
            "critic_opts":   [o.state_dict() for o in self.critic_opts],
            "noise_sigma":   self.noise_sigma,
        }, path)
        print(f"  Saved -> {path}")

    def load(self, path=CHECKPOINT_PATH):
        if not os.path.exists(path):
            print(f"  No checkpoint at {path}")
            return False
        data = torch.load(path, map_location=DEVICE)
        def unwrap(m):
            return m._orig_mod if hasattr(m, "_orig_mod") else m
        for i in range(N_SALPS):
            unwrap(self.actors[i]).load_state_dict(data["actors"][i])
            unwrap(self.critics[i]).load_state_dict(data["critics"][i])
            unwrap(self.target_actors[i]).load_state_dict(data["target_actors"][i])
            unwrap(self.target_critics[i]).load_state_dict(data["target_critics"][i])
            self.actor_opts[i].load_state_dict(data["actor_opts"][i])
            self.critic_opts[i].load_state_dict(data["critic_opts"][i])
        self.noise_sigma = data.get("noise_sigma", OU_SIGMA)
        for env_noises in self.noises:
            for n in env_noises:
                n.sigma = self.noise_sigma
        print(f"  Loaded <- {path}")
        return True


# =============================================================================
# ENVIRONMENT
# =============================================================================

class FastEnv:
    RADII   = np.array([14., 16., 18., 16., 14.], dtype=np.float32)
    SEMI_A  = RADII * 1.3
    SEMI_B  = RADII * 0.8
    MAX_EXTENT = np.maximum(SEMI_A, SEMI_B)

    def __init__(self, W=W_DEFAULT, H=H_DEFAULT, margin=MARGIN_DEFAULT,
                 worker_id=0, env_idx=0):
        self.W, self.H, self.margin = W, H, margin
        self.worker_id = worker_id
        self.env_idx   = env_idx

        self.pos = np.zeros((N_SALPS, 2), dtype=np.float32)
        self.vel = np.zeros((N_SALPS, 2), dtype=np.float32)

        self.land_polys: List[np.ndarray] = []
        self._reset_counter = 0

        self.goal_pos           = np.zeros(2, dtype=np.float32)
        self.episode_steps      = 0
        self.max_steps          = 700
        self.total_reward       = 0.0
        self._prev_dists        = None
        self.collision_flags    = np.zeros(N_SALPS, dtype=bool)
        self.wall_flags         = np.zeros(N_SALPS, dtype=bool)
        self.last_actions       = np.zeros((N_SALPS, 2), dtype=np.float32)
        self.episodes_completed = 0

        # Per-salp episode accumulators (reset on episode reset)
        self.salp_collision_counts = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_wall_counts      = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_reward_totals    = np.zeros(N_SALPS, dtype=np.float64)

        # Timestep log buffer — flushed back to main process via result queue
        # Each entry: one row per salp per step
        self.step_log_buffer: List[dict] = []

    # ── Obstacles ─────────────────────────────────────────────────────────────

    def _build_land(self):
        m     = self.margin + 120
        specs = [(55,18),(70,20),(78,22),(48,16),(42,14),(52,16)]
        placed, polys = [], []
        scale  = random.uniform(OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX)
        max_r  = min(self.W, self.H) // 6

        for r_base, pts in specs:
            r = min(int(r_base * scale), max_r)
            for _ in range(100):
                cx = random.uniform(m, self.W - m)
                cy = random.uniform(m, self.H - m)
                if not all(math.hypot(cx-ox, cy-oy) >= r+orad+60 for ox,oy,orad in placed):
                    continue
                raw     = generate_land_polygon(cx, cy, r, points=pts)
                clamped = raw.copy()
                clamped[:, 0] = np.clip(raw[:, 0], m, self.W - m)
                clamped[:, 1] = np.clip(raw[:, 1], m, self.H - m)
                test_pts = np.array([
                    [self.W*0.25, cy],[self.W*0.5, cy],[self.W*0.75, cy],
                    [cx, self.H*0.25],[cx, self.H*0.5],[cx, self.H*0.75],
                ], dtype=np.float32)
                if any(point_in_poly(p[0], p[1], clamped) for p in test_pts):
                    continue
                placed.append((cx, cy, r))
                polys.append(clamped)
                break
        self.land_polys = polys

    def _point_in_land(self, px, py, buffer_px=0.0):
        for poly in self.land_polys:
            if point_in_poly(px, py, poly):
                return True
            if buffer_px > 0:
                cx, cy = closest_on_poly(px, py, poly)
                if math.hypot(cx - px, cy - py) <= buffer_px:
                    return True
        return False

    def _rand_goal(self):
        inn = self.margin + 90
        for _ in range(1000):
            gx = random.uniform(inn, self.W - inn)
            gy = random.uniform(inn, self.H - inn)
            if not self._point_in_land(gx, gy, buffer_px=GOAL_RADIUS + 20):
                return np.array([gx, gy], dtype=np.float32)
        return np.array([self.W * 0.85, self.H * 0.15], dtype=np.float32)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _apply_springs(self, iters=3):
        for _ in range(iters):
            delta = self.pos[1:] - self.pos[:-1]
            dist  = np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-6)
            unit  = delta / dist
            corr  = unit * ((dist - LINK_LEN) * SPRING_K * 0.5)
            self.pos[:-1] += corr
            self.pos[1:]  -= corr
            rel_v = self.vel[1:] - self.vel[:-1]
            damp  = (rel_v * unit).sum(axis=1, keepdims=True) * unit * 0.06
            self.vel[:-1] += damp
            self.vel[1:]  -= damp

    def _enforce_separation(self):
        restitution = 0.6
        for i in range(N_SALPS):
            for j in range(i + 1, N_SALPS):
                d    = self.pos[j] - self.pos[i]
                dist = math.hypot(d[0], d[1])
                if dist < 1e-6:
                    continue
                min_d = self.RADII[i] + self.RADII[j]
                if dist < min_d:
                    unit = d / dist
                    corr = unit * ((min_d - dist) * 0.7)
                    self.pos[i] -= corr;  self.pos[j] += corr
                    vn = float(np.dot(self.vel[j] - self.vel[i], unit))
                    if vn < 0:
                        imp = (-(1 + restitution) * vn / 2.0) * unit
                        self.vel[i] -= imp;  self.vel[j] += imp

    def _wall_bounce(self):
        self.wall_flags[:] = False
        lo   = self.margin + self.MAX_EXTENT
        hi_x = self.W - self.margin - self.MAX_EXTENT
        hi_y = self.H - self.margin - self.MAX_EXTENT
        for i in range(N_SALPS):
            hit = False
            if self.pos[i, 0] < lo[i]:
                self.pos[i, 0] = lo[i];   self.vel[i, 0] =  abs(self.vel[i, 0]) * 0.35; hit = True
            if self.pos[i, 0] > hi_x[i]:
                self.pos[i, 0] = hi_x[i]; self.vel[i, 0] = -abs(self.vel[i, 0]) * 0.35; hit = True
            if self.pos[i, 1] < lo[i]:
                self.pos[i, 1] = lo[i];   self.vel[i, 1] =  abs(self.vel[i, 1]) * 0.35; hit = True
            if self.pos[i, 1] > hi_y[i]:
                self.pos[i, 1] = hi_y[i]; self.vel[i, 1] = -abs(self.vel[i, 1]) * 0.35; hit = True
            self.wall_flags[i] = hit

    def _resolve_land_collisions(self):
        self.collision_flags[:] = False
        for i in range(N_SALPS):
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            for poly in self.land_polys:
                cx, cy = closest_on_poly(px, py, poly)
                dx, dy = px - cx, py - cy
                dist   = math.hypot(dx, dy)
                inside = point_in_poly(px, py, poly)
                min_sep = float(self.MAX_EXTENT[i]) + 2.0
                if inside or dist < min_sep:
                    if dist < 1e-8:
                        cent   = polygon_centroid(poly)
                        dx, dy = px - cent[0], py - cent[1]
                        dist   = math.hypot(dx, dy)
                        if dist < 1e-8:
                            dx, dist = 1.0, 1.0
                    nx, ny = dx / dist, dy / dist
                    self.pos[i] = np.array([cx + nx * min_sep, cy + ny * min_sep], dtype=np.float32)
                    vn = self.vel[i, 0] * nx + self.vel[i, 1] * ny
                    if vn < 0:
                        self.vel[i, 0] -= 1.55 * vn * nx
                        self.vel[i, 1] -= 1.55 * vn * ny
                    self.vel[i] *= 0.90
                    self.collision_flags[i] = True
                    px, py = float(self.pos[i, 0]), float(self.pos[i, 1])

    def _resolve_link_collisions(self):
        for i in range(N_SALPS - 1):
            for poly in self.land_polys:
                pts = poly
                n   = len(pts)
                for j in range(n):
                    p1, p2 = pts[j], pts[(j + 1) % n]
                    if segments_intersect(self.pos[i], self.pos[i+1], p1, p2):
                        edge   = p2 - p1
                        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
                        normal /= (np.linalg.norm(normal) + 1e-8)
                        mid = (self.pos[i] + self.pos[i+1]) * 0.5
                        if np.dot(mid - p1, normal) < 0:
                            normal = -normal
                        for k in (i, i+1):
                            self.pos[k] += normal * 6.0
                            vn = np.dot(self.vel[k], normal)
                            if vn < 0:
                                self.vel[k] -= vn * normal

    def _enforce_rigid_links(self):
        for i in range(N_SALPS - 1):
            d    = self.pos[i+1] - self.pos[i]
            dist = np.linalg.norm(d) + 1e-8
            corr = d * 0.5 * (dist - LINK_LEN) / dist
            self.pos[i]   += corr
            self.pos[i+1] -= corr

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions, spring_iters=3):
        self.last_actions = np.array(actions, dtype=np.float32)
        tv = self.last_actions * THRUST_MAG
        self.vel += tv
        self.vel *= DRAG
        spd  = np.linalg.norm(self.vel, axis=1, keepdims=True)
        mask = spd > MAX_SPEED
        self.vel = np.where(mask, self.vel * MAX_SPEED / spd, self.vel)
        self.pos += self.vel

        self._apply_springs(iters=spring_iters)
        for _ in range(3):
            self._enforce_separation()
            self._resolve_link_collisions()
            self._resolve_land_collisions()
            self._enforce_rigid_links()
        self._wall_bounce()
        self.episode_steps += 1

    # ── Observation ───────────────────────────────────────────────────────────

    def get_obs(self):
        gx, gy   = self.goal_pos
        gx_n, gy_n = gx / self.W, gy / self.H
        obs_list = []
        for i in range(N_SALPS):
            x, y = float(self.pos[i, 0]), float(self.pos[i, 1])
            dx, dy = gx - x, gy - y
            dist_to_goal = math.hypot(dx, dy) / self.W
            ang          = math.atan2(dy, dx)

            if self.land_polys:
                min_d, best_ang = self.W, 0.0
                for poly in self.land_polys:
                    cx, cy = closest_on_poly(x, y, poly)
                    d = math.hypot(cx - x, cy - y)
                    if d < min_d:
                        min_d, best_ang = d, math.atan2(cy - y, cx - x)
                d1, a1 = min_d, best_ang
            else:
                d1, a1 = self.W, 0.0

            obs_list.append(np.array([
                x / self.W, y / self.H,
                gx_n, gy_n,
                dist_to_goal,
                math.sin(ang), math.cos(ang),
                d1 / self.W, math.sin(a1), math.cos(a1),
            ], dtype=np.float32))
        return obs_list

    # ── Reward (updated) ──────────────────────────────────────────────────────

    def reward(self):
        """
        Reward function — no clipping anywhere:
        - Smooth normalised distance shaping: delta/diag * 10  (replaces binary +3/-0.5)
        - Time penalty: -0.5/step  (was -0.05)
        - Obstacle collision penalty: -15  (was -5)
        - Wall hit penalty: -3  (was -2)
        - Success: 50 base + 200*(1 - steps/max_steps) efficiency bonus (unclipped)
        - Timeout: -50 flat (unclipped)
        - No reward clipping at any point
        Returns: rewards, team_reward, min_dist, done, success, timeout, per_salp_log
        """
        diff        = self.pos - self.goal_pos
        local_dists = np.linalg.norm(diff, axis=1)
        min_d       = float(local_dists.min())

        if self._prev_dists is None:
            self._prev_dists = local_dists.copy()

        rewards      = []
        per_salp_log = []   # one dict per salp for timestep logging

        for i in range(N_SALPS):
            prev_d = self._prev_dists[i]
            curr_d = local_dists[i]

            # Smooth distance shaping — normalised by env diagonal
            delta = (prev_d - curr_d) / (self.W + self.H)
            r     = delta * 10.0

            # Stronger time penalty
            r -= 0.5

            collision_hit = bool(self.collision_flags[i])
            wall_hit      = bool(self.wall_flags[i])

            if collision_hit:
                r -= 15.0
                self.salp_collision_counts[i] += 1
            if wall_hit:
                r -= 3.0
                self.salp_wall_counts[i] += 1

            rewards.append(r)
            per_salp_log.append({
                "salp":          i,
                "reward_step":   r,
                "collision":     int(collision_hit),
                "wall_hit":      int(wall_hit),
                "dist_to_goal":  float(curr_d),
                "prev_dist":     float(prev_d),
                "dist_delta":    float(prev_d - curr_d),
                "pos_x":         float(self.pos[i, 0]),
                "pos_y":         float(self.pos[i, 1]),
                "vel_x":         float(self.vel[i, 0]),
                "vel_y":         float(self.vel[i, 1]),
                "speed":         float(math.hypot(self.vel[i, 0], self.vel[i, 1])),
                "action_x":      float(self.last_actions[i, 0]),
                "action_y":      float(self.last_actions[i, 1]),
                "goal_x":        float(self.goal_pos[0]),
                "goal_y":        float(self.goal_pos[1]),
            })

        self._prev_dists = local_dists.copy()

        done = success = False
        timeout = False

        if min_d < GOAL_RADIUS:
            efficiency_bonus = 200.0 * (1.0 - self.episode_steps / self.max_steps)
            terminal_r = 50.0 + efficiency_bonus
            rewards = [terminal_r] 
            for entry in per_salp_log:
                entry["reward_step"] = terminal_r
            done = success = True

        elif self.episode_steps >= self.max_steps:
            rewards = [-50.0] 
            for entry in per_salp_log:
                entry["reward_step"] = -50.0
            done    = True
            timeout = True

        # Compute inter-salp link distances for logging
        link_dists = [float(np.linalg.norm(self.pos[i+1] - self.pos[i]))
                      for i in range(N_SALPS - 1)]

        # No clipping — full reward signal reaches the critic
        rewards = [float(r) for r in rewards]
        team_reward = float(np.mean(rewards))

        # Update per-salp episode totals and finalise log entries
        for i, r in enumerate(rewards):
            self.salp_reward_totals[i] += r
            entry = per_salp_log[i]
            entry["reward_step"]   = r          # overwrite with terminal value if done
            entry["team_reward"]   = team_reward
            entry["success"]       = int(success)
            entry["timeout"]       = int(timeout)
            entry["min_dist_team"] = min_d      # closest any salp is to goal this step
            entry["episode_steps"] = self.episode_steps
            # Link distances: this salp's link to next (N/A for last salp)
            entry["link_dist_next"] = link_dists[i] if i < N_SALPS - 1 else float("nan")

        return rewards, team_reward, min_d, done, success, timeout, per_salp_log

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_counter += 1
        if self._reset_counter % LAND_REBUILD_INTERVAL == 0 or not self.land_polys:
            self._build_land()

        pad   = self.margin + 100
        start = np.array([pad, self.H - pad], dtype=np.float32)

        for _ in range(200):
            heading = random.uniform(0, 2 * math.pi)
            ax, ay  = math.cos(heading), math.sin(heading)
            positions = []
            valid = True
            for i in range(N_SALPS):
                pos = start + np.array([ax * i * LINK_LEN, ay * i * LINK_LEN], dtype=np.float32)
                if not (self.margin <= pos[0] <= self.W - self.margin and
                        self.margin <= pos[1] <= self.H - self.margin):
                    valid = False; break
                if self._point_in_land(float(pos[0]), float(pos[1]), buffer_px=28):
                    valid = False; break
                positions.append(pos)
            if valid:
                break
        else:
            s2 = 1.0 / math.sqrt(2)
            positions = [start + np.array([s2*i*LINK_LEN, -s2*i*LINK_LEN]) for i in range(N_SALPS)]

        for i, pos in enumerate(positions):
            self.pos[i] = pos
        self.vel[:]              = 0.0
        self.goal_pos            = self._rand_goal()
        self.episode_steps       = 0
        self.total_reward        = 0.0
        self._prev_dists         = None
        self.collision_flags[:]  = False
        self.wall_flags[:]       = False
        self.last_actions[:]     = 0.0
        self.step_log_buffer     = []

        # Reset per-salp accumulators
        self.salp_collision_counts[:] = 0
        self.salp_wall_counts[:]      = 0
        self.salp_reward_totals[:]    = 0.0


# =============================================================================
# WORKER PROCESS
# =============================================================================

def worker_process(worker_id: int, n_envs: int,
                   action_q: MPQueue, result_q: MPQueue):
    envs = [FastEnv(worker_id=worker_id, env_idx=e) for e in range(n_envs)]
    for e in envs:
        e.reset()

    # Send initial observations
    for env_idx, env in enumerate(envs):
        result_q.put({
            "type":          "step",
            "worker_id":     worker_id,
            "env_idx":       env_idx,
            "obs":           env.get_obs(),
            "reward":        [0.0] * N_SALPS,
            "team_reward":   0.0,
            "dist":          float("inf"),
            "done":          False,
            "success":       False,
            "timeout":       False,
            "total_reward":  0.0,
            "episode_steps": 0,
            "collision_sum": 0,
            "per_salp_log":  [],
            # Episode-level per-salp summary (only populated on done)
            "salp_collision_counts": [0]*N_SALPS,
            "salp_wall_counts":      [0]*N_SALPS,
            "salp_reward_totals":    [0.0]*N_SALPS,
        })

    global_step = 0

    while True:
        msg = action_q.get()
        if msg is None:
            break

        actions_batch = msg

        for env_idx, env in enumerate(envs):
            env.step(actions_batch[env_idx], spring_iters=3)
            rewards, team_r, dist, done, success, timeout, per_salp_log = env.reward()
            env.total_reward += team_r
            global_step      += 1

            # Attach global step and episode context to each salp log entry
            for entry in per_salp_log:
                entry["worker_id"]     = worker_id
                entry["env_idx"]       = env_idx
                entry["episode_step"]  = env.episode_steps
                entry["global_step"]   = global_step

            result = {
                "type":          "step",
                "worker_id":     worker_id,
                "env_idx":       env_idx,
                "obs":           env.get_obs(),
                "reward":        rewards,
                "team_reward":   team_r,
                "dist":          dist,
                "done":          done,
                "success":       success,
                "timeout":       timeout,
                "total_reward":  env.total_reward,
                "episode_steps": env.episode_steps,
                "collision_sum": int(env.collision_flags.sum()),
                "per_salp_log":  per_salp_log,
                "salp_collision_counts": env.salp_collision_counts.tolist(),
                "salp_wall_counts":      env.salp_wall_counts.tolist(),
                "salp_reward_totals":    env.salp_reward_totals.tolist(),
            }
            result_q.put(result)

            if done:
                env.episodes_completed += 1
                env.reset()


# =============================================================================
# LOGGING HELPERS
# =============================================================================

class EpisodeLogger:
    """Writes episode_log.csv with full per-salp episode summaries."""

    EPISODE_COLS = [
        "episode", "worker_id", "env_idx",
        "reward", "success", "timeout", "collisions", "steps",
        # per-salp totals
        "salp0_reward", "salp1_reward", "salp2_reward", "salp3_reward", "salp4_reward",
        "salp0_collisions", "salp1_collisions", "salp2_collisions",
        "salp3_collisions", "salp4_collisions",
        "salp0_wall_hits", "salp1_wall_hits", "salp2_wall_hits",
        "salp3_wall_hits", "salp4_wall_hits",
    ]

    def __init__(self, path=EPISODE_LOG_PATH):
        self.path = path
        self._buffer: List[dict] = []
        # Write header if new file
        if not os.path.exists(path):
            pd.DataFrame(columns=self.EPISODE_COLS).to_csv(path, index=False)

    def log(self, episode_idx: int, r: dict):
        row = {
            "episode":    episode_idx,
            "worker_id":  r["worker_id"],
            "env_idx":    r["env_idx"],
            "reward":     r["total_reward"],
            "success":    1 if r["success"] else 0,
            "timeout":    1 if r["timeout"] else 0,
            "collisions": r["collision_sum"],
            "steps":      r["episode_steps"],
        }
        for i in range(N_SALPS):
            row[f"salp{i}_reward"]     = r["salp_reward_totals"][i]
            row[f"salp{i}_collisions"] = r["salp_collision_counts"][i]
            row[f"salp{i}_wall_hits"]  = r["salp_wall_counts"][i]
        self._buffer.append(row)
        if len(self._buffer) >= 50:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        pd.DataFrame(self._buffer).to_csv(
            self.path, mode="a", header=False, index=False)
        self._buffer = []


class TimestepLogger:
    """
    Writes timestep_log.csv — one row per salp per environment step.
    Every measurable quantity logged: position, velocity, speed, action,
    distance to goal, distance delta, link distances, collision/wall flags,
    step reward, team reward, outcome flags, episode step count.
    """

    TS_COLS = [
        # identifiers
        "episode", "worker_id", "env_idx", "episode_step", "global_step", "salp",
        # reward signal
        "reward_step", "team_reward",
        # events
        "collision", "wall_hit",
        # goal
        "dist_to_goal", "prev_dist", "dist_delta", "min_dist_team",
        "goal_x", "goal_y",
        # position & motion
        "pos_x", "pos_y", "vel_x", "vel_y", "speed",
        # action taken
        "action_x", "action_y",
        # chain geometry
        "link_dist_next",
        # episode outcome (non-zero only on terminal step)
        "success", "timeout", "episode_steps",
    ]

    def __init__(self, path=TIMESTEP_LOG_PATH, flush_every=TIMESTEP_LOG_FLUSH_INTERVAL):
        self.path        = path
        self.flush_every = flush_every
        self._buffer: List[dict] = []
        self._count  = 0
        if not os.path.exists(path):
            pd.DataFrame(columns=self.TS_COLS).to_csv(path, index=False)

    def log_step(self, episode_idx: int, per_salp_log: List[dict]):
        for entry in per_salp_log:
            row = {"episode": episode_idx}
            row.update(entry)
            self._buffer.append(row)
            self._count += 1
        if self._count >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        pd.DataFrame(self._buffer).to_csv(
            self.path, mode="a", header=False, index=False)
        self._buffer = []
        self._count  = 0


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(agent, episodes_done):
    agent.save(CHECKPOINT_PATH)
    with open(os.path.join(SAVE_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"episodes_done": episodes_done}, f)


def load_checkpoint(agent):
    meta_path = os.path.join(SAVE_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        return 0
    if not agent.load(CHECKPOINT_PATH):
        return 0
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Resuming from episode {data['episodes_done']}")
    return data["episodes_done"]


# =============================================================================
# ANALYSIS / PLOTTING
# =============================================================================

def analyze_and_plot():
    if not os.path.exists(EPISODE_LOG_PATH):
        return
    df = pd.read_csv(EPISODE_LOG_PATH)
    print(f"\n===== SUMMARY =====")
    print(f"Episodes:       {len(df)}")
    print(f"Successes:      {df['success'].sum()} ({100*df['success'].mean():.1f}%)")
    print(f"Avg collisions: {df['collisions'].mean():.2f}")
    ss = df[df["success"]==1]["steps"]
    if len(ss):
        print(f"Avg steps/goal: {ss.mean():.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Salp MADDPG — {TOTAL_ENVS} envs ({NUM_WORKERS} workers)")
    w = 100

    df["reward"].rolling(w).mean().plot(ax=axes[0,0],   title="Team reward (rolling)")
    df["success"].rolling(w).mean().plot(ax=axes[0,1],  title="Success rate (rolling)")
    df["collisions"].rolling(w).mean().plot(ax=axes[0,2], title="Collisions (rolling)")
    df["steps"].rolling(w).mean().plot(ax=axes[1,0],    title="Steps (rolling)")

    # Per-salp reward
    for i in range(N_SALPS):
        df[f"salp{i}_reward"].rolling(w).mean().plot(ax=axes[1,1], label=f"salp {i}")
    axes[1,1].set_title("Per-salp reward (rolling)")
    axes[1,1].legend(fontsize=8)

    # Per-salp collisions
    for i in range(N_SALPS):
        df[f"salp{i}_collisions"].rolling(w).mean().plot(ax=axes[1,2], label=f"salp {i}")
    axes[1,2].set_title("Per-salp collisions (rolling)")
    axes[1,2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(out, dpi=100)
    print(f"Plot -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESUME = True

    agent    = MADDPG()
    buffer   = ReplayBuffer(capacity=BUFFER_CAPACITY)
    start_ep = 0

    if RESUME:
        start_ep = load_checkpoint(agent)

    ep_logger = EpisodeLogger(EPISODE_LOG_PATH)
    ts_logger = TimestepLogger(TIMESTEP_LOG_PATH)

    # ── Spawn workers ─────────────────────────────────────────────────────────
    action_qs = [MPQueue(maxsize=4) for _ in range(NUM_WORKERS)]
    result_q  = MPQueue(maxsize=TOTAL_ENVS * 4)

    workers = []
    for wid in range(NUM_WORKERS):
        p = Process(
            target=worker_process,
            args=(wid, ENVS_PER_WORKER, action_qs[wid], result_q),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(f"{NUM_WORKERS} worker processes started ({TOTAL_ENVS} total envs)")

    obs_store = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
    for _ in range(TOTAL_ENVS):
        r = result_q.get()
        obs_store[r["worker_id"]][r["env_idx"]] = r["obs"]

    def flat_obs():
        return [obs_store[w][e]
                for w in range(NUM_WORKERS)
                for e in range(ENVS_PER_WORKER)]

    episodes_done = start_ep
    success_count = 0
    total_steps   = 0

    # Track current episode index per (worker, env) slot
    ep_idx_store = [[episodes_done + w * ENVS_PER_WORKER + e
                     for e in range(ENVS_PER_WORKER)]
                    for w in range(NUM_WORKERS)]

    pbar = tqdm(
        total=MAX_EPISODES,
        initial=start_ep,
        desc=f"Training ({TOTAL_ENVS} envs, {DEVICE})",
        dynamic_ncols=True,
    )

    last_obs  = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
    last_acts = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]

    for w in range(NUM_WORKERS):
        for e in range(ENVS_PER_WORKER):
            last_obs[w][e] = obs_store[w][e]

    try:
        while episodes_done < MAX_EPISODES:

            # ── Choose actions ────────────────────────────────────────────────
            all_obs     = flat_obs()
            use_random  = total_steps < WARMUP_STEPS
            if use_random:
                all_acts = [
                    [np.random.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
                     for _ in range(N_SALPS)]
                    for _ in range(TOTAL_ENVS)
                ]
            else:
                all_acts = agent.act_batch(all_obs, explore=True)

            for w in range(NUM_WORKERS):
                worker_acts = []
                for e in range(ENVS_PER_WORKER):
                    flat_idx = w * ENVS_PER_WORKER + e
                    last_acts[w][e] = all_acts[flat_idx]
                    worker_acts.append(all_acts[flat_idx])
                action_qs[w].put(worker_acts)

            # ── Collect results ───────────────────────────────────────────────
            for _ in range(TOTAL_ENVS):
                r    = result_q.get()
                w, e = r["worker_id"], r["env_idx"]

                prev_obs = last_obs[w][e]
                actions  = last_acts[w][e]
                next_obs = r["obs"]

                buffer.push(prev_obs, actions, r["reward"], next_obs, r["done"])

                obs_store[w][e] = next_obs
                last_obs[w][e]  = next_obs
                total_steps    += 1

                # Timestep logging — attach current episode index
                current_ep = ep_idx_store[w][e]
                ts_logger.log_step(current_ep, r["per_salp_log"])

                if r["done"]:
                    if r["success"]:
                        success_count += 1

                    ep_logger.log(episodes_done, r)

                    pbar.update(1)
                    pbar.set_postfix(
                        ep=episodes_done,
                        goals=success_count,
                        steps=r["episode_steps"],
                        sigma=f"{agent.noise_sigma:.3f}",
                        buf=len(buffer),
                    )

                    agent.decay_noise()
                    # Reset noise only for this env slot
                    flat_env_idx = w * ENVS_PER_WORKER + e
                    agent.reset_noise(env_idx=flat_env_idx)

                    ep_idx_store[w][e] = episodes_done
                    episodes_done += 1

                    if episodes_done % CHECKPOINT_INTERVAL == 0:
                        ep_logger.flush()
                        ts_logger.flush()
                        save_checkpoint(agent, episodes_done)
                        tqdm.write(f"  Checkpoint @ ep {episodes_done}")

                    if episodes_done >= MAX_EPISODES:
                        break

            # ── Gradient updates ──────────────────────────────────────────────
            if total_steps >= WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
                for _ in range(UPDATES_PER_STEP):
                    agent.update(buffer)
                buffer.prefetch()

    except KeyboardInterrupt:
        print("\nInterrupted — saving...")

    finally:
        for q in action_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in workers:
            p.join(timeout=3)

        pbar.close()
        ep_logger.flush()
        ts_logger.flush()
        save_checkpoint(agent, episodes_done)
        print("All saved.")
        analyze_and_plot()


if __name__ == "__main__":
    main()
