"""
Salp Chain MAPPO — STATIC EDITION
==================================
MAPPO (Multi-Agent PPO) with static obstacle layout (obstacles built once
per environment instance, never rebuilt between episodes).

Changes from salp_mappo_parallel.py
--------------------------------------
* Static obstacles: Each FastEnv builds its land polygons exactly once
  (like salp_iql_static). The LAND_REBUILD_INTERVAL logic is removed.
  _land_built flag guards a single _build_land() call in reset().

* Reward function: Ported from salp_iql_static — on success and timeout,
  the terminal reward is broadcast to ALL N_SALPS slots:
      success: rewards = [terminal_r] * N_SALPS
      timeout: rewards = [-50.0]     * N_SALPS
  (The parallel MAPPO version incorrectly used a 1-element list.)

* Warmup: WARMUP_STEPS raised from 0 → 7 000.

Architecture (unchanged from parallel MAPPO version)
------------------------------------------------------
* Actor:    Gaussian policy (mean + log_std), separate per agent.
* Critic:   Centralised value function V(global_obs), shared.
* Storage:  On-policy RolloutBuffer with GAE-λ.
* Update:   PPO-clip + value-function loss + entropy bonus, multiple epochs.

Parallelism stack (unchanged)
-------------------------------
1. SUBPROCESS WORKER POOL  (multiprocessing)
2. VECTORISED PHYSICS inside each worker
3. BATCHED NEURAL-NET INFERENCE  (act_batch)
4. TORCH COMPILE  (torch.compile, PyTorch >= 2.0 + CUDA only)
5. PINNED MEMORY + NON-BLOCKING CUDA TRANSFERS
6. FUSED ADAM (torch.optim.Adam with fused=True on CUDA)
"""

from __future__ import annotations

import dataclasses
import math
import os
import pickle
import random
import threading
import time
from multiprocessing import Process, Queue as MPQueue
from typing import List, Optional, Tuple

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
OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX = 0.5, 1.5

# Network dims
LOCAL_OBS_DIM    = 10
GLOBAL_STATE_DIM = LOCAL_OBS_DIM * N_SALPS
ACTION_DIM       = 2

# MAPPO-specific
ROLLOUT_STEPS    = 512
PPO_EPOCHS       = 5
PPO_CLIP         = 0.2
VALUE_COEFF      = 0.5
ENTROPY_COEFF    = 0.01
MAX_GRAD_NORM    = 0.5
SHARE_PARAMS     = False

# Training
MAX_EPISODES        = 10_000
CHECKPOINT_INTERVAL = 500
WARMUP_STEPS        = 7_000   # ← raised from 0

# Default hypers
DEFAULT_LR_ACTOR    = 3e-4
DEFAULT_LR_CRITIC   = 1e-3
DEFAULT_GAMMA       = 0.97
DEFAULT_GAE_LAMBDA  = 0.95
DEFAULT_BATCH_SIZE  = 256
DEFAULT_PPO_CLIP    = PPO_CLIP
DEFAULT_PPO_EPOCHS  = PPO_EPOCHS
DEFAULT_ENTROPY     = ENTROPY_COEFF
DEFAULT_VALUE_COEFF = VALUE_COEFF

# Logging
TIMESTEP_LOG_FLUSH_INTERVAL = 5_000

# Paths
SAVE_DIR          = "salp_saves_mappo_static"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH   = os.path.join(SAVE_DIR, "mappo_checkpoint.pt")
EPISODE_LOG_PATH  = os.path.join(SAVE_DIR, "episode_log.csv")
TIMESTEP_LOG_PATH = os.path.join(SAVE_DIR, "timestep_log.csv")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COMPILE = hasattr(torch, "compile") and DEVICE.type == "cuda"

print(f"Device: {DEVICE}  |  torch.compile: {USE_COMPILE}"
      f"  |  Workers: {NUM_WORKERS}  |  Envs/worker: {ENVS_PER_WORKER}"
      f"  |  Total envs: {TOTAL_ENVS}  |  Algo: MAPPO-Static"
      f"  |  Shared params: {SHARE_PARAMS}"
      f"  |  Warmup steps: {WARMUP_STEPS}")


# =============================================================================
# HYPERPARAMETER DATACLASS
# =============================================================================

@dataclasses.dataclass
class HParams:
    lr_actor:     float = DEFAULT_LR_ACTOR
    lr_critic:    float = DEFAULT_LR_CRITIC
    gamma:        float = DEFAULT_GAMMA
    gae_lambda:   float = DEFAULT_GAE_LAMBDA
    batch_size:   int   = DEFAULT_BATCH_SIZE
    ppo_clip:     float = DEFAULT_PPO_CLIP
    ppo_epochs:   int   = DEFAULT_PPO_EPOCHS
    entropy_coef: float = DEFAULT_ENTROPY
    value_coef:   float = DEFAULT_VALUE_COEFF


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
    best = int(np.argmin(dx * dx + dy * dy))
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

    def forward(self, obs: torch.Tensor):
        h    = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        lsig = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std  = lsig.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs: torch.Tensor):
        mean, std = self(obs)
        dist      = torch.distributions.Normal(mean, std)
        raw       = dist.rsample()
        action    = raw.clamp(-1.0, 1.0)
        log_prob  = dist.log_prob(raw).sum(-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std = self(obs)
        dist      = torch.distributions.Normal(mean, std)
        log_prob  = dist.log_prob(action).sum(-1)
        entropy   = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, global_dim: int = GLOBAL_STATE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, hp: HParams):
        T, E, A = rollout_steps, n_envs, N_SALPS
        self.T, self.E, self.A = T, E, A
        self.hp   = hp
        self.ptr  = 0
        self.full = False

        self.obs       = np.zeros((T, E, A, LOCAL_OBS_DIM),  dtype=np.float32)
        self.actions   = np.zeros((T, E, A, ACTION_DIM),     dtype=np.float32)
        self.log_probs = np.zeros((T, E, A),                 dtype=np.float32)
        self.rewards   = np.zeros((T, E, A),                 dtype=np.float32)
        self.dones     = np.zeros((T, E),                    dtype=np.float32)
        self.values    = np.zeros((T, E),                    dtype=np.float32)

        self.returns    = np.zeros((T, E),    dtype=np.float32)
        self.advantages = np.zeros((T, E, A), dtype=np.float32)

    def reset(self):
        self.ptr  = 0
        self.full = False

    def push(self, obs, actions, log_probs, rewards, dones, values):
        t = self.ptr
        self.obs[t]       = obs
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.dones[t]     = dones
        self.values[t]    = values
        self.ptr += 1
        if self.ptr >= self.T:
            self.full = True

    def compute_returns(self, last_values: np.ndarray):
        gamma  = self.hp.gamma
        lam    = self.hp.gae_lambda
        T, E   = self.T, self.E

        gae      = np.zeros(E, dtype=np.float32)
        last_val = last_values.copy()

        for t in reversed(range(T)):
            team_r  = self.rewards[t].mean(axis=-1)
            next_v  = last_val * (1.0 - self.dones[t])
            delta   = team_r + gamma * next_v - self.values[t]
            gae     = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            self.returns[t]    = gae + self.values[t]
            self.advantages[t] = gae[:, None]
            last_val = self.values[t]

        flat = self.advantages.reshape(-1)
        self.advantages = ((self.advantages - flat.mean()) /
                           (flat.std() + 1e-8))

    def get_loader(self, batch_size: int):
        T, E, A = self.T, self.E, self.A
        N = T * E
        obs_f = self.obs.reshape(N, A, LOCAL_OBS_DIM)
        act_f = self.actions.reshape(N, A, ACTION_DIM)
        lp_f  = self.log_probs.reshape(N, A)
        ret_f = self.returns.reshape(N)
        adv_f = self.advantages.reshape(N, A)

        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            b = idx[start : start + batch_size]
            def _t(arr):
                return torch.from_numpy(arr[b]).to(DEVICE, non_blocking=True)
            yield _t(obs_f), _t(act_f), _t(lp_f), _t(ret_f), _t(adv_f)


# =============================================================================
# MAPPO AGENT
# =============================================================================

def _unwrap_module(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):   return m._orig_mod
    if hasattr(m, "__wrapped__"): return m.__wrapped__
    return m


class MAPPO:
    def __init__(self, hp: HParams):
        self.hp = hp

        if SHARE_PARAMS:
            shared = Actor().to(DEVICE)
            self.actors = [shared] * N_SALPS
        else:
            self.actors = [Actor().to(DEVICE) for _ in range(N_SALPS)]

        self.critic = Critic().to(DEVICE)

        adam_kw = {"fused": True} if DEVICE.type == "cuda" else {}
        if SHARE_PARAMS:
            self.actor_opts = [
                optim.Adam(self.actors[0].parameters(), lr=hp.lr_actor, **adam_kw)
            ]
        else:
            self.actor_opts = [
                optim.Adam(a.parameters(), lr=hp.lr_actor, **adam_kw)
                for a in self.actors
            ]
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=hp.lr_critic, **adam_kw
        )

        if USE_COMPILE:
            compiled_actors = [torch.compile(a, mode="reduce-overhead")
                               for a in (self.actors[:1] if SHARE_PARAMS
                                         else self.actors)]
            self.actors = compiled_actors * N_SALPS if SHARE_PARAMS else compiled_actors
            self.critic = torch.compile(self.critic, mode="reduce-overhead")

    def act_batch(
        self,
        all_env_obs: List[List[np.ndarray]],
    ) -> Tuple[List[List[np.ndarray]], List[List[float]], np.ndarray]:
        n_envs = len(all_env_obs)

        agent_obs_t = []
        for i in range(N_SALPS):
            obs_np = np.stack([all_env_obs[e][i] for e in range(n_envs)])
            agent_obs_t.append(
                torch.from_numpy(obs_np).to(DEVICE, non_blocking=True)
            )

        global_obs_t = torch.cat(agent_obs_t, dim=-1)
        all_actions  = []
        all_log_prob = []

        with torch.no_grad():
            for i in range(N_SALPS):
                a, lp = self.actors[i].get_action(agent_obs_t[i])
                all_actions.append(a.cpu().numpy())
                all_log_prob.append(lp.cpu().numpy())
            values = self.critic(global_obs_t).cpu().numpy()

        actions_out   = [[all_actions[ag][ev]           for ag in range(N_SALPS)]
                         for ev in range(n_envs)]
        log_probs_out = [[float(all_log_prob[ag][ev])   for ag in range(N_SALPS)]
                         for ev in range(n_envs)]

        return actions_out, log_probs_out, values

    def get_values(self, all_env_obs: List[List[np.ndarray]]) -> np.ndarray:
        n_envs = len(all_env_obs)
        obs_np = np.concatenate(
            [np.stack([all_env_obs[e][i] for e in range(n_envs)])
             for i in range(N_SALPS)],
            axis=-1,
        )
        obs_t = torch.from_numpy(obs_np).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            return self.critic(obs_t).cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict:
        hp     = self.hp
        losses = {"actor": [], "critic": [], "entropy": []}

        for _epoch in range(hp.ppo_epochs):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in \
                    buffer.get_loader(hp.batch_size):
                B = obs_b.shape[0]

                global_obs_b = obs_b.reshape(B, -1)
                values_pred  = self.critic(global_obs_b)
                critic_loss  = F.mse_loss(values_pred, ret_b)

                self.critic_opt.zero_grad(set_to_none=True)
                (hp.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_opt.step()
                losses["critic"].append(critic_loss.item())

                total_actor_loss   = 0.0
                total_entropy_loss = 0.0

                unique_actors = [self.actors[0]] if SHARE_PARAMS else self.actors
                for opt in self.actor_opts:
                    opt.zero_grad(set_to_none=True)

                for i, actor in enumerate(self.actors):
                    new_lp, entropy = actor.evaluate(obs_b[:, i], act_b[:, i])
                    ratio  = (new_lp - old_lp_b[:, i]).exp()
                    adv_i  = adv_b[:, i]
                    surr1  = ratio * adv_i
                    surr2  = ratio.clamp(1.0 - hp.ppo_clip,
                                         1.0 + hp.ppo_clip) * adv_i
                    actor_loss   = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()
                    loss = actor_loss + hp.entropy_coef * entropy_loss
                    loss.backward()
                    total_actor_loss   += actor_loss.item()
                    total_entropy_loss += entropy_loss.item()

                for opt in self.actor_opts:
                    for actor in (unique_actors if SHARE_PARAMS else self.actors):
                        nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                    opt.step()

                losses["actor"].append(total_actor_loss / N_SALPS)
                losses["entropy"].append(total_entropy_loss / N_SALPS)

        return {k: float(np.mean(v)) for k, v in losses.items()}

    def save(self, path: str = CHECKPOINT_PATH) -> None:
        def state(m):
            return _unwrap_module(m).state_dict()
        unique_actors = [self.actors[0]] if SHARE_PARAMS else self.actors
        torch.save({
            "actors":       [state(a) for a in unique_actors],
            "critic":       state(self.critic),
            "actor_opts":   [o.state_dict() for o in self.actor_opts],
            "critic_opt":   self.critic_opt.state_dict(),
            "share_params": SHARE_PARAMS,
        }, path)
        print(f"  Saved -> {path}")

    def load(self, path: str = CHECKPOINT_PATH) -> bool:
        if not os.path.exists(path):
            print(f"  No checkpoint at {path}")
            return False
        data = torch.load(path, map_location=DEVICE, weights_only=True)
        unique_actors = [self.actors[0]] if SHARE_PARAMS else self.actors
        for i, a in enumerate(unique_actors):
            actor_states = data["actors"]
            idx = 0 if (data.get("share_params") and not SHARE_PARAMS) else i
            idx = min(idx, len(actor_states) - 1)
            _unwrap_module(a).load_state_dict(actor_states[idx])
        _unwrap_module(self.critic).load_state_dict(data["critic"])
        for o, s in zip(self.actor_opts, data["actor_opts"]):
            o.load_state_dict(s)
        self.critic_opt.load_state_dict(data["critic_opt"])
        print(f"  Loaded <- {path}")
        return True


# =============================================================================
# ENVIRONMENT  (static obstacle layout)
# =============================================================================

class FastEnv:
    RADII      = np.array([14., 16., 18., 16., 14.], dtype=np.float32)
    SEMI_A     = RADII * 1.3
    SEMI_B     = RADII * 0.8
    MAX_EXTENT = np.maximum(SEMI_A, SEMI_B)

    def __init__(self, W=W_DEFAULT, H=H_DEFAULT, margin=MARGIN_DEFAULT,
                 worker_id=0, env_idx=0):
        self.W, self.H, self.margin = W, H, margin
        self.worker_id  = worker_id
        self.env_idx    = env_idx

        self.pos               = np.zeros((N_SALPS, 2), dtype=np.float32)
        self.vel               = np.zeros((N_SALPS, 2), dtype=np.float32)
        self.land_polys: List[np.ndarray] = []
        self._land_built       = False   # ← static: build once, never again
        self.goal_pos          = np.zeros(2, dtype=np.float32)
        self.episode_steps     = 0
        self.max_steps         = 700
        self.total_reward      = 0.0
        self._prev_dists       = None
        self.collision_flags   = np.zeros(N_SALPS, dtype=bool)
        self.wall_flags        = np.zeros(N_SALPS, dtype=bool)
        self.last_actions      = np.zeros((N_SALPS, 2), dtype=np.float32)
        self.episodes_completed = 0

        # Per-salp episode accumulators (reset each episode)
        self.salp_collision_counts = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_wall_counts      = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_reward_totals    = np.zeros(N_SALPS, dtype=np.float64)

    def _build_land(self):
        m     = self.margin + 120
        specs = [(55,18),(70,20),(78,22),(48,16),(42,14),(52,16)]
        placed, polys = [], []
        scale = random.uniform(OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX)
        max_r = min(self.W, self.H) // 6
        for r_base, pts in specs:
            r = min(int(r_base * scale), max_r)
            for _ in range(100):
                cx = random.uniform(m, self.W - m)
                cy = random.uniform(m, self.H - m)
                if not all(math.hypot(cx-ox, cy-oy) >= r+orad+60
                           for ox, oy, orad in placed):
                    continue
                raw     = generate_land_polygon(cx, cy, r, points=pts)
                clamped = raw.copy()
                clamped[:, 0] = np.clip(raw[:, 0], m, self.W - m)
                clamped[:, 1] = np.clip(raw[:, 1], m, self.H - m)
                test_pts = np.array([
                    [self.W*0.25, cy], [self.W*0.5, cy], [self.W*0.75, cy],
                    [cx, self.H*0.25], [cx, self.H*0.5], [cx, self.H*0.75],
                ], dtype=np.float32)
                if any(point_in_poly(p[0], p[1], clamped) for p in test_pts):
                    continue
                placed.append((cx, cy, r))
                polys.append(clamped)
                break
        self.land_polys = polys

    def _point_in_land(self, px, py, buffer_px=0.0) -> bool:
        for poly in self.land_polys:
            if point_in_poly(px, py, poly):
                return True
            if buffer_px > 0:
                cx, cy = closest_on_poly(px, py, poly)
                if math.hypot(cx - px, cy - py) <= buffer_px:
                    return True
        return False

    def _rand_goal(self) -> np.ndarray:
        inn = self.margin + 90
        for _ in range(1000):
            gx = random.uniform(inn, self.W - inn)
            gy = random.uniform(inn, self.H - inn)
            if not self._point_in_land(gx, gy, buffer_px=GOAL_RADIUS + 20):
                return np.array([gx, gy], dtype=np.float32)
        return np.array([self.W * 0.85, self.H * 0.15], dtype=np.float32)

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
                if dist < 1e-6: continue
                min_d = self.RADII[i] + self.RADII[j]
                if dist < min_d:
                    unit = d / dist
                    corr = unit * ((min_d - dist) * 0.7)
                    self.pos[i] -= corr; self.pos[j] += corr
                    vn = float(np.dot(self.vel[j] - self.vel[i], unit))
                    if vn < 0:
                        imp = (-(1 + restitution) * vn / 2.0) * unit
                        self.vel[i] -= imp; self.vel[j] += imp

    def _wall_bounce(self):
        self.wall_flags[:] = False
        lo   = self.margin + self.MAX_EXTENT
        hi_x = self.W - self.margin - self.MAX_EXTENT
        hi_y = self.H - self.margin - self.MAX_EXTENT
        for i in range(N_SALPS):
            hit = False
            if self.pos[i, 0] < lo[i]:
                self.pos[i, 0] = lo[i];   self.vel[i, 0] =  abs(self.vel[i, 0])*0.35; hit=True
            if self.pos[i, 0] > hi_x[i]:
                self.pos[i, 0] = hi_x[i]; self.vel[i, 0] = -abs(self.vel[i, 0])*0.35; hit=True
            if self.pos[i, 1] < lo[i]:
                self.pos[i, 1] = lo[i];   self.vel[i, 1] =  abs(self.vel[i, 1])*0.35; hit=True
            if self.pos[i, 1] > hi_y[i]:
                self.pos[i, 1] = hi_y[i]; self.vel[i, 1] = -abs(self.vel[i, 1])*0.35; hit=True
            self.wall_flags[i] = hit

    def _resolve_land_collisions(self):
        self.collision_flags[:] = False
        for i in range(N_SALPS):
            px, py = float(self.pos[i, 0]), float(self.pos[i, 1])
            for poly in self.land_polys:
                cx, cy  = closest_on_poly(px, py, poly)
                dx, dy  = px - cx, py - cy
                dist    = math.hypot(dx, dy)
                inside  = point_in_poly(px, py, poly)
                min_sep = float(self.MAX_EXTENT[i]) + 2.0
                if inside or dist < min_sep:
                    if dist < 1e-8:
                        cent   = polygon_centroid(poly)
                        dx, dy = px - cent[0], py - cent[1]
                        dist   = math.hypot(dx, dy)
                        if dist < 1e-8:
                            dx, dist = 1.0, 1.0
                    nx, ny = dx / dist, dy / dist
                    self.pos[i] = np.array([cx + nx*min_sep, cy + ny*min_sep],
                                           dtype=np.float32)
                    vn = self.vel[i, 0]*nx + self.vel[i, 1]*ny
                    if vn < 0:
                        self.vel[i, 0] -= 1.55 * vn * nx
                        self.vel[i, 1] -= 1.55 * vn * ny
                    self.vel[i] *= 0.90
                    self.collision_flags[i] = True
                    px, py = float(self.pos[i, 0]), float(self.pos[i, 1])

    def _resolve_link_collisions(self):
        for i in range(N_SALPS - 1):
            for poly in self.land_polys:
                n = len(poly)
                for j in range(n):
                    p1, p2 = poly[j], poly[(j+1) % n]
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

    def step(self, actions, spring_iters=3):
        self.last_actions = np.array(actions, dtype=np.float32)
        tv  = self.last_actions * THRUST_MAG
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

    def get_obs(self) -> List[np.ndarray]:
        gx, gy     = self.goal_pos
        gx_n, gy_n = gx / self.W, gy / self.H
        obs_list   = []
        for i in range(N_SALPS):
            x, y         = float(self.pos[i, 0]), float(self.pos[i, 1])
            dx, dy       = gx - x, gy - y
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

    # ── Reward (ported from salp_iql_static) ─────────────────────────────────

    def reward(self):
        """
        Reward function (matching salp_iql_static):
          - Smooth normalised distance shaping: delta/(W+H) * 10
          - Time penalty: -0.5 / step
          - Obstacle collision: -15.0
          - Wall hit: -3.0
          - Success: 50 base + 200*(1 - steps/max_steps) efficiency bonus,
                     broadcast to ALL N_SALPS slots
          - Timeout: -50.0, broadcast to ALL N_SALPS slots

        Returns: rewards, team_reward, min_dist, done, success, timeout, per_salp_log
        """
        diff        = self.pos - self.goal_pos
        local_dists = np.linalg.norm(diff, axis=1)
        min_d       = float(local_dists.min())

        if self._prev_dists is None:
            self._prev_dists = local_dists.copy()

        diag         = self.W + self.H
        rewards      = []
        per_salp_log = []

        for i in range(N_SALPS):
            prev_d = self._prev_dists[i]
            curr_d = local_dists[i]

            # Smooth distance shaping normalised by env diagonal
            delta = (prev_d - curr_d) / diag
            r     = delta * 10.0

            # Time penalty
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
                "salp":         i,
                "reward_step":  r,
                "collision":    int(collision_hit),
                "wall_hit":     int(wall_hit),
                "dist_to_goal": float(curr_d),
                "prev_dist":    float(prev_d),
                "dist_delta":   float(prev_d - curr_d),
                "pos_x":        float(self.pos[i, 0]),
                "pos_y":        float(self.pos[i, 1]),
                "vel_x":        float(self.vel[i, 0]),
                "vel_y":        float(self.vel[i, 1]),
                "speed":        float(math.hypot(self.vel[i, 0], self.vel[i, 1])),
                "action_x":     float(self.last_actions[i, 0]),
                "action_y":     float(self.last_actions[i, 1]),
                "goal_x":       float(self.goal_pos[0]),
                "goal_y":       float(self.goal_pos[1]),
            })

        self._prev_dists = local_dists.copy()

        done = success = timeout = False

        if min_d < GOAL_RADIUS:
            efficiency_bonus = 200.0 * (1.0 - self.episode_steps / self.max_steps)
            terminal_r = 50.0 + efficiency_bonus
            # Broadcast terminal reward to all salps (matching iql_static)
            rewards = [terminal_r] * N_SALPS
            for entry in per_salp_log:
                entry["reward_step"] = terminal_r
            done = success = True

        elif self.episode_steps >= self.max_steps:
            # Broadcast timeout penalty to all salps (matching iql_static)
            rewards = [-50.0] * N_SALPS
            for entry in per_salp_log:
                entry["reward_step"] = -50.0
            done    = True
            timeout = True

        # Link distances for logging
        link_dists = [float(np.linalg.norm(self.pos[i+1] - self.pos[i]))
                      for i in range(N_SALPS - 1)]

        rewards     = [float(r) for r in rewards]
        team_reward = float(np.mean(rewards))

        # Update per-salp episode totals and finalise log entries
        for i, r in enumerate(rewards):
            self.salp_reward_totals[i] += r
            entry = per_salp_log[i]
            entry["reward_step"]    = r
            entry["team_reward"]    = team_reward
            entry["success"]        = int(success)
            entry["timeout"]        = int(timeout)
            entry["min_dist_team"]  = min_d
            entry["episode_steps"]  = self.episode_steps
            entry["link_dist_next"] = link_dists[i] if i < N_SALPS - 1 else float("nan")

        return rewards, team_reward, min_d, done, success, timeout, per_salp_log

    def reset(self):
        # Static: build obstacles exactly once, never rebuild
        if not self._land_built:
            self._build_land()
            self._land_built = True

        pad   = self.margin + 100
        start = np.array([pad, self.H - pad], dtype=np.float32)

        for _ in range(200):
            heading = random.uniform(0, 2 * math.pi)
            ax, ay  = math.cos(heading), math.sin(heading)
            positions = []
            valid     = True
            for i in range(N_SALPS):
                pos = start + np.array([ax*i*LINK_LEN, ay*i*LINK_LEN], dtype=np.float32)
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
            positions = [start + np.array([s2*i*LINK_LEN, -s2*i*LINK_LEN])
                         for i in range(N_SALPS)]

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
            "worker_id":             worker_id,
            "env_idx":               env_idx,
            "obs":                   env.get_obs(),
            "reward":                [0.0] * N_SALPS,
            "team_reward":           0.0,
            "dist":                  float("inf"),
            "done":                  False,
            "success":               False,
            "timeout":               False,
            "total_reward":          0.0,
            "episode_steps":         0,
            "collision_sum":         0,
            "per_salp_log":          [],
            "salp_collision_counts": [0] * N_SALPS,
            "salp_wall_counts":      [0] * N_SALPS,
            "salp_reward_totals":    [0.0] * N_SALPS,
        })

    while True:
        msg = action_q.get()
        if msg is None:
            break

        actions_batch = msg

        for env_idx, env in enumerate(envs):
            env.step(actions_batch[env_idx], spring_iters=3)
            rewards, team_r, dist, done, success, timeout, per_salp_log = env.reward()
            env.total_reward += team_r

            # Attach worker/env context to each salp log entry
            for entry in per_salp_log:
                entry["worker_id"]    = worker_id
                entry["env_idx"]      = env_idx
                entry["episode_step"] = env.episode_steps

            result = {
                "worker_id":             worker_id,
                "env_idx":               env_idx,
                "obs":                   env.get_obs(),
                "reward":                rewards,
                "team_reward":           team_r,
                "dist":                  dist,
                "done":                  done,
                "success":               success,
                "timeout":               timeout,
                "total_reward":          env.total_reward,
                "episode_steps":         env.episode_steps,
                # Use full-episode collision count (not just current step flags)
                "collision_sum":         int(env.salp_collision_counts.sum()),
                "per_salp_log":          per_salp_log,
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
    """Writes episode_log.csv — one row per completed episode."""

    EPISODE_COLS = [
        "episode", "worker_id", "env_idx",
        "reward", "success", "timeout", "collisions", "steps",
        "salp0_reward",      "salp1_reward",      "salp2_reward",
        "salp3_reward",      "salp4_reward",
        "salp0_collisions",  "salp1_collisions",  "salp2_collisions",
        "salp3_collisions",  "salp4_collisions",
        "salp0_wall_hits",   "salp1_wall_hits",   "salp2_wall_hits",
        "salp3_wall_hits",   "salp4_wall_hits",
    ]

    def __init__(self, path: str = EPISODE_LOG_PATH):
        self.path    = path
        self._buffer: List[dict] = []
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
    """

    TS_COLS = [
        "episode", "worker_id", "env_idx", "episode_step", "salp",
        "reward_step", "team_reward",
        "collision", "wall_hit",
        "dist_to_goal", "prev_dist", "dist_delta", "min_dist_team",
        "goal_x", "goal_y",
        "pos_x", "pos_y", "vel_x", "vel_y", "speed",
        "action_x", "action_y",
        "link_dist_next",
        "success", "timeout", "episode_steps",
    ]

    def __init__(self, path: str = TIMESTEP_LOG_PATH,
                 flush_every: int = TIMESTEP_LOG_FLUSH_INTERVAL):
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

def save_checkpoint(agent: MAPPO, episodes_done: int) -> None:
    agent.save(CHECKPOINT_PATH)
    with open(os.path.join(SAVE_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"episodes_done": episodes_done}, f)


def load_checkpoint(agent: MAPPO):
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

def analyze_and_plot() -> None:
    if not os.path.exists(EPISODE_LOG_PATH):
        return
    df = pd.read_csv(EPISODE_LOG_PATH)
    print(f"\n===== SUMMARY =====")
    print(f"Episodes:       {len(df)}")
    print(f"Successes:      {df['success'].sum()} ({100*df['success'].mean():.1f}%)")
    print(f"Avg collisions: {df['collisions'].mean():.2f}")
    ss = df[df["success"] == 1]["steps"]
    if len(ss):
        print(f"Avg steps/goal: {ss.mean():.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Salp MAPPO-Static — {TOTAL_ENVS} envs ({NUM_WORKERS} workers)")
    w = 100

    df["reward"].rolling(w).mean().plot(ax=axes[0,0],   title="Team reward (rolling)")
    df["success"].rolling(w).mean().plot(ax=axes[0,1],  title="Success rate (rolling)")
    df["collisions"].rolling(w).mean().plot(ax=axes[0,2], title="Collisions (rolling)")
    df["steps"].rolling(w).mean().plot(ax=axes[1,0],    title="Steps (rolling)")

    for i in range(N_SALPS):
        df[f"salp{i}_reward"].rolling(w).mean().plot(ax=axes[1,1], label=f"salp {i}")
    axes[1,1].set_title("Per-salp reward (rolling)")
    axes[1,1].legend(fontsize=8)

    for i in range(N_SALPS):
        df[f"salp{i}_collisions"].rolling(w).mean().plot(ax=axes[1,2], label=f"salp {i}")
    axes[1,2].set_title("Per-salp collisions (rolling)")
    axes[1,2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(out, dpi=100)
    print(f"Plot -> {out}")


# =============================================================================
# CORE TRAINING LOOP
# =============================================================================

def _run_training_loop(
    agent:        MAPPO,
    ep_logger:    EpisodeLogger,
    ts_logger:    TimestepLogger,
    max_episodes: int,
    start_ep:     int = 0,
) -> None:

    hp = agent.hp

    buffer    = RolloutBuffer(ROLLOUT_STEPS, TOTAL_ENVS, hp)
    action_qs = [MPQueue(maxsize=4) for _ in range(NUM_WORKERS)]
    result_q  = MPQueue(maxsize=0)

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

    # Collect initial observations
    obs_store = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
    for _ in range(TOTAL_ENVS):
        r = result_q.get()
        w, e = r["worker_id"], r["env_idx"]
        obs_store[w][e] = r["obs"]

    episodes_done = start_ep
    success_count = 0
    total_steps   = 0

    # Track current episode index per (worker, env) slot
    ep_idx_store = [[episodes_done + w * ENVS_PER_WORKER + e
                     for e in range(ENVS_PER_WORKER)]
                    for w in range(NUM_WORKERS)]

    pbar = tqdm(
        total=max_episodes,
        initial=start_ep,
        desc=f"MAPPO-Static ({TOTAL_ENVS} envs, {DEVICE})",
        dynamic_ncols=True,
    )

    try:
        while episodes_done < max_episodes:
            # ================================================================
            # ROLLOUT PHASE
            # ================================================================
            buffer.reset()

            last_acts = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
            last_lps  = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]

            for step in range(ROLLOUT_STEPS):
                all_obs = [obs_store[w][e]
                           for w in range(NUM_WORKERS)
                           for e in range(ENVS_PER_WORKER)]

                # Warmup: use random actions before WARMUP_STEPS
                if total_steps < WARMUP_STEPS:
                    all_acts_list = [
                        [np.random.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
                         for _ in range(N_SALPS)]
                        for _ in range(TOTAL_ENVS)
                    ]
                    all_lps_list = [[0.0] * N_SALPS for _ in range(TOTAL_ENVS)]
                    all_vals_np  = np.zeros(TOTAL_ENVS, dtype=np.float32)
                else:
                    all_acts_list, all_lps_list, all_vals_np = agent.act_batch(all_obs)

                # Send actions to workers
                for w in range(NUM_WORKERS):
                    worker_acts = []
                    for e in range(ENVS_PER_WORKER):
                        flat_idx = w * ENVS_PER_WORKER + e
                        last_acts[w][e] = all_acts_list[flat_idx]
                        last_lps[w][e]  = all_lps_list[flat_idx]
                        worker_acts.append(all_acts_list[flat_idx])
                    action_qs[w].put(worker_acts)

                # Prepare buffer arrays for this step
                step_obs   = np.zeros((TOTAL_ENVS, N_SALPS, LOCAL_OBS_DIM), dtype=np.float32)
                step_acts  = np.zeros((TOTAL_ENVS, N_SALPS, ACTION_DIM),    dtype=np.float32)
                step_lps   = np.zeros((TOTAL_ENVS, N_SALPS),                dtype=np.float32)
                step_rews  = np.zeros((TOTAL_ENVS, N_SALPS),                dtype=np.float32)
                step_dones = np.zeros((TOTAL_ENVS,),                        dtype=np.float32)
                step_vals  = all_vals_np.copy()

                for w in range(NUM_WORKERS):
                    for e in range(ENVS_PER_WORKER):
                        flat = w * ENVS_PER_WORKER + e
                        step_obs[flat]  = np.stack(obs_store[w][e])
                        step_acts[flat] = np.stack(last_acts[w][e])
                        step_lps[flat]  = np.array(last_lps[w][e], dtype=np.float32)

                # Collect results
                for _ in range(TOTAL_ENVS):
                    r    = result_q.get()
                    w, e = r["worker_id"], r["env_idx"]
                    flat_idx = w * ENVS_PER_WORKER + e

                    step_rews[flat_idx]  = np.array(r["reward"], dtype=np.float32)
                    step_dones[flat_idx] = float(r["done"])
                    obs_store[w][e]      = r["obs"]
                    total_steps         += 1

                    # Timestep logging
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
                            warmup=(total_steps < WARMUP_STEPS),
                        )

                        ep_idx_store[w][e] = episodes_done
                        episodes_done += 1

                        if episodes_done % CHECKPOINT_INTERVAL == 0:
                            ep_logger.flush()
                            ts_logger.flush()
                            save_checkpoint(agent, episodes_done)
                            tqdm.write(f"  Checkpoint @ ep {episodes_done}")

                        if episodes_done >= max_episodes:
                            break

                buffer.push(step_obs, step_acts, step_lps,
                            step_rews, step_dones, step_vals)

                if episodes_done >= max_episodes:
                    break

            # ================================================================
            # GAE BOOTSTRAP + PPO UPDATE (skip during warmup)
            # ================================================================
            if total_steps >= WARMUP_STEPS:
                all_obs_now = [obs_store[w][e]
                               for w in range(NUM_WORKERS)
                               for e in range(ENVS_PER_WORKER)]
                last_values = agent.get_values(all_obs_now)
                buffer.compute_returns(last_values)
                agent.update(buffer)

    except KeyboardInterrupt:
        print("\nInterrupted — saving...")

    finally:
        for q in action_qs:
            try: q.put(None)
            except Exception: pass
        for p in workers:
            p.join(timeout=3)
        pbar.close()
        ep_logger.flush()
        ts_logger.flush()
        save_checkpoint(agent, episodes_done)
        print("All saved.")
        analyze_and_plot()


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESUME = True

    hp    = HParams()
    agent = MAPPO(hp)

    ep_logger = EpisodeLogger(EPISODE_LOG_PATH)
    ts_logger = TimestepLogger(TIMESTEP_LOG_PATH)

    start_ep = 0
    if RESUME:
        start_ep = load_checkpoint(agent)

    _run_training_loop(
        agent=agent,
        ep_logger=ep_logger,
        ts_logger=ts_logger,
        max_episodes=MAX_EPISODES,
        start_ep=start_ep,
    )


if __name__ == "__main__":
    main()
