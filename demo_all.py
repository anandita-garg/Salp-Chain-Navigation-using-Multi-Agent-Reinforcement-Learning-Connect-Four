# =============================================================================
# SHARED ENV UTILITIES  (used by IQLEnv, SalpFastEnv)
# =============================================================================
import math, random
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import os
import pickle
import queue
import sys
import threading
import time
from collections import deque
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Physics constants ────────────────────────────────────────────────────────
_N_SALPS     = 5
_LINK_LEN    = 68.0
_SPRING_K    = 1.0
_DRAG        = 0.955
_MAX_SPEED   = 10.0
_THRUST_MAG  = 0.60
_GOAL_RADIUS = 30
_W, _H, _MARGIN = 1400, 800, 70
_OBS_MIN, _OBS_MAX = 0.5, 1.5
HEADLESS = False
if not HEADLESS:
    import pygame

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COMPILE = hasattr(torch, "compile") and DEVICE.type == "cuda"

# ── MADDPG hyperparams ────────────────────────────────────────────────────────
_LR_ACTOR        = 1e-4
_LR_CRITIC       = 3e-4
_GAMMA_NN        = 0.97          # used by MADDPG / MAPPO (IQL uses its own)
_TAU             = 0.005
_BATCH_SIZE      = 256
_BUFFER_CAPACITY = 200000
_WARMUP_STEPS_MADDPG = 10000
_UPDATE_EVERY    = 4
_UPDATES_PER_STEP = 4
_OU_MU, _OU_THETA, _OU_SIGMA = 0.0, 0.15, 0.20
_OU_SIGMA_MIN, _OU_SIGMA_DECAY = 0.02, 0.9999
_LOCAL_OBS_DIM    = 10
_GLOBAL_STATE_DIM = _LOCAL_OBS_DIM * _N_SALPS
_ACTION_DIM       = 2
_CRITIC_INPUT_DIM = _GLOBAL_STATE_DIM + _ACTION_DIM * _N_SALPS

# ── MAPPO hyperparams ─────────────────────────────────────────────────────────
_ROLLOUT_STEPS  = 512
_PPO_EPOCHS     = 5
_PPO_CLIP       = 0.2
_VALUE_COEFF    = 0.5
_ENTROPY_COEFF  = 0.01
_MAX_GRAD_NORM  = 0.5
_SHARE_PARAMS   = False
_WARMUP_STEPS_MAPPO = 7000
_DEFAULT_LR_ACTOR   = 3e-4
_DEFAULT_LR_CRITIC  = 1e-3
_DEFAULT_GAMMA      = 0.97
_DEFAULT_GAE_LAMBDA = 0.95
_DEFAULT_BATCH_SIZE = 256

# ── IQL action space ─────────────────────────────────────────────────────────
_ACTION_DIRS = [
    (0.0,  0.0),
    (0.0, -1.0), (1.0, -1.0), (1.0,  0.0), (1.0,  1.0),
    (0.0,  1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0,-1.0),
]
_N_ACTIONS  = len(_ACTION_DIRS)
_ANGLE_BINS = 8


def _action_to_vector(action_idx: int) -> np.ndarray:
    dx, dy = _ACTION_DIRS[action_idx]
    vec = np.array([dx, dy], dtype=float)
    n   = math.hypot(dx, dy)
    return np.zeros(2) if n < 1e-9 else vec * (_THRUST_MAG / n)


def _angle_bin(angle_rad: float, num_bins: int = _ANGLE_BINS) -> int:
    wrapped = (angle_rad + math.pi) % (2 * math.pi)
    return int((wrapped / (2 * math.pi)) * num_bins) % num_bins


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _point_in_poly_xy(px: float, py: float, poly: np.ndarray) -> bool:
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cond   = (yi > py) != (yj > py)
    x_int  = (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
    return bool(np.sum(cond & (px < x_int)) % 2 == 1)


def _point_in_poly_arr(point, poly: np.ndarray) -> bool:
    """Accepts a 2-element array/tuple."""
    return _point_in_poly_xy(float(point[0]), float(point[1]), poly)


def _closest_on_poly(px: float, py: float, poly: np.ndarray) -> Tuple[float, float]:
    a   = poly
    b   = np.roll(poly, -1, axis=0)
    ab  = b - a
    ab2 = np.einsum("vi,vi->v", ab, ab)
    ap  = np.array([px - a[:, 0], py - a[:, 1]], dtype=np.float32).T
    t   = np.clip(np.einsum("vi,vi->v", ap, ab) / (ab2 + 1e-12), 0.0, 1.0)
    cp  = a + t[:, None] * ab
    d2  = (cp[:, 0] - px)**2 + (cp[:, 1] - py)**2
    b_  = int(np.argmin(d2))
    return float(cp[b_, 0]), float(cp[b_, 1])


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    return poly.mean(axis=0)


def _generate_land_polygon(cx, cy, base_r, points=18, jitter=0.2):
    ao     = random.uniform(0, 2 * math.pi)
    angles = ao + np.arange(points) * (2 * math.pi / points)
    rs     = base_r * (1 + np.random.uniform(-jitter, jitter, points))
    xs, ys = cx + np.cos(angles) * rs, cy + np.sin(angles) * rs
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)


# ── Salp (used only by IQLEnv) ───────────────────────────────────────────────

class _Salp:
    def __init__(self, radius, pos):
        self.radius    = radius
        self.semi_a    = radius * 1.3
        self.semi_b    = radius * 0.8
        self.pos       = np.array(pos, dtype=float)
        self.vel       = np.zeros(2, dtype=float)
        self.nozzle    = 0.0
        self.phase     = "rest"
        self.thrust_on = False

    @property
    def max_extent(self):
        return max(self.semi_a, self.semi_b)

    def reset(self, pos):
        self.pos[:]    = pos
        self.vel[:]    = 0.0
        self.nozzle    = 0.0
        self.phase     = "rest"
        self.thrust_on = False

# =============================================================================
# IQL AGENT
# =============================================================================

_IQL_LEARNING_RATE = 0.10
_IQL_GAMMA         = 0.97
_IQL_EPS_START     = 1.0
_IQL_EPS_MIN       = 0.05
_IQL_EPS_DECAY     = 0.995


class IndependentQLearner:
    def __init__(self, n_agents=_N_SALPS, n_actions=_N_ACTIONS,
                 alpha=_IQL_LEARNING_RATE, gamma=_IQL_GAMMA,
                 eps=_IQL_EPS_START, eps_min=_IQL_EPS_MIN,
                 eps_decay=_IQL_EPS_DECAY):
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self._make_tables()
        self.last_actions = [0] * n_agents

    def _make_tables(self):
        na = self.n_actions
        self.q_tables = [
            defaultdict(lambda: np.zeros(na, dtype=float))
            for _ in range(self.n_agents)
        ]

    def act(self, states, train=True):
        actions = []
        for i, s in enumerate(states):
            if train and random.random() < self.eps:
                a = random.randrange(self.n_actions)
            else:
                q    = self.q_tables[i][s]
                best = np.flatnonzero(q == q.max())
                a    = int(random.choice(best))
            actions.append(a)
        self.last_actions = actions
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        for i in range(self.n_agents):
            q      = self.q_tables[i][states[i]]
            next_q = self.q_tables[i][next_states[i]]
            target = rewards[i] + (1.0 - float(dones[i])) * self.gamma * next_q.max()
            q[actions[i]] += self.alpha * (target - q[actions[i]])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        plain = [{k: v.copy() for k, v in qt.items()} for qt in self.q_tables]
        with open(path, "wb") as f:
            pickle.dump({"q_tables": plain, "eps": self.eps}, f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        na = self.n_actions
        self.q_tables = []
        for plain in data["q_tables"]:
            qt = defaultdict(lambda: np.zeros(na, dtype=float))
            qt.update(plain)
            self.q_tables.append(qt)
        self.eps = data.get("eps", self.eps)
        return True
    
# =============================================================================
# MADDPG NETWORKS & AGENT
# =============================================================================

class _MaddpgActor(nn.Module):
    def __init__(self, obs_dim=_LOCAL_OBS_DIM, action_dim=_ACTION_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs)


class _MaddpgCritic(nn.Module):
    def __init__(self, input_dim=_CRITIC_INPUT_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, global_obs, all_actions):
        return self.net(torch.cat([global_obs, all_actions], dim=-1)).squeeze(-1)


class OUNoise:
    def __init__(self, size, mu=_OU_MU, theta=_OU_THETA, sigma=_OU_SIGMA):
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


class ReplayBuffer:
    def __init__(self, capacity=_BUFFER_CAPACITY):
        self.capacity = capacity
        self.size     = 0
        self.pos      = 0
        self.obs  = np.zeros((capacity, _N_SALPS, _LOCAL_OBS_DIM), dtype=np.float32)
        self.acts = np.zeros((capacity, _N_SALPS, _ACTION_DIM),    dtype=np.float32)
        self.rews = np.zeros((capacity, _N_SALPS),                 dtype=np.float32)
        self.nobs = np.zeros((capacity, _N_SALPS, _LOCAL_OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity,),                          dtype=np.float32)
        self._prefetch_result = None
        self._prefetch_thread = None

    def push(self, local_obs, actions, rewards, next_obs, done):
        p = self.pos
        self.obs[p]  = np.stack(local_obs)  if isinstance(local_obs, list) else local_obs
        self.acts[p] = np.stack(actions)     if isinstance(actions,   list) else actions
        self.rews[p] = np.array(rewards,     dtype=np.float32)
        self.nobs[p] = np.stack(next_obs)    if isinstance(next_obs,  list) else next_obs
        self.done[p] = float(done)
        self.pos  = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_tensors(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        def t(arr):
            return torch.from_numpy(arr[idx]).to(DEVICE, non_blocking=True)
        return t(self.obs), t(self.acts), t(self.rews), t(self.nobs), t(self.done)

    def sample(self, batch_size=_BATCH_SIZE):
        return self._sample_tensors(batch_size)

    def prefetch(self, batch_size=_BATCH_SIZE):
        if self.size < batch_size:
            return
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        def _work():
            self._prefetch_result = self._sample_tensors(batch_size)
        self._prefetch_thread = threading.Thread(target=_work, daemon=True)
        self._prefetch_thread.start()

    def get_prefetched(self, batch_size=_BATCH_SIZE):
        if self._prefetch_thread:
            self._prefetch_thread.join()
        if self._prefetch_result is not None:
            result = self._prefetch_result
            self._prefetch_result = None
            return result
        return self._sample_tensors(batch_size)

    def __len__(self):
        return self.size


def _soft_update(target, source, tau=_TAU):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


class MADDPG:
    def __init__(self, total_envs=1):
        self.actors         = [_MaddpgActor().to(DEVICE)  for _ in range(_N_SALPS)]
        self.critics        = [_MaddpgCritic().to(DEVICE) for _ in range(_N_SALPS)]
        self.target_actors  = [_MaddpgActor().to(DEVICE)  for _ in range(_N_SALPS)]
        self.target_critics = [_MaddpgCritic().to(DEVICE) for _ in range(_N_SALPS)]
        for i in range(_N_SALPS):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        adam_kwargs = {"fused": True} if DEVICE.type == "cuda" else {}
        self.actor_opts  = [optim.Adam(a.parameters(), lr=_LR_ACTOR,  **adam_kwargs) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=_LR_CRITIC, **adam_kwargs) for c in self.critics]
        self.noises      = [[OUNoise(_ACTION_DIM) for _ in range(_N_SALPS)]
                            for _ in range(total_envs)]
        self.noise_sigma = _OU_SIGMA

    def act_batch(self, all_env_obs: List[List[np.ndarray]], explore: bool = True) -> List[List[np.ndarray]]:
        n_envs = len(all_env_obs)
        agent_actions = []
        for i in range(_N_SALPS):
            obs_np = np.stack([all_env_obs[e][i] for e in range(n_envs)])
            obs_t  = torch.from_numpy(obs_np).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                a = self.actors[i](obs_t).cpu().numpy()
            if explore:
                sigma  = self.noise_sigma
                theta  = _OU_THETA
                states = np.stack([self.noises[e][i].state for e in range(n_envs)])
                noise  = theta * (-states) + sigma * np.random.randn(n_envs, _ACTION_DIM).astype(np.float32)
                states += noise
                for e in range(n_envs):
                    self.noises[e][i].state = states[e]
                a += states
            agent_actions.append(np.clip(a, -1.0, 1.0))
        return [
            [agent_actions[ag][ev] for ag in range(_N_SALPS)]
            for ev in range(n_envs)
        ]

    def update(self, buffer: ReplayBuffer):
        if len(buffer) < _BATCH_SIZE:
            return
        obs_b, act_b, rew_b, nobs_b, done_b = buffer.get_prefetched()
        B           = obs_b.shape[0]
        global_obs  = obs_b.reshape(B, -1)
        global_nobs = nobs_b.reshape(B, -1)
        all_actions = act_b.reshape(B, -1)
        with torch.no_grad():
            tgt_acts      = torch.stack([self.target_actors[i](nobs_b[:, i]) for i in range(_N_SALPS)], dim=1)
            tgt_acts_flat = tgt_acts.reshape(B, -1)
        for i in range(_N_SALPS):
            with torch.no_grad():
                tgt_q = rew_b[:, i] + _GAMMA_NN * (1.0 - done_b) * \
                        self.target_critics[i](global_nobs, tgt_acts_flat)
            cur_q       = self.critics[i](global_obs, all_actions)
            critic_loss = F.mse_loss(cur_q, tgt_q)
            self.critic_opts[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_opts[i].step()
            acts_for_actor = act_b.clone()
            for j in range(_N_SALPS):
                if j == i:
                    acts_for_actor[:, j] = self.actors[i](obs_b[:, i])
                else:
                    with torch.no_grad():
                        acts_for_actor[:, j] = self.actors[j](obs_b[:, j])
            actor_loss = -self.critics[i](global_obs, acts_for_actor.reshape(B, -1)).mean()
            self.actor_opts[i].zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_opts[i].step()
        for i in range(_N_SALPS):
            _soft_update(self.target_actors[i],  self.actors[i])
            _soft_update(self.target_critics[i], self.critics[i])
        buffer.prefetch()

    def decay_noise(self):
        self.noise_sigma = max(_OU_SIGMA_MIN, self.noise_sigma * _OU_SIGMA_DECAY)
        for env_noises in self.noises:
            for n in env_noises:
                n.sigma = self.noise_sigma

    def reset_noise(self, env_idx=None):
        if env_idx is not None:
            for n in self.noises[env_idx]: n.reset()
        else:
            for env_noises in self.noises:
                for n in env_noises: n.reset()

    def save(self, path):
        def state(m): return (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
        torch.save({
            "actors":         [state(a) for a in self.actors],
            "critics":        [state(c) for c in self.critics],
            "target_actors":  [state(a) for a in self.target_actors],
            "target_critics": [state(c) for c in self.target_critics],
            "actor_opts":     [o.state_dict() for o in self.actor_opts],
            "critic_opts":    [o.state_dict() for o in self.critic_opts],
            "noise_sigma":    self.noise_sigma,
        }, path)

    def load(self, path):
        if not os.path.exists(path): return False
        # Try torch.load first; fall back to pickle for .pkl files saved with pickle.dump
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=False)
        except RuntimeError:
            with open(path, "rb") as f:
                data = pickle.load(f)
        def unwrap(m): return m._orig_mod if hasattr(m, "_orig_mod") else m
        for i in range(_N_SALPS):
            unwrap(self.actors[i]).load_state_dict(data["actors"][i])
            unwrap(self.critics[i]).load_state_dict(data["critics"][i])
            unwrap(self.target_actors[i]).load_state_dict(data["target_actors"][i])
            unwrap(self.target_critics[i]).load_state_dict(data["target_critics"][i])
            self.actor_opts[i].load_state_dict(data["actor_opts"][i])
            self.critic_opts[i].load_state_dict(data["critic_opts"][i])
        self.noise_sigma = data.get("noise_sigma", _OU_SIGMA)
        for env_noises in self.noises:
            for n in env_noises: n.sigma = self.noise_sigma
        return True
    
# =============================================================================
# MAPPO NETWORKS & AGENT
# =============================================================================

@dataclasses.dataclass
class HParams:
    lr_actor:     float = _DEFAULT_LR_ACTOR
    lr_critic:    float = _DEFAULT_LR_CRITIC
    gamma:        float = _DEFAULT_GAMMA
    gae_lambda:   float = _DEFAULT_GAE_LAMBDA
    batch_size:   int   = _DEFAULT_BATCH_SIZE
    ppo_clip:     float = _PPO_CLIP
    ppo_epochs:   int   = _PPO_EPOCHS
    entropy_coef: float = _ENTROPY_COEFF
    value_coef:   float = _VALUE_COEFF


class _MappoActor(nn.Module):
    LOG_STD_MIN = -4.0
    LOG_STD_MAX =  1.0

    def __init__(self, obs_dim: int = _LOCAL_OBS_DIM,
                 action_dim: int = _ACTION_DIM, hidden: int = 64):
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


class _MappoCritic(nn.Module):
    def __init__(self, global_dim: int = _GLOBAL_STATE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, hp: HParams):
        T, E, A   = rollout_steps, n_envs, _N_SALPS
        self.T, self.E, self.A = T, E, A
        self.hp   = hp
        self.ptr  = 0
        self.full = False
        self.obs       = np.zeros((T, E, A, _LOCAL_OBS_DIM), dtype=np.float32)
        self.actions   = np.zeros((T, E, A, _ACTION_DIM),    dtype=np.float32)
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
        self.advantages = ((self.advantages - flat.mean()) / (flat.std() + 1e-8))

    def get_loader(self, batch_size: int):
        T, E, A = self.T, self.E, self.A
        N = T * E
        obs_f = self.obs.reshape(N, A, _LOCAL_OBS_DIM)
        act_f = self.actions.reshape(N, A, _ACTION_DIM)
        lp_f  = self.log_probs.reshape(N, A)
        ret_f = self.returns.reshape(N)
        adv_f = self.advantages.reshape(N, A)
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            b = idx[start: start + batch_size]
            def _t(arr):
                return torch.from_numpy(arr[b]).to(DEVICE, non_blocking=True)
            yield _t(obs_f), _t(act_f), _t(lp_f), _t(ret_f), _t(adv_f)


def _unwrap_module(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):   return m._orig_mod
    if hasattr(m, "__wrapped__"): return m.__wrapped__
    return m


class MAPPO:
    def __init__(self, hp: HParams = None):
        self.hp = hp or HParams()
        if _SHARE_PARAMS:
            shared = _MappoActor().to(DEVICE)
            self.actors = [shared] * _N_SALPS
        else:
            self.actors = [_MappoActor().to(DEVICE) for _ in range(_N_SALPS)]
        self.critic = _MappoCritic().to(DEVICE)
        adam_kw = {"fused": True} if DEVICE.type == "cuda" else {}
        if _SHARE_PARAMS:
            self.actor_opts = [optim.Adam(self.actors[0].parameters(), lr=self.hp.lr_actor, **adam_kw)]
        else:
            self.actor_opts = [optim.Adam(a.parameters(), lr=self.hp.lr_actor, **adam_kw) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.hp.lr_critic, **adam_kw)

    def act_batch(self, all_env_obs):
        n_envs = len(all_env_obs)
        agent_obs_t = []
        for i in range(_N_SALPS):
            obs_np = np.stack([all_env_obs[e][i] for e in range(n_envs)])
            agent_obs_t.append(torch.from_numpy(obs_np).to(DEVICE, non_blocking=True))
        global_obs_t = torch.cat(agent_obs_t, dim=-1)
        all_actions, all_log_prob = [], []
        with torch.no_grad():
            for i in range(_N_SALPS):
                a, lp = self.actors[i].get_action(agent_obs_t[i])
                all_actions.append(a.cpu().numpy())
                all_log_prob.append(lp.cpu().numpy())
            values = self.critic(global_obs_t).cpu().numpy()
        actions_out   = [[all_actions[ag][ev]         for ag in range(_N_SALPS)] for ev in range(n_envs)]
        log_probs_out = [[float(all_log_prob[ag][ev]) for ag in range(_N_SALPS)] for ev in range(n_envs)]
        return actions_out, log_probs_out, values

    def get_values(self, all_env_obs):
        n_envs = len(all_env_obs)
        obs_np = np.concatenate(
            [np.stack([all_env_obs[e][i] for e in range(n_envs)]) for i in range(_N_SALPS)], axis=-1)
        obs_t = torch.from_numpy(obs_np).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            return self.critic(obs_t).cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict:
        hp     = self.hp
        losses = {"actor": [], "critic": [], "entropy": []}
        for _epoch in range(hp.ppo_epochs):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in buffer.get_loader(hp.batch_size):
                B           = obs_b.shape[0]
                global_obs_b = obs_b.reshape(B, -1)
                values_pred  = self.critic(global_obs_b)
                critic_loss  = F.mse_loss(values_pred, ret_b)
                self.critic_opt.zero_grad(set_to_none=True)
                (hp.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), _MAX_GRAD_NORM)
                self.critic_opt.step()
                losses["critic"].append(critic_loss.item())
                total_actor_loss = total_entropy_loss = 0.0
                unique_actors = [self.actors[0]] if _SHARE_PARAMS else self.actors
                for opt in self.actor_opts:
                    opt.zero_grad(set_to_none=True)
                for i, actor in enumerate(self.actors):
                    new_lp, entropy = actor.evaluate(obs_b[:, i], act_b[:, i])
                    ratio  = (new_lp - old_lp_b[:, i]).exp()
                    adv_i  = adv_b[:, i]
                    surr1  = ratio * adv_i
                    surr2  = ratio.clamp(1.0 - hp.ppo_clip, 1.0 + hp.ppo_clip) * adv_i
                    loss   = -torch.min(surr1, surr2).mean() + hp.entropy_coef * (-entropy.mean())
                    loss.backward()
                    total_actor_loss   += (-torch.min(surr1, surr2).mean()).item()
                    total_entropy_loss += (-entropy.mean()).item()
                for opt in self.actor_opts:
                    for actor in (unique_actors if _SHARE_PARAMS else self.actors):
                        nn.utils.clip_grad_norm_(actor.parameters(), _MAX_GRAD_NORM)
                    opt.step()
                losses["actor"].append(total_actor_loss / _N_SALPS)
                losses["entropy"].append(total_entropy_loss / _N_SALPS)
        return {k: float(np.mean(v)) for k, v in losses.items()}

    def save(self, path: str) -> None:
        def state(m): return _unwrap_module(m).state_dict()
        unique_actors = [self.actors[0]] if _SHARE_PARAMS else self.actors
        torch.save({
            "actors":       [state(a) for a in unique_actors],
            "critic":       state(self.critic),
            "actor_opts":   [o.state_dict() for o in self.actor_opts],
            "critic_opt":   self.critic_opt.state_dict(),
            "share_params": _SHARE_PARAMS,
        }, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path): return False
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        unique_actors = [self.actors[0]] if _SHARE_PARAMS else self.actors
        for i, a in enumerate(unique_actors):
            idx = min(0 if data.get("share_params") else i, len(data["actors"]) - 1)
            _unwrap_module(a).load_state_dict(data["actors"][idx])
        _unwrap_module(self.critic).load_state_dict(data["critic"])
        for o, s in zip(self.actor_opts, data["actor_opts"]):
            o.load_state_dict(s)
        self.critic_opt.load_state_dict(data["critic_opt"])
        return True

# =============================================================================
# IQL ENVIRONMENT  (Salp-based, supports static and parallel obstacle modes)
# =============================================================================

class IQLEnv:
    """
    Args:
        rebuild_every_episode: False → static obstacles (IQL static)
                               True  → rebuild obstacles each reset (IQL parallel)
        headless: passed through for display flag
    """

    def __init__(self, W=_W, H=_H, margin=_MARGIN,
                 worker_id=0, env_idx=0,
                 headless=True, rebuild_every_episode=False):
        self.W, self.H = W, H
        self.margin    = margin
        self.worker_id = worker_id
        self.env_idx   = env_idx
        self.headless  = headless
        self.rebuild_every_episode = rebuild_every_episode

        radii      = [14, 16, 18, 16, 14]
        cx         = W / 2 - (_N_SALPS - 1) * _LINK_LEN / 2
        self.salps = [_Salp(radii[i], (cx + i * _LINK_LEN, H / 2))
                      for i in range(_N_SALPS)]

        self.land_polys            = []
        self._land_built           = False
        self._reset_counter        = 0
        self.goal_pos              = np.zeros(2, dtype=float)
        self.episode_steps         = 0
        self.max_steps             = 700
        self.total_reward          = 0.0
        self._prev_local_dists     = None
        self.last_collision_flags  = [False] * _N_SALPS
        self.last_wall_flags       = [False] * _N_SALPS
        self.last_actions          = [0]     * _N_SALPS
        self.episodes_completed    = 0
        self.salp_collision_counts = np.zeros(_N_SALPS, dtype=np.int32)
        self.salp_wall_counts      = np.zeros(_N_SALPS, dtype=np.int32)
        self.salp_reward_totals    = np.zeros(_N_SALPS, dtype=np.float64)

    # ── Obstacles ─────────────────────────────────────────────────────────────

    def _build_land(self):
        m     = self.margin + 120
        specs = [(55,18),(70,20),(78,22),(48,16),(42,14),(52,16)]
        placed, polys = [], []
        scale = random.uniform(_OBS_MIN, _OBS_MAX)
        max_r = min(self.W, self.H) // 6

        for r_base, pts in specs:
            r = min(int(r_base * scale), max_r)
            for _ in range(100):
                cx = random.uniform(m, self.W - m)
                cy = random.uniform(m, self.H - m)
                if not all(math.hypot(cx-ox,cy-oy) >= r+orad+60
                           for ox,oy,orad in placed):
                    continue
                raw     = _generate_land_polygon(cx, cy, r, points=pts)
                clamped = raw.copy()
                clamped[:, 0] = np.clip(raw[:, 0], m, self.W - m)
                clamped[:, 1] = np.clip(raw[:, 1], m, self.H - m)
                test_pts = np.array([
                    [self.W*0.25, cy],[self.W*0.5, cy],[self.W*0.75, cy],
                    [cx, self.H*0.25],[cx, self.H*0.5],[cx, self.H*0.75],
                ], dtype=np.float32)
                if any(_point_in_poly_arr(p, clamped) for p in test_pts):
                    continue
                placed.append((cx, cy, r))
                polys.append(clamped)
                break
        self.land_polys = polys

    def _point_in_land(self, point, buffer_px=0.0) -> bool:
        p = np.asarray(point, dtype=float)
        for poly in self.land_polys:
            if _point_in_poly_arr(p, poly):
                return True
            if buffer_px > 0:
                cx, cy = _closest_on_poly(float(p[0]), float(p[1]), poly)
                if math.hypot(cx - p[0], cy - p[1]) <= buffer_px:
                    return True
        return False

    def _rand_goal(self) -> np.ndarray:
        inn = self.margin + 90
        for _ in range(1_000):
            g = np.array([random.uniform(inn, self.W - inn),
                          random.uniform(inn, self.H - inn)], dtype=float)
            if not self._point_in_land(g, buffer_px=_GOAL_RADIUS + 20):
                return g
        return np.array([self.W * 0.85, self.H * 0.15], dtype=float)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _enforce_separation(self):
        for i in range(_N_SALPS):
            for j in range(i + 1, _N_SALPS):
                a, b   = self.salps[i], self.salps[j]
                dx, dy = b.pos[0]-a.pos[0], b.pos[1]-a.pos[1]
                dist   = math.hypot(dx, dy)
                min_d  = a.radius + b.radius
                if dist < min_d and dist > 1e-6:
                    nx, ny     = dx/dist, dy/dist
                    correction = (min_d - dist) * 0.5
                    a.pos[0] -= nx*correction; a.pos[1] -= ny*correction
                    b.pos[0] += nx*correction; b.pos[1] += ny*correction

    def _apply_springs(self):
        for _ in range(4):
            for i in range(_N_SALPS - 1):
                a, b  = self.salps[i], self.salps[i+1]
                delta = b.pos - a.pos
                dist  = math.hypot(delta[0], delta[1])
                if dist < 1e-6:
                    continue
                unit = delta / dist
                corr = unit * (dist - _LINK_LEN) * _SPRING_K * 0.5
                a.pos += corr; b.pos -= corr
                rel_v = b.vel - a.vel
                damp  = float(np.dot(rel_v, unit)) * unit * 0.06
                a.vel += damp; b.vel -= damp

    def _resolve_link_collisions(self):
        for i in range(_N_SALPS - 1):
            a, b = self.salps[i], self.salps[i+1]
            for poly in self.land_polys:
                n = len(poly)
                for j in range(n):
                    p1, p2 = poly[j], poly[(j+1) % n]
                    if _segments_intersect(a.pos, b.pos, p1, p2):
                        edge   = p2 - p1
                        normal = np.array([-edge[1], edge[0]], dtype=float)
                        normal /= math.hypot(normal[0], normal[1]) + 1e-8
                        mid    = (a.pos + b.pos) * 0.5
                        if np.dot(mid - p1, normal) < 0:
                            normal *= -1
                        for s in (a, b):
                            s.pos += normal * 6.0
                            vn = np.dot(s.vel, normal)
                            if vn < 0:
                                s.vel -= vn * normal
                        break

    def _enforce_rigid_links(self):
        for i in range(_N_SALPS - 1):
            a, b = self.salps[i], self.salps[i+1]
            dx   = b.pos[0]-a.pos[0]; dy = b.pos[1]-a.pos[1]
            dist = math.hypot(dx, dy) + 1e-8
            diff = (dist - _LINK_LEN) / dist
            cx_  = dx * 0.5 * diff; cy_ = dy * 0.5 * diff
            a.pos[0] += cx_; a.pos[1] += cy_
            b.pos[0] -= cx_; b.pos[1] -= cy_

    def _resolve_land_collisions(self):
        self.last_collision_flags = [False] * _N_SALPS
        for i, s in enumerate(self.salps):
            for poly in self.land_polys:
                cp    = np.array(_closest_on_poly(float(s.pos[0]), float(s.pos[1]), poly))
                delta = s.pos - cp
                dist  = math.hypot(delta[0], delta[1])
                inside = _point_in_poly_arr(s.pos, poly)
                min_sep = s.max_extent + 2.0
                if inside or dist < min_sep:
                    if dist < 1e-8:
                        cent  = _polygon_centroid(poly)
                        delta = s.pos - cent
                        dist  = math.hypot(delta[0], delta[1])
                        if dist < 1e-8:
                            delta = np.array([1.0, 0.0]); dist = 1.0
                    normal = delta / dist
                    s.pos  = cp + normal * min_sep
                    vn     = np.dot(s.vel, normal)
                    if vn < 0:
                        s.vel -= 1.55 * vn * normal
                    s.vel *= 0.90
                    self.last_collision_flags[i] = True

    def _wall_bounce_all(self):
        self.last_wall_flags = [False] * _N_SALPS
        for i, s in enumerate(self.salps):
            r = s.max_extent; hit = False
            if s.pos[0] < self.margin + r:
                s.pos[0] = self.margin + r;          s.vel[0] =  abs(s.vel[0])*0.35; hit=True
            if s.pos[0] > self.W - self.margin - r:
                s.pos[0] = self.W - self.margin - r; s.vel[0] = -abs(s.vel[0])*0.35; hit=True
            if s.pos[1] < self.margin + r:
                s.pos[1] = self.margin + r;          s.vel[1] =  abs(s.vel[1])*0.35; hit=True
            if s.pos[1] > self.H - self.margin - r:
                s.pos[1] = self.H - self.margin - r; s.vel[1] = -abs(s.vel[1])*0.35; hit=True
            self.last_wall_flags[i] = hit

    def _update_nozzles(self):
        for i, s in enumerate(self.salps):
            vec    = _action_to_vector(self.last_actions[i])
            n      = math.hypot(*vec)
            target = math.atan2(vec[1], vec[0]) if n > 1e-8 else \
                     math.atan2(self.goal_pos[1]-s.pos[1], self.goal_pos[0]-s.pos[0])
            diff   = ((target - s.nozzle) + math.pi) % (2*math.pi) - math.pi
            s.nozzle += float(np.clip(diff * 0.22, -0.12, 0.12))

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, thrust_vectors):
        for i, s in enumerate(self.salps):
            tv         = thrust_vectors[i]
            thrusting  = math.hypot(tv[0], tv[1]) > 1e-6
            s.thrust_on = thrusting
            s.phase     = "exhaling" if thrusting else "rest"
            s.vel      += tv
            s.vel      *= _DRAG
            spd         = math.hypot(s.vel[0], s.vel[1])
            if spd > _MAX_SPEED:
                s.vel *= _MAX_SPEED / spd
            s.pos += s.vel
        self._apply_springs()
        for _ in range(3):
            self._enforce_separation()
            self._resolve_link_collisions()
            self._resolve_land_collisions()
            self._enforce_rigid_links()
        self._wall_bounce_all()
        self._update_nozzles()
        self.episode_steps += 1

    # ── Observation ───────────────────────────────────────────────────────────

    def get_agent_states(self):
        """Discretised state tuples for Q-table lookup."""
        states     = []
        gx, gy     = self.goal_pos
        gx_n, gy_n = gx / self.W, gy / self.H
        for s in self.salps:
            x, y         = s.pos
            dx, dy       = gx - x, gy - y
            dist_to_goal = math.hypot(dx, dy) / self.W
            goal_bin_val = _angle_bin(math.atan2(dy, dx))
            if self.land_polys:
                min_d, best_ang = self.W, 0.0
                pos_arr = np.array([x, y])
                for poly in self.land_polys:
                    cx, cy = _closest_on_poly(float(x), float(y), poly)
                    d = math.hypot(cx - x, cy - y)
                    if d < min_d:
                        min_d    = d
                        best_ang = math.atan2(cy-y, cx-x)
                d1, a1 = min_d, best_ang
            else:
                d1, a1 = self.W, 0.0
            states.append((
                int(x / self.W * 10), int(y / self.H * 10),
                int(gx_n * 10),       int(gy_n * 10),
                int(dist_to_goal * 10),
                goal_bin_val,
                int((d1 / self.W) * 10),
                _angle_bin(a1),
            ))
        return states

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward(self):
        positions   = np.array([s.pos for s in self.salps], dtype=float)
        local_dists = np.linalg.norm(positions - self.goal_pos, axis=1)
        min_d       = float(local_dists.min())
        if self._prev_local_dists is None:
            self._prev_local_dists = local_dists.copy()
        diag    = self.W + self.H
        rewards = []
        per_salp_log = []
        for i in range(_N_SALPS):
            s      = self.salps[i]
            prev_d = self._prev_local_dists[i]
            curr_d = local_dists[i]
            delta  = (prev_d - curr_d) / diag
            r      = delta * 10.0 - 0.5
            collision_hit = bool(self.last_collision_flags[i])
            wall_hit      = bool(self.last_wall_flags[i])
            if collision_hit:
                r -= 15.0
                self.salp_collision_counts[i] += 1
            if wall_hit:
                r -= 3.0
                self.salp_wall_counts[i] += 1
            act_vec = _action_to_vector(self.last_actions[i])
            rewards.append(r)
            per_salp_log.append({
                "salp": i, "reward_step": r,
                "collision": int(collision_hit), "wall_hit": int(wall_hit),
                "dist_to_goal": float(curr_d), "pos_x": float(s.pos[0]),
                "pos_y": float(s.pos[1]), "goal_x": float(self.goal_pos[0]),
                "goal_y": float(self.goal_pos[1]),
            })
        self._prev_local_dists = local_dists.copy()
        done = success = timeout = False
        if min_d < _GOAL_RADIUS:
            eff_bonus = 200.0 * (1.0 - self.episode_steps / self.max_steps)
            rewards   = [50.0 + eff_bonus] * _N_SALPS
            done = success = True
        elif self.episode_steps >= self.max_steps:
            rewards = [-50.0] * _N_SALPS
            done = timeout = True
        team_r = float(np.mean(rewards))
        return rewards, team_r, min_d, done, success, timeout, per_salp_log

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_counter += 1
        # Static mode: build once; Parallel mode: rebuild every episode
        if self.rebuild_every_episode:
            self._build_land()
        elif not self._land_built:
            self._build_land()
            self._land_built = True

        padding = self.margin + 100
        start   = np.array([padding, self.H - padding], dtype=float)
        for _ in range(200):
            heading = random.uniform(0, 2 * math.pi)
            ax, ay  = math.cos(heading), math.sin(heading)
            positions, valid = [], True
            for i in range(_N_SALPS):
                pos = np.array([start[0] + ax*i*_LINK_LEN,
                                start[1] + ay*i*_LINK_LEN], dtype=float)
                if not (self.margin <= pos[0] <= self.W - self.margin and
                        self.margin <= pos[1] <= self.H - self.margin):
                    valid = False; break
                if self._point_in_land(pos, buffer_px=28):
                    valid = False; break
                positions.append(pos)
            if valid:
                break
        else:
            s2        = 1.0 / math.sqrt(2)
            positions = [np.array([start[0]+s2*i*_LINK_LEN,
                                   start[1]-s2*i*_LINK_LEN]) for i in range(_N_SALPS)]

        for i, s in enumerate(self.salps):
            s.reset(positions[i])
        self.goal_pos              = self._rand_goal()
        self.episode_steps         = 0
        self.total_reward          = 0.0
        self._prev_local_dists     = None
        self.last_collision_flags  = [False] * _N_SALPS
        self.last_wall_flags       = [False] * _N_SALPS
        self.last_actions          = [0]     * _N_SALPS
        self.salp_collision_counts[:] = 0
        self.salp_wall_counts[:]      = 0
        self.salp_reward_totals[:]    = 0.0

# =============================================================================
# FAST ENVIRONMENT  (numpy-array based — used by MADDPG and MAPPO)
# =============================================================================

class SalpFastEnv:
    """
    Args:
        rebuild_every_episode: False → static obstacles (static variants)
                               True  → rebuild each reset (parallel/dynamic variants)
        headless: display flag (added — was missing from original MADDPG/MAPPO envs)
    """
    RADII      = np.array([14., 16., 18., 16., 14.], dtype=np.float32)
    SEMI_A     = RADII * 1.3
    SEMI_B     = RADII * 0.8
    MAX_EXTENT = np.maximum(SEMI_A, SEMI_B)

    def __init__(self, W=_W, H=_H, margin=_MARGIN,
                 worker_id=0, env_idx=0,
                 headless=True, rebuild_every_episode=False):
        self.W, self.H, self.margin   = W, H, margin
        self.worker_id                = worker_id
        self.env_idx                  = env_idx
        self.headless                 = headless
        self.rebuild_every_episode    = rebuild_every_episode

        self.pos = np.zeros((_N_SALPS, 2), dtype=np.float32)
        self.vel = np.zeros((_N_SALPS, 2), dtype=np.float32)

        self.land_polys:      List[np.ndarray]              = []
        self._poly_edges_a:   List[np.ndarray]              = []
        self._poly_edges_ab:  List[np.ndarray]              = []
        self._poly_edges_ab2: List[np.ndarray]              = []
        self._poly_bbox:      List[Tuple[float,float,float,float]] = []
        self._land_built    = False
        self._reset_counter = 0

        self.goal_pos            = np.zeros(2, dtype=np.float32)
        self.episode_steps       = 0
        self.max_steps           = 700
        self.total_reward        = 0.0
        self._prev_dists         = None
        self.collision_flags     = np.zeros(_N_SALPS, dtype=bool)
        self.wall_flags          = np.zeros(_N_SALPS, dtype=bool)
        self.last_actions        = np.zeros((_N_SALPS, 2), dtype=np.float32)
        self.episodes_completed  = 0
        self.salp_collision_counts = np.zeros(_N_SALPS, dtype=np.int32)
        self.salp_wall_counts      = np.zeros(_N_SALPS, dtype=np.int32)
        self.salp_reward_totals    = np.zeros(_N_SALPS, dtype=np.float64)

    # ── Obstacles ─────────────────────────────────────────────────────────────

    def _build_land(self):
        m     = self.margin + 120
        specs = [(55,18),(70,20),(78,22),(48,16),(42,14),(52,16)]
        placed, polys = [], []
        scale = random.uniform(_OBS_MIN, _OBS_MAX)
        max_r = min(self.W, self.H) // 6

        for r_base, pts in specs:
            r = min(int(r_base * scale), max_r)
            for _ in range(100):
                cx = random.uniform(m, self.W - m)
                cy = random.uniform(m, self.H - m)
                if not all(math.hypot(cx-ox,cy-oy) >= r+orad+60
                           for ox,oy,orad in placed):
                    continue
                raw     = _generate_land_polygon(cx, cy, r, points=pts)
                clamped = raw.copy()
                clamped[:, 0] = np.clip(raw[:, 0], m, self.W - m)
                clamped[:, 1] = np.clip(raw[:, 1], m, self.H - m)
                test_pts = np.array([
                    [self.W*0.25, cy],[self.W*0.5, cy],[self.W*0.75, cy],
                    [cx, self.H*0.25],[cx, self.H*0.5],[cx, self.H*0.75],
                ], dtype=np.float32)
                if any(_point_in_poly_xy(p[0], p[1], clamped) for p in test_pts):
                    continue
                placed.append((cx, cy, r))
                polys.append(clamped)
                break
        self.land_polys = polys
        # Precompute per-polygon edge arrays for fast queries
        self._poly_edges_a, self._poly_edges_ab = [], []
        self._poly_edges_ab2, self._poly_bbox   = [], []
        for poly in polys:
            a  = poly; b = np.roll(poly, -1, axis=0)
            ab = b - a
            ab2 = np.einsum("vi,vi->v", ab, ab)
            self._poly_edges_a.append(a)
            self._poly_edges_ab.append(ab)
            self._poly_edges_ab2.append(ab2)
            self._poly_bbox.append((
                float(poly[:, 0].min()), float(poly[:, 1].min()),
                float(poly[:, 0].max()), float(poly[:, 1].max()),
            ))

    def _closest_on_poly_fast(self, px: float, py: float, k: int) -> Tuple[float, float]:
        a   = self._poly_edges_a[k]
        ab  = self._poly_edges_ab[k]
        ab2 = self._poly_edges_ab2[k]
        apx = px - a[:, 0]; apy = py - a[:, 1]
        t   = np.clip((apx*ab[:, 0] + apy*ab[:, 1]) / (ab2 + 1e-12), 0.0, 1.0)
        cpx = a[:, 0] + t * ab[:, 0]
        cpy = a[:, 1] + t * ab[:, 1]
        best = int(np.argmin((cpx-px)**2 + (cpy-py)**2))
        return float(cpx[best]), float(cpy[best])

    def _point_in_land(self, px, py, buffer_px=0.0) -> bool:
        for k, poly in enumerate(self.land_polys):
            xmn, ymn, xmx, ymx = self._poly_bbox[k]
            pad = buffer_px
            if px < xmn-pad or px > xmx+pad or py < ymn-pad or py > ymx+pad:
                continue
            if _point_in_poly_xy(px, py, poly):
                return True
            if buffer_px > 0:
                cx, cy = self._closest_on_poly_fast(px, py, k)
                if math.hypot(cx-px, cy-py) <= buffer_px:
                    return True
        return False

    def _rand_goal(self) -> np.ndarray:
        inn = self.margin + 90
        for _ in range(1000):
            gx = random.uniform(inn, self.W - inn)
            gy = random.uniform(inn, self.H - inn)
            if not self._point_in_land(gx, gy, buffer_px=_GOAL_RADIUS + 20):
                return np.array([gx, gy], dtype=np.float32)
        return np.array([self.W * 0.85, self.H * 0.15], dtype=np.float32)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _apply_springs(self, iters=3):
        for _ in range(iters):
            delta = self.pos[1:] - self.pos[:-1]
            dist  = np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-6)
            unit  = delta / dist
            corr  = unit * ((dist - _LINK_LEN) * _SPRING_K * 0.5)
            self.pos[:-1] += corr; self.pos[1:] -= corr
            rel_v = self.vel[1:] - self.vel[:-1]
            damp  = (rel_v * unit).sum(axis=1, keepdims=True) * unit * 0.06
            self.vel[:-1] += damp; self.vel[1:] -= damp

    def _enforce_separation(self):
        for i in range(_N_SALPS):
            for j in range(i + 1, _N_SALPS):
                dx   = self.pos[j,0] - self.pos[i,0]
                dy   = self.pos[j,1] - self.pos[i,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-6: continue
                min_d = self.RADII[i] + self.RADII[j]
                if dist < min_d:
                    nx, ny     = dx/dist, dy/dist
                    correction = (min_d - dist) * 0.5
                    self.pos[i,0] -= nx*correction; self.pos[i,1] -= ny*correction
                    self.pos[j,0] += nx*correction; self.pos[j,1] += ny*correction

    def _resolve_land_collisions(self):
        self.collision_flags[:] = False
        for i in range(_N_SALPS):
            px, py = float(self.pos[i,0]), float(self.pos[i,1])
            for k, poly in enumerate(self.land_polys):
                xmn,ymn,xmx,ymx = self._poly_bbox[k]
                me = float(self.MAX_EXTENT[i])
                if px < xmn-me or px > xmx+me or py < ymn-me or py > ymx+me:
                    continue
                cx, cy = self._closest_on_poly_fast(px, py, k)
                delta  = np.array([px-cx, py-cy], dtype=np.float64)
                dist   = math.hypot(delta[0], delta[1])
                inside = _point_in_poly_xy(px, py, poly)
                min_sep = float(self.MAX_EXTENT[i]) + 2.0
                if inside or dist < min_sep:
                    if dist < 1e-8:
                        cent  = _polygon_centroid(poly).astype(np.float64)
                        delta = self.pos[i].astype(np.float64) - cent
                        dist  = math.hypot(delta[0], delta[1])
                        if dist < 1e-8:
                            delta = np.array([1.0, 0.0]); dist = 1.0
                    normal = delta / dist
                    self.pos[i]  = np.array([cx, cy], dtype=np.float32) + (normal * min_sep).astype(np.float32)
                    vn = np.dot(self.vel[i].astype(np.float64), normal)
                    if vn < 0:
                        self.vel[i] -= (1.55 * vn * normal).astype(np.float32)
                    self.vel[i] *= 0.90
                    self.collision_flags[i] = True

    def _resolve_link_collisions(self):
        for i in range(_N_SALPS - 1):
            p1, p2 = self.pos[i], self.pos[i+1]
            for k, poly in enumerate(self.land_polys):
                n = len(poly)
                for j in range(n):
                    q1, q2 = poly[j], poly[(j+1) % n]
                    if _segments_intersect(p1, p2, q1, q2):
                        edge   = q2 - q1
                        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
                        nl     = math.hypot(normal[0], normal[1]) + 1e-8
                        normal /= nl
                        mid    = (p1 + p2) * 0.5
                        if np.dot(mid - q1, normal) < 0:
                            normal *= -1
                        for idx in (i, i+1):
                            self.pos[idx] += normal * 6.0
                            vn = np.dot(self.vel[idx], normal)
                            if vn < 0:
                                self.vel[idx] -= vn * normal
                        break

    def _enforce_rigid_links(self):
        delta = self.pos[1:] - self.pos[:-1]
        dist  = np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-8)
        diff  = (dist - _LINK_LEN) / dist
        corr  = delta * 0.5 * diff
        self.pos[:-1] += corr; self.pos[1:] -= corr

    def _wall_bounce(self):
        self.wall_flags[:] = False
        lo = self.margin + self.MAX_EXTENT
        hi_x = self.W - self.margin - self.MAX_EXTENT
        hi_y = self.H - self.margin - self.MAX_EXTENT
        for i in range(_N_SALPS):
            hit = False
            if self.pos[i,0] < lo[i]:
                self.pos[i,0] = lo[i];    self.vel[i,0] =  abs(self.vel[i,0])*0.35; hit=True
            if self.pos[i,0] > hi_x[i]:
                self.pos[i,0] = hi_x[i];  self.vel[i,0] = -abs(self.vel[i,0])*0.35; hit=True
            if self.pos[i,1] < lo[i]:
                self.pos[i,1] = lo[i];    self.vel[i,1] =  abs(self.vel[i,1])*0.35; hit=True
            if self.pos[i,1] > hi_y[i]:
                self.pos[i,1] = hi_y[i];  self.vel[i,1] = -abs(self.vel[i,1])*0.35; hit=True
            self.wall_flags[i] = hit

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        """actions: (N_SALPS, 2) float array or list of 2-arrays"""
        acts = np.array(actions, dtype=np.float32)
        self.last_actions = acts.copy()
        self.vel += acts
        self.vel *= _DRAG
        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        mask   = speeds > _MAX_SPEED
        self.vel[mask[:, 0]] *= (_MAX_SPEED / speeds[mask[:, 0]])
        self.pos += self.vel
        self._apply_springs()
        for _ in range(3):
            self._enforce_separation()
            self._resolve_link_collisions()
            self._resolve_land_collisions()
            self._enforce_rigid_links()
        self._wall_bounce()
        self.episode_steps += 1

    # ── Observation ───────────────────────────────────────────────────────────

    def get_obs(self) -> List[np.ndarray]:
        """10-dim float observation per salp (compatible with MADDPG and MAPPO)."""
        gx, gy     = float(self.goal_pos[0]), float(self.goal_pos[1])
        gx_n, gy_n = gx / self.W, gy / self.H
        obs_list   = []
        for i in range(_N_SALPS):
            x, y         = float(self.pos[i,0]), float(self.pos[i,1])
            dx, dy       = gx - x, gy - y
            dist_to_goal = math.hypot(dx, dy) / self.W
            ang          = math.atan2(dy, dx)
            if self.land_polys:
                min_d, best_ang = self.W, 0.0
                for k in range(len(self.land_polys)):
                    xmn,ymn,xmx,ymx = self._poly_bbox[k]
                    bd = max(xmn-x,0.0,x-xmx)
                    by = max(ymn-y,0.0,y-ymx)
                    if math.hypot(bd, by) >= min_d:
                        continue
                    cx_, cy_ = self._closest_on_poly_fast(x, y, k)
                    d = math.hypot(cx_-x, cy_-y)
                    if d < min_d:
                        min_d, best_ang = d, math.atan2(cy_-y, cx_-x)
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

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward(self):
        diff        = self.pos - self.goal_pos
        local_dists = np.linalg.norm(diff, axis=1)
        min_d       = float(local_dists.min())
        if self._prev_dists is None:
            self._prev_dists = local_dists.copy()
        diag = self.W + self.H
        rewards, per_salp_log = [], []
        for i in range(_N_SALPS):
            prev_d = self._prev_dists[i]; curr_d = local_dists[i]
            r      = (prev_d - curr_d) / diag * 10.0 - 0.5
            c_hit  = bool(self.collision_flags[i])
            w_hit  = bool(self.wall_flags[i])
            if c_hit:
                r -= 15.0; self.salp_collision_counts[i] += 1
            if w_hit:
                r -= 3.0;  self.salp_wall_counts[i] += 1
            rewards.append(r)
            per_salp_log.append({
                "salp": i, "reward_step": r,
                "collision": int(c_hit), "wall_hit": int(w_hit),
                "dist_to_goal": float(curr_d),
                "pos_x": float(self.pos[i,0]), "pos_y": float(self.pos[i,1]),
                "goal_x": float(self.goal_pos[0]), "goal_y": float(self.goal_pos[1]),
            })
        self._prev_dists = local_dists.copy()
        done = success = timeout = False
        if min_d < _GOAL_RADIUS:
            eff_bonus = 200.0 * (1.0 - self.episode_steps / self.max_steps)
            rewards   = [50.0 + eff_bonus] * _N_SALPS
            done = success = True
        elif self.episode_steps >= self.max_steps:
            rewards = [-50.0] * _N_SALPS
            done = timeout = True
        rewards    = [float(r) for r in rewards]
        team_r     = float(np.mean(rewards))
        for i, r in enumerate(rewards):
            self.salp_reward_totals[i] += r
            per_salp_log[i].update({
                "reward_step": r, "team_reward": team_r,
                "success": int(success), "timeout": int(timeout),
                "min_dist_team": min_d,
            })
        return rewards, team_r, min_d, done, success, timeout, per_salp_log

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_counter += 1
        if self.rebuild_every_episode:
            self._build_land()
        elif not self._land_built:
            self._build_land()
            self._land_built = True

        pad   = self.margin + 100
        start = np.array([pad, self.H - pad], dtype=np.float32)
        for _ in range(200):
            heading = random.uniform(0, 2 * math.pi)
            ax, ay  = math.cos(heading), math.sin(heading)
            positions, valid = [], True
            for i in range(_N_SALPS):
                pos = start + np.array([ax*i*_LINK_LEN, ay*i*_LINK_LEN], dtype=np.float32)
                if not (self.margin <= pos[0] <= self.W - self.margin and
                        self.margin <= pos[1] <= self.H - self.margin):
                    valid = False; break
                if self._point_in_land(float(pos[0]), float(pos[1]), buffer_px=28):
                    valid = False; break
                positions.append(pos)
            if valid:
                break
        else:
            s2        = 1.0 / math.sqrt(2)
            positions = [start + np.array([s2*i*_LINK_LEN, -s2*i*_LINK_LEN])
                         for i in range(_N_SALPS)]

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
        self.salp_collision_counts[:] = 0
        self.salp_wall_counts[:]      = 0
        self.salp_reward_totals[:]    = 0.0
    
class PygameRenderer:
    COLOURS = [
        (100, 165, 230), (80, 205, 165), (210, 160, 100),
        (195, 115, 175), (165, 115, 215),
    ]

    def __init__(self, env, algo_name=""):
        pygame.init()
        self.env       = env
        self.algo_name = algo_name
        self.screen    = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption(f"Salp Chain — {algo_name}" if algo_name else "Salp Chain Demo")
        pygame.display.flip()
        self.clock     = pygame.time.Clock()
        self.font      = pygame.font.Font(None, 24)
        self.font_hud  = pygame.font.Font(None, 18)
        self.font_name = pygame.font.SysFont("Arial", 32, bold=True)
        self._flash    = 0
        self._flash_dur = 22

    def _get_salp_props(self, i):
        """Return (pos, vel, semi_a, semi_b, max_extent, nozzle, thrust_on) for salp i,
        compatible with both IQLEnv (salps list) and SalpFastEnv (pos/vel arrays)."""
        if hasattr(self.env, 'salps'):
            s = self.env.salps[i]
            return s.pos, s.vel, s.semi_a, s.semi_b, s.max_extent, s.nozzle, s.thrust_on
        else:
            pos        = self.env.pos[i]
            vel        = self.env.vel[i]
            semi_a     = float(SalpFastEnv.SEMI_A[i])
            semi_b     = float(SalpFastEnv.SEMI_B[i])
            max_extent = float(SalpFastEnv.MAX_EXTENT[i])
            nozzle     = float(math.atan2(vel[1], vel[0])) if math.hypot(vel[0], vel[1]) > 0.3 else 0.0
            thrust_on  = math.hypot(vel[0], vel[1]) > 0.1
            return pos, vel, semi_a, semi_b, max_extent, nozzle, thrust_on

    def draw_land(self):
        for poly in self.env.land_polys:
            pts = [(int(x), int(y)) for x, y in poly]
            pygame.draw.polygon(self.screen, (76, 118, 66), pts)
            pygame.draw.polygon(self.screen, (165, 192, 140), pts, 4)

    def draw_goal(self):
        gx, gy = int(self.env.goal_pos[0]), int(self.env.goal_pos[1])
        pygame.draw.circle(self.screen, (160, 135, 20), (gx, gy), _GOAL_RADIUS, 1)
        pts = [(gx + math.cos(math.pi/2 + k*math.pi/5) * (18 if k%2==0 else 8),
                gy - math.sin(math.pi/2 + k*math.pi/5) * (18 if k%2==0 else 8))
               for k in range(10)]
        pygame.draw.polygon(self.screen, (255, 210, 0), pts)

    def draw_links(self):
        for i in range(_N_SALPS - 1):
            pos_a = self._get_salp_props(i)[0]
            pos_b = self._get_salp_props(i + 1)[0]
            d     = math.hypot(pos_b[0]-pos_a[0], pos_b[1]-pos_a[1])
            st    = min(1.0, abs(d - _LINK_LEN) / _LINK_LEN)
            lc    = (int(50 + 180*st), int(160 - 120*st), 180)
            pygame.draw.line(self.screen, lc,
                             (int(pos_a[0]), int(pos_a[1])),
                             (int(pos_b[0]), int(pos_b[1])), 5)

    def draw_salp(self, i, colour):
        pos, vel, semi_a, semi_b, max_extent, nozzle, thrust_on = self._get_salp_props(i)
        rx, ry  = int(pos[0]), int(pos[1])
        spd     = math.hypot(vel[0], vel[1])
        body_a  = math.atan2(vel[1], vel[0]) if spd > 0.3 else nozzle
        ew, eh  = int(semi_a * 2), int(semi_b * 2)
        if ew > 1 and eh > 1:
            surf  = pygame.Surface((ew, eh), pygame.SRCALPHA)
            alpha = 225 if thrust_on else 170
            pygame.draw.ellipse(surf, colour + (alpha,), (0, 0, ew, eh))
            rot   = pygame.transform.rotate(surf, -math.degrees(body_a))
            self.screen.blit(rot, rot.get_rect(center=(rx, ry)))
        darker = tuple(max(0, c - 45) for c in colour)
        pygame.draw.circle(self.screen, darker, (rx, ry), int(max_extent), 2)

    def draw_hud(self):
        positions = (np.array([self.env.salps[i].pos for i in range(_N_SALPS)])
                     if hasattr(self.env, 'salps') else self.env.pos)
        cent = positions.mean(axis=0)
        dist = math.hypot(cent[0] - self.env.goal_pos[0],
                          cent[1] - self.env.goal_pos[1])
        vels = (np.array([self.env.salps[i].vel for i in range(_N_SALPS)])
                if hasattr(self.env, 'salps') else self.env.vel)
        speed = float(np.linalg.norm(vels, axis=1).mean())
        step  = getattr(self.env, 'episode_steps', 0)
        txt   = self.font_hud.render(
            f"Step {step}   Dist {dist:.0f}px   Spd {speed:.2f}",
            True, (185, 185, 185))
        self.screen.blit(txt, (10, self.env.H - 24))

    def render(self):
        flash = getattr(self.env, '_flash', 0)
        flash_dur = getattr(self.env, '_flash_dur', 22)
        bg = (int(55 * flash / flash_dur), 5, 10) if flash else (8, 20, 45)
        self.screen.fill(bg)
        m = self.env.margin
        pygame.draw.rect(self.screen, (25, 55, 95),
                         (m, m, self.env.W - 2*m, self.env.H - 2*m), 3)
        self.draw_land()
        self.draw_goal()
        self.draw_links()
        for i in range(_N_SALPS):
            self.draw_salp(i, self.COLOURS[i])
        self.draw_hud()
        # Algo name overlay
        if self.algo_name:
            shadow = self.font_name.render(self.algo_name, True, (0, 0, 0))
            label  = self.font_name.render(self.algo_name, True, (255, 255, 255))
            hint   = self.font_hud.render("Close window to advance to next algorithm", True, (160, 160, 160))
            self.screen.blit(shadow, (22, 22))
            self.screen.blit(label,  (20, 20))
            self.screen.blit(hint,   (20, 58))
        pygame.display.flip()
        self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        # Hide the window but keep pygame alive for the next algorithm.
        # Full pygame.quit() is called once at the very end by run_demo().
        pygame.display.set_caption("")
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

# =============================================================================
# ENV FACTORY FUNCTIONS  — called by the load_* demo functions
# =============================================================================

def make_iql_static_env(headless=False) -> IQLEnv:
    """Static obstacles — built once, never rebuilt. (IQL static)"""
    env = IQLEnv(headless=headless, rebuild_every_episode=False)
    env.reset()
    return env


def make_iql_parallel_env(headless=False) -> IQLEnv:
    """Dynamic obstacles — rebuilt every episode. (IQL parallel)"""
    env = IQLEnv(headless=headless, rebuild_every_episode=True)
    env.reset()
    return env


def make_maddpg_static_env(headless=False) -> SalpFastEnv:
    """Static obstacles, numpy-array env for MADDPG static."""
    env = SalpFastEnv(headless=headless, rebuild_every_episode=False)
    env.reset()
    return env


def make_maddpg_parallel_env(headless=False) -> SalpFastEnv:
    """Dynamic obstacles, numpy-array env for MADDPG parallel."""
    env = SalpFastEnv(headless=headless, rebuild_every_episode=True)
    env.reset()
    return env


def make_mappo_static_env(headless=False) -> SalpFastEnv:
    """Static obstacles, numpy-array env for MAPPO static."""
    env = SalpFastEnv(headless=headless, rebuild_every_episode=False)
    env.reset()
    return env


def make_mappo_parallel_env(headless=False) -> SalpFastEnv:
    """Dynamic obstacles, numpy-array env for MAPPO parallel."""
    env = SalpFastEnv(headless=headless, rebuild_every_episode=True)
    env.reset()
    return env


# =============================================================================
# DEMO LOAD FUNCTIONS  — one per trained variant
# =============================================================================

def load_iql_static():
    """Run one episode with the IQL static Q-table."""
    env   = make_iql_static_env(headless=False)
    agent = IndependentQLearner()
    agent.load("pickled_models/q_tables_iql_static.pkl")
    renderer = PygameRenderer(env,"IQL Static Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        states  = env.get_agent_states()
        actions_idx = agent.act(states, train=False)
        thrust_vecs = [_action_to_vector(a) for a in actions_idx]
        env.last_actions = actions_idx
        env.step(thrust_vecs)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def load_iql_parallel():
    """Run one episode with the IQL parallel Q-table."""
    env   = make_iql_parallel_env(headless=False)
    agent = IndependentQLearner()
    agent.load("pickled_models/iql_parallel.pkl")
    renderer = PygameRenderer(env,"IQL Dynamic Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        states  = env.get_agent_states()
        actions_idx = agent.act(states, train=False)
        thrust_vecs = [_action_to_vector(a) for a in actions_idx]
        env.last_actions = actions_idx
        env.step(thrust_vecs)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def _load_maddpg_agent(pkl_meta_path, pt_path_fallback):
    """
    The meta_maddpg_*.pkl files are pickle dicts written by save_checkpoint()
    containing {"episodes_done": N, "log_data": {...}}.
    The actual network weights are in the sibling .pt checkpoint.
    Try the .pt file first; if absent fall back to loading the pkl directly
    with torch (in case it was saved with torch.save instead).
    """
    agent = MADDPG(total_envs=1)
    # First try the proper .pt weights file
    if os.path.exists(pt_path_fallback):
        agent.load(pt_path_fallback)
        return agent
    # Fall back: maybe the .pkl IS the weights (torch.save to .pkl)
    try:
        data = torch.load(pkl_meta_path, map_location=DEVICE, weights_only=False)
        if "actors" in data:
            def unwrap(m): return m._orig_mod if hasattr(m, "_orig_mod") else m
            for i in range(_N_SALPS):
                unwrap(agent.actors[i]).load_state_dict(data["actors"][i])
                unwrap(agent.critics[i]).load_state_dict(data["critics"][i])
                unwrap(agent.target_actors[i]).load_state_dict(data["target_actors"][i])
                unwrap(agent.target_critics[i]).load_state_dict(data["target_critics"][i])
                agent.actor_opts[i].load_state_dict(data["actor_opts"][i])
                agent.critic_opts[i].load_state_dict(data["critic_opts"][i])
            return agent
    except Exception:
        pass
    # Last resort: pkl is a meta dict — extract what we can
    with open(pkl_meta_path, "rb") as f:
        meta = pickle.load(f)
    if isinstance(meta, dict) and "actors" in meta:
        def unwrap(m): return m._orig_mod if hasattr(m, "_orig_mod") else m
        for i in range(_N_SALPS):
            unwrap(agent.actors[i]).load_state_dict(meta["actors"][i])
        print(f"  Loaded actor weights from meta pkl: {pkl_meta_path}")
    else:
        print(f"  WARNING: could not load weights from {pkl_meta_path}, running with random weights.")
    return agent


def load_maddpg_static():
    """Run one episode with MADDPG static model."""
    env   = make_maddpg_static_env(headless=False)
    agent = _load_maddpg_agent(
        "pickled_models/meta_maddpg_static.pkl",
        "pickled_models/maddpg_static_checkpoint.pt"
    )
    renderer = PygameRenderer(env, "MADDPG Static Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        obs     = env.get_obs()
        # act_batch expects list-of-envs; wrap single env obs
        actions_list = agent.act_batch([obs], explore=False)[0]
        env.last_actions = np.array(actions_list, dtype=np.float32)
        env.step(actions_list)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def load_maddpg_parallel():
    """Run one episode with MADDPG parallel (dynamic) model."""
    env   = make_maddpg_parallel_env(headless=False)
    agent = _load_maddpg_agent(
        "pickled_models/meta_maddpg_dynamic.pkl",
        "pickled_models/maddpg_dynamic_checkpoint.pt"
    )
    renderer = PygameRenderer(env, "MADDPG Dynamic Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        obs     = env.get_obs()
        actions_list = agent.act_batch([obs], explore=False)[0]
        env.last_actions = np.array(actions_list, dtype=np.float32)
        env.step(actions_list)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def load_mappo_static():
    """Run one episode with MAPPO static model."""
    env   = make_mappo_static_env(headless=False)
    agent = MAPPO()
    agent.load("pickled_models/mappo_checkpoint_mappo_static.pt")
    renderer = PygameRenderer(env, "MAPPO Static Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        obs     = env.get_obs()
        actions_list, _, _ = agent.act_batch([obs])
        actions_list = actions_list[0]   # unwrap single-env
        env.last_actions = np.array(actions_list, dtype=np.float32)
        env.step(actions_list)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def load_mappo_parallel():
    """Run one episode with MAPPO parallel (dynamic) model."""
    env   = make_mappo_parallel_env(headless=False)
    agent = MAPPO()
    agent.load("pickled_models/mappo_dynamic_checkpoint.pt")
    renderer = PygameRenderer(env, "MAPPO Dynamic Obstacles")
    running = True
    while running:
        running = renderer.handle_events()
        obs     = env.get_obs()
        actions_list, _, _ = agent.act_batch([obs])
        actions_list = actions_list[0]
        env.last_actions = np.array(actions_list, dtype=np.float32)
        env.step(actions_list)
        rewards, team_r, min_d, done, success, timeout, _ = env.reward()
        renderer.render()
        if done:
            break
    renderer.close()


def run_demo():
    """Runs all 6 trained variants sequentially for one episode each."""
    print("=== IQL Static ===");    load_iql_static()
    print("=== IQL Parallel ===");  load_iql_parallel()
    print("=== MADDPG Static ==="); load_maddpg_static()
    print("=== MADDPG Dynamic ===");load_maddpg_parallel()
    print("=== MAPPO Static ===");  load_mappo_static()
    print("=== MAPPO Dynamic ==="); load_mappo_parallel()

if __name__ == "__main__":
    run_demo()