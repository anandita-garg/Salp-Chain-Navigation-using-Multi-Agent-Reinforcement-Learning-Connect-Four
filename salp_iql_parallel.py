"""
Salp Chain IQL — PARALLEL EDITION  (fixed + full logging)
===========================================================

LOGGING ADDED:
  * timestep_log.csv  — one row per salp per env step
      episode, worker_id, env_idx, episode_step, global_step, salp,
      reward_step, team_reward, collision, wall_hit,
      dist_to_goal, prev_dist, dist_delta, min_dist_team,
      goal_x, goal_y, pos_x, pos_y, vel_x, vel_y, speed,
      action_idx, action_x, action_y, link_dist_next,
      success, timeout, episode_steps

  * episode_log.csv   — one row per completed episode
      episode, worker_id, env_idx, reward, success, timeout,
      collisions, steps, eps,
      salp0..4_reward, salp0..4_collisions, salp0..4_wall_hits
"""

from __future__ import annotations

import math
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Process, Queue as MPQueue
from typing import List, Tuple

import numpy as np
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
LINK_LEN    = 68
SPRING_K    = 1
DRAG        = 0.955
MAX_SPEED   = 10.0
THRUST_MAG  = 0.60
GOAL_RADIUS = 30

# Env geometry
W_DEFAULT, H_DEFAULT, MARGIN_DEFAULT = 1400, 800, 70
LAND_REBUILD_INTERVAL  = 1
OBSTACLE_SIZE_MIN, OBSTACLE_SIZE_MAX = 0.5, 1.5

# IQL
ACTION_DIRS = [
    (0.0,  0.0),
    (0.0, -1.0), (1.0, -1.0), (1.0, 0.0), (1.0,  1.0),
    (0.0,  1.0), (-1.0, 1.0), (-1.0, 0.0),(-1.0, -1.0),
]
ACTION_NAMES = ["coast", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
N_ACTIONS    = len(ACTION_DIRS)
ANGLE_BINS   = 8

LEARNING_RATE = 0.10
GAMMA         = 0.97
EPS_START     = 1.0
EPS_MIN       = 0.05
EPS_DECAY     = 0.995

# Training
MAX_EPISODES        = 10_000
CHECKPOINT_INTERVAL = 500

# Logging
TIMESTEP_LOG_FLUSH_INTERVAL = 5_000   # rows before flush to disk

# Paths
SAVE_DIR          = "salp_saves_iql_parallel"
os.makedirs(SAVE_DIR, exist_ok=True)
QTABLE_PATH       = os.path.join(SAVE_DIR, "q_tables.pkl")
LOG_PATH          = os.path.join(SAVE_DIR, "training_log.csv")
CHECKPOINT_PATH   = os.path.join(SAVE_DIR, "checkpoint.pkl")
EPISODE_LOG_PATH  = os.path.join(SAVE_DIR, "episode_log.csv")
TIMESTEP_LOG_PATH = os.path.join(SAVE_DIR, "timestep_log.csv")

print(f"Workers: {NUM_WORKERS}  |  Envs/worker: {ENVS_PER_WORKER}"
      f"  |  Total envs: {TOTAL_ENVS}")


# =============================================================================
# GEOMETRY
# =============================================================================

def point_in_poly(point, poly_arr: np.ndarray) -> bool:
    x, y   = float(point[0]), float(point[1])
    xi, yi = poly_arr[:, 0], poly_arr[:, 1]
    xj     = np.roll(xi, 1)
    yj     = np.roll(yi, 1)
    cond   = ((yi > y) != (yj > y))
    x_int  = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
    return bool(np.sum(cond & (x < x_int)) % 2 == 1)


def closest_point_on_polygon(point: np.ndarray, poly_arr: np.ndarray) -> np.ndarray:
    p   = point.astype(float)
    a   = poly_arr
    b   = np.roll(poly_arr, -1, axis=0)
    ab  = b - a
    ap  = p - a
    ab2 = np.sum(ab * ab, axis=1) + 1e-12
    t   = np.clip(np.sum(ap * ab, axis=1) / ab2, 0.0, 1.0)
    closest = a + ab * t[:, None]
    d2 = np.sum((closest - p) ** 2, axis=1)
    return closest[np.argmin(d2)]


def polygon_centroid(poly_arr: np.ndarray) -> np.ndarray:
    return poly_arr.mean(axis=0)


def generate_land_polygon(cx, cy, base_r, points=18, jitter=0.2):
    ao     = random.uniform(0, 2 * math.pi)
    angles = ao + np.arange(points) * (2 * math.pi / points)
    rs     = base_r * (1 + np.random.uniform(-jitter, jitter, points))
    xs     = cx + np.cos(angles) * rs
    ys     = cy + np.sin(angles) * rs
    return np.stack([xs, ys], axis=1).astype(np.float32)


def angle_bin(angle_rad, num_bins=ANGLE_BINS):
    wrapped = (angle_rad + math.pi) % (2 * math.pi)
    return int((wrapped / (2 * math.pi)) * num_bins) % num_bins


def action_to_vector(action_idx: int) -> np.ndarray:
    dx, dy = ACTION_DIRS[action_idx]
    vec    = np.array([dx, dy], dtype=float)
    n      = math.hypot(dx, dy)
    if n < 1e-9:
        return np.zeros(2, dtype=float)
    return vec * (THRUST_MAG / n)


def segments_intersect(p1, p2, q1, q2) -> bool:
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)


# =============================================================================
# SALP
# =============================================================================

class Salp:
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

class IndependentQLearner:
    def __init__(self, n_agents=N_SALPS, n_actions=N_ACTIONS,
                 alpha=LEARNING_RATE, gamma=GAMMA,
                 eps=EPS_START, eps_min=EPS_MIN, eps_decay=EPS_DECAY):
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

    def save(self, path=QTABLE_PATH):
        plain = [{k: v.copy() for k, v in qt.items()} for qt in self.q_tables]
        with open(path, "wb") as f:
            pickle.dump({"q_tables": plain, "eps": self.eps}, f)

    def load(self, path=QTABLE_PATH):
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
# ENVIRONMENT
# =============================================================================

class Env:
    def __init__(self, W=W_DEFAULT, H=H_DEFAULT, margin=MARGIN_DEFAULT,
                 worker_id=0, env_idx=0, headless=True):
        self.W, self.H = W, H
        self.margin    = margin
        self.worker_id = worker_id
        self.env_idx   = env_idx
        self.headless  = headless

        radii  = [14, 16, 18, 16, 14]
        cx     = W / 2 - (N_SALPS - 1) * LINK_LEN / 2
        self.salps = [Salp(radii[i], (cx + i * LINK_LEN, H / 2))
                      for i in range(N_SALPS)]

        self.land_polys     = []
        self._reset_counter = 0

        self.goal_pos             = np.zeros(2, dtype=float)
        self.episode_steps        = 0
        self.max_steps            = 700
        self.total_reward         = 0.0
        self._prev_local_dists    = None
        self.last_collision_flags = [False] * N_SALPS
        self.last_wall_flags      = [False] * N_SALPS
        self.last_actions         = [0]     * N_SALPS   # list of action ints
        self.episodes_completed   = 0

        # Per-salp episode accumulators — reset each episode
        self.salp_collision_counts = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_wall_counts      = np.zeros(N_SALPS, dtype=np.int32)
        self.salp_reward_totals    = np.zeros(N_SALPS, dtype=np.float64)

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
                if any(point_in_poly(p, clamped) for p in test_pts):
                    continue
                placed.append((cx, cy, r))
                polys.append(clamped)
                break
        self.land_polys = polys

    def _point_in_land(self, point, buffer_px=0.0) -> bool:
        p = np.asarray(point, dtype=float)
        for poly in self.land_polys:
            if point_in_poly(p, poly):
                return True
            if buffer_px > 0:
                cp = closest_point_on_polygon(p, poly)
                if math.hypot(*(p - cp)) <= buffer_px:
                    return True
        return False

    def _rand_goal(self) -> np.ndarray:
        inn = self.margin + 90
        for _ in range(1_000):
            g = np.array([random.uniform(inn, self.W - inn),
                          random.uniform(inn, self.H - inn)], dtype=float)
            if not self._point_in_land(g, buffer_px=GOAL_RADIUS + 20):
                return g
        return np.array([self.W * 0.85, self.H * 0.15], dtype=float)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _enforce_separation(self):
        for i in range(N_SALPS):
            for j in range(i + 1, N_SALPS):
                a, b     = self.salps[i], self.salps[j]
                dx, dy   = b.pos[0]-a.pos[0], b.pos[1]-a.pos[1]
                dist     = math.hypot(dx, dy)
                min_dist = a.radius + b.radius
                if dist < min_dist and dist > 1e-6:
                    nx, ny     = dx/dist, dy/dist
                    correction = (min_dist - dist) * 0.5
                    a.pos[0] -= nx*correction; a.pos[1] -= ny*correction
                    b.pos[0] += nx*correction; b.pos[1] += ny*correction

    def _apply_springs(self):
        for _ in range(4):
            for i in range(N_SALPS - 1):
                a, b  = self.salps[i], self.salps[i+1]
                delta = b.pos - a.pos
                dist  = math.hypot(delta[0], delta[1])
                if dist < 1e-6:
                    continue
                unit  = delta / dist
                corr  = unit * (dist - LINK_LEN) * SPRING_K * 0.5
                a.pos += corr; b.pos -= corr
                rel_v = b.vel - a.vel
                damp  = float(np.dot(rel_v, unit)) * unit * 0.06
                a.vel += damp; b.vel -= damp

    def _resolve_link_collisions(self):
        for i in range(N_SALPS - 1):
            a, b = self.salps[i], self.salps[i+1]
            for poly in self.land_polys:
                n = len(poly)
                for j in range(n):
                    p1, p2 = poly[j], poly[(j+1) % n]
                    if segments_intersect(a.pos, b.pos, p1, p2):
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
        for i in range(N_SALPS - 1):
            a, b = self.salps[i], self.salps[i+1]
            dx   = b.pos[0] - a.pos[0]; dy = b.pos[1] - a.pos[1]
            dist = math.hypot(dx, dy) + 1e-8
            diff = (dist - LINK_LEN) / dist
            cx_  = dx * 0.5 * diff; cy_ = dy * 0.5 * diff
            a.pos[0] += cx_; a.pos[1] += cy_
            b.pos[0] -= cx_; b.pos[1] -= cy_

    def _resolve_land_collisions(self):
        self.last_collision_flags = [False] * N_SALPS
        for i, s in enumerate(self.salps):
            px, py = float(s.pos[0]), float(s.pos[1])
            for poly in self.land_polys:
                cp     = closest_point_on_polygon(s.pos, poly)
                delta  = s.pos - cp
                dist   = math.hypot(delta[0], delta[1])
                inside = point_in_poly(s.pos, poly)
                min_sep = s.max_extent + 2.0
                if inside or dist < min_sep:
                    if dist < 1e-8:
                        cent  = polygon_centroid(poly)
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
        self.last_wall_flags = [False] * N_SALPS
        for i, s in enumerate(self.salps):
            r   = s.max_extent
            hit = False
            if s.pos[0] < self.margin + r:
                s.pos[0] = self.margin + r;          s.vel[0] =  abs(s.vel[0]) * 0.35; hit = True
            if s.pos[0] > self.W - self.margin - r:
                s.pos[0] = self.W - self.margin - r; s.vel[0] = -abs(s.vel[0]) * 0.35; hit = True
            if s.pos[1] < self.margin + r:
                s.pos[1] = self.margin + r;          s.vel[1] =  abs(s.vel[1]) * 0.35; hit = True
            if s.pos[1] > self.H - self.margin - r:
                s.pos[1] = self.H - self.margin - r; s.vel[1] = -abs(s.vel[1]) * 0.35; hit = True
            self.last_wall_flags[i] = hit

    def _update_nozzles(self):
        for i, s in enumerate(self.salps):
            vec    = action_to_vector(self.last_actions[i])
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
            s.vel      *= DRAG
            spd         = math.hypot(s.vel[0], s.vel[1])
            if spd > MAX_SPEED:
                s.vel *= MAX_SPEED / spd
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
            goal_bin_val = angle_bin(math.atan2(dy, dx))

            if self.land_polys:
                min_d, best_ang = self.W, 0.0
                pos_arr = np.array([x, y])
                for poly in self.land_polys:
                    cp = closest_point_on_polygon(pos_arr, poly)
                    d  = math.hypot(*(pos_arr - cp))
                    if d < min_d:
                        min_d    = d
                        best_ang = math.atan2(cp[1]-y, cp[0]-x)
                d1, a1 = min_d, best_ang
            else:
                d1, a1 = self.W, 0.0

            states.append((
                int(x / self.W * 10), int(y / self.H * 10),
                int(gx_n * 10),       int(gy_n * 10),
                int(dist_to_goal * 10),
                goal_bin_val,
                int((d1 / self.W) * 10),
                angle_bin(a1),
            ))
        return states

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward(self):
        """
        Returns: rewards, team_reward, min_dist, done, success, timeout, per_salp_log

        Reward components:
          - Smooth distance shaping: (prev_d - curr_d) / diag * 10
          - Time penalty:            -0.5 per step
          - Collision penalty:       -15
          - Wall hit penalty:        -3
          - Success terminal:        50 + 200*(1 - steps/max_steps)
          - Timeout terminal:        -50
        """
        # Build position array from salp objects (fixed vs FastEnv copy-paste)
        positions   = np.array([s.pos for s in self.salps], dtype=float)  # (N,2)
        local_dists = np.linalg.norm(positions - self.goal_pos, axis=1)   # (N,)
        min_d       = float(local_dists.min())

        if self._prev_local_dists is None:
            self._prev_local_dists = local_dists.copy()

        diag    = self.W + self.H
        rewards = []
        per_salp_log = []

        for i in range(N_SALPS):
            s      = self.salps[i]
            prev_d = self._prev_local_dists[i]
            curr_d = local_dists[i]

            # Smooth normalised progress reward
            delta = (prev_d - curr_d) / diag
            r     = delta * 10.0

            # Time penalty
            r -= 0.5

            collision_hit = bool(self.last_collision_flags[i])
            wall_hit      = bool(self.last_wall_flags[i])

            if collision_hit:
                r -= 15.0
                self.salp_collision_counts[i] += 1
            if wall_hit:
                r -= 3.0
                self.salp_wall_counts[i] += 1

            # Resolved action vector for logging
            act_vec = action_to_vector(self.last_actions[i])

            rewards.append(r)
            per_salp_log.append({
                "salp":         i,
                "reward_step":  r,
                "collision":    int(collision_hit),
                "wall_hit":     int(wall_hit),
                "dist_to_goal": float(curr_d),
                "prev_dist":    float(prev_d),
                "dist_delta":   float(prev_d - curr_d),
                "pos_x":        float(s.pos[0]),
                "pos_y":        float(s.pos[1]),
                "vel_x":        float(s.vel[0]),
                "vel_y":        float(s.vel[1]),
                "speed":        float(math.hypot(s.vel[0], s.vel[1])),
                "action_idx":   int(self.last_actions[i]),
                "action_x":     float(act_vec[0]),
                "action_y":     float(act_vec[1]),
                "goal_x":       float(self.goal_pos[0]),
                "goal_y":       float(self.goal_pos[1]),
            })

        self._prev_local_dists = local_dists.copy()

        done = success = timeout = False

        if min_d < GOAL_RADIUS:
            efficiency_bonus = 200.0 * (1.0 - self.episode_steps / self.max_steps)
            terminal_r       = 50.0 + efficiency_bonus
            rewards          = [terminal_r] * N_SALPS
            for entry in per_salp_log:
                entry["reward_step"] = terminal_r
            done = success = True

        elif self.episode_steps >= self.max_steps:
            rewards = [-50.0] * N_SALPS
            for entry in per_salp_log:
                entry["reward_step"] = -50.0
            done    = True
            timeout = True

        # Link distances for chain geometry logging
        link_dists = [
            float(np.linalg.norm(self.salps[i+1].pos - self.salps[i].pos))
            for i in range(N_SALPS - 1)
        ]

        rewards     = [float(r) for r in rewards]
        team_reward = float(np.mean(rewards))

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

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_counter += 1
        if self._reset_counter % LAND_REBUILD_INTERVAL == 0 or not self.land_polys:
            self._build_land()

        padding = self.margin + 100
        start   = np.array([padding, self.H - padding], dtype=float)

        for _ in range(200):
            heading  = random.uniform(0, 2 * math.pi)
            ax, ay   = math.cos(heading), math.sin(heading)
            positions, valid = [], True
            for i in range(N_SALPS):
                pos = np.array([start[0] + ax*i*LINK_LEN,
                                start[1] + ay*i*LINK_LEN], dtype=float)
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
            positions = [np.array([start[0]+s2*i*LINK_LEN,
                                   start[1]-s2*i*LINK_LEN]) for i in range(N_SALPS)]

        for i, s in enumerate(self.salps):
            s.reset(positions[i])

        self.goal_pos              = self._rand_goal()
        self.episode_steps         = 0
        self.total_reward          = 0.0
        self._prev_local_dists     = None
        self.last_collision_flags  = [False] * N_SALPS
        self.last_wall_flags       = [False] * N_SALPS
        self.last_actions          = [0]     * N_SALPS

        # Reset per-salp accumulators (was missing in original)
        self.salp_collision_counts[:] = 0
        self.salp_wall_counts[:]      = 0
        self.salp_reward_totals[:]    = 0.0


# =============================================================================
# WORKER PROCESS
# =============================================================================

def worker_process(worker_id: int, n_envs: int,
                   action_q: MPQueue, result_q: MPQueue):
    """
    Runs n_envs Env instances in a tight loop.

    Protocol:
      1. Startup: send n_envs initial state dicts.
      2. Loop: receive (n_envs, N_SALPS) action lists, step all envs, send results.
      3. On episode end: reset env, send terminal state + fresh start state separately.
      4. None message → exit.
    """
    envs = [Env(worker_id=worker_id, env_idx=e, headless=True)
            for e in range(n_envs)]
    for env in envs:
        env.reset()

    for env_idx, env in enumerate(envs):
        initial_states = env.get_agent_states()
        result_q.put({
            "worker_id":               worker_id,
            "env_idx":                 env_idx,
            "states":                  initial_states,
            "next_start_states":       initial_states,
            "rewards":                 [0.0] * N_SALPS,
            "team_reward":             0.0,
            "dist":                    float("inf"),
            "done":                    False,
            "success":                 False,
            "timeout":                 False,
            "total_reward":            0.0,
            "episode_steps":           0,
            "collision_sum":           0,
            "per_salp_log":            [],
            "salp_collision_counts":   [0] * N_SALPS,
            "salp_wall_counts":        [0] * N_SALPS,
            "salp_reward_totals":      [0.0] * N_SALPS,
        })

    global_step = 0

    while True:
        msg = action_q.get()
        if msg is None:
            break

        actions_batch = msg   # list of n_envs action-index lists

        for env_idx, env in enumerate(envs):
            actions = actions_batch[env_idx]        # list of N_SALPS ints
            thrusts = [action_to_vector(a) for a in actions]
            env.last_actions = actions[:]
            env.step(thrusts)
            global_step += 1

            rewards, team_r, dist, done, success, timeout, per_salp_log = env.reward()
            env.total_reward += team_r

            # Attach context needed for logging
            for entry in per_salp_log:
                entry["worker_id"]    = worker_id
                entry["env_idx"]      = env_idx
                entry["episode_step"] = env.episode_steps
                entry["global_step"]  = global_step

            terminal_states = env.get_agent_states()
            total_reward    = env.total_reward
            episode_steps   = env.episode_steps
            collision_sum   = sum(env.last_collision_flags)
            sc              = env.salp_collision_counts.tolist()
            sw              = env.salp_wall_counts.tolist()
            sr              = env.salp_reward_totals.tolist()

            if done:
                env.episodes_completed += 1
                env.reset()
                next_start_states = env.get_agent_states()
            else:
                next_start_states = terminal_states

            result_q.put({
                "worker_id":               worker_id,
                "env_idx":                 env_idx,
                "states":                  terminal_states,
                "next_start_states":       next_start_states,
                "rewards":                 rewards,
                "team_reward":             team_r,
                "dist":                    dist,
                "done":                    done,
                "success":                 success,
                "timeout":                 timeout,
                "total_reward":            total_reward,
                "episode_steps":           episode_steps,
                "collision_sum":           collision_sum,
                "per_salp_log":            per_salp_log,
                "salp_collision_counts":   sc,
                "salp_wall_counts":        sw,
                "salp_reward_totals":      sr,
            })


# =============================================================================
# LOGGING
# =============================================================================

class EpisodeLogger:
    """Appends one row per completed episode to episode_log.csv."""

    COLS = [
        "episode", "worker_id", "env_idx",
        "reward", "success", "timeout", "collisions", "steps", "eps",
        # per-salp
        "salp0_reward","salp1_reward","salp2_reward","salp3_reward","salp4_reward",
        "salp0_collisions","salp1_collisions","salp2_collisions",
        "salp3_collisions","salp4_collisions",
        "salp0_wall_hits","salp1_wall_hits","salp2_wall_hits",
        "salp3_wall_hits","salp4_wall_hits",
    ]

    def __init__(self, path=EPISODE_LOG_PATH):
        self.path    = path
        self._buffer: List[dict] = []
        if not os.path.exists(path):
            pd.DataFrame(columns=self.COLS).to_csv(path, index=False)

    def log(self, episode_idx: int, r: dict, eps: float):
        row = {
            "episode":    episode_idx,
            "worker_id":  r["worker_id"],
            "env_idx":    r["env_idx"],
            "reward":     r["total_reward"],
            "success":    int(r["success"]),
            "timeout":    int(r["timeout"]),
            "collisions": r["collision_sum"],
            "steps":      r["episode_steps"],
            "eps":        eps,
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
    """Appends one row per salp per env step to timestep_log.csv."""

    COLS = [
        # identifiers
        "episode", "worker_id", "env_idx", "episode_step", "global_step", "salp",
        # reward
        "reward_step", "team_reward",
        # events
        "collision", "wall_hit",
        # goal proximity
        "dist_to_goal", "prev_dist", "dist_delta", "min_dist_team",
        "goal_x", "goal_y",
        # motion
        "pos_x", "pos_y", "vel_x", "vel_y", "speed",
        # action
        "action_idx", "action_x", "action_y",
        # chain
        "link_dist_next",
        # terminal flags (non-zero only on terminal step)
        "success", "timeout", "episode_steps",
    ]

    def __init__(self, path=TIMESTEP_LOG_PATH,
                 flush_every=TIMESTEP_LOG_FLUSH_INTERVAL):
        self.path        = path
        self.flush_every = flush_every
        self._buffer: List[dict] = []
        self._count  = 0
        if not os.path.exists(path):
            pd.DataFrame(columns=self.COLS).to_csv(path, index=False)

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
# CHECKPOINT
# =============================================================================

def save_checkpoint(agent, log_data, episodes_done):
    plain = [{k: v.copy() for k, v in qt.items()} for qt in agent.q_tables]
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({
            "q_tables":      plain,
            "eps":           agent.eps,
            "log_data":      log_data,
            "episodes_done": episodes_done,
        }, f)
    pd.DataFrame(log_data).to_csv(LOG_PATH, index=False)


def load_checkpoint(agent):
    if not os.path.exists(CHECKPOINT_PATH):
        return None, 0
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)
    na = agent.n_actions
    agent.q_tables = []
    for plain in data["q_tables"]:
        qt = defaultdict(lambda: np.zeros(na, dtype=float))
        qt.update(plain)
        agent.q_tables.append(qt)
    agent.eps = data["eps"]
    print(f"  Checkpoint loaded — resuming from episode {data['episodes_done']}"
          f"  eps={agent.eps:.4f}")
    return data["log_data"], data["episodes_done"]


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_and_plot(log_data):
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_PATH, index=False)
    print(f"\n===== SUMMARY =====")
    print(f"Episodes:       {len(df)}")
    print(f"Successes:      {df['success'].sum()}  ({100*df['success'].mean():.1f}%)")
    print(f"Timeouts:       {df['timeout'].sum()}")
    print(f"Avg collisions: {df['collisions'].mean():.2f}")
    ss = df[df["success"] == 1]["steps"]
    if len(ss):
        print(f"Avg steps/goal: {ss.mean():.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Salp IQL Parallel — {TOTAL_ENVS} envs ({NUM_WORKERS} workers)")
    w = 100
    df["reward"].rolling(w).mean().plot(    ax=axes[0,0], title="Reward (rolling)")
    df["success"].rolling(w).mean().plot(   ax=axes[0,1], title="Success Rate (rolling)")
    df["collisions"].rolling(w).mean().plot(ax=axes[1,0], title="Collisions (rolling)")
    df["steps"].rolling(w).mean().plot(     ax=axes[1,1], title="Steps (rolling)")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(out, dpi=100)
    print(f"Plot -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESUME = True

    log_data = {"episode": [], "reward": [], "success": [],
                "timeout": [], "collisions": [], "steps": []}

    agent    = IndependentQLearner()
    start_ep = 0

    if RESUME:
        loaded_log, start_ep = load_checkpoint(agent)
        if loaded_log:
            log_data = loaded_log

    ep_logger = EpisodeLogger(EPISODE_LOG_PATH)
    ts_logger = TimestepLogger(TIMESTEP_LOG_PATH)

    # ── Spawn workers ─────────────────────────────────────────────────────────
    action_qs = [MPQueue(maxsize=max(8, ENVS_PER_WORKER * 2))
                 for _ in range(NUM_WORKERS)]
    result_q  = MPQueue(maxsize=max(TOTAL_ENVS * 8, 32))

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

    # Collect initial states
    state_store = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
    for _ in range(TOTAL_ENVS):
        r = result_q.get()
        state_store[r["worker_id"]][r["env_idx"]] = r["next_start_states"]

    def flat_states():
        return [state_store[w][e]
                for w in range(NUM_WORKERS)
                for e in range(ENVS_PER_WORKER)]

    # Track which episode index each (worker, env) slot is currently on
    ep_idx_store = [[start_ep + w * ENVS_PER_WORKER + e
                     for e in range(ENVS_PER_WORKER)]
                    for w in range(NUM_WORKERS)]

    episodes_done = start_ep
    success_count = int(sum(log_data.get("success", []))) if log_data else 0
    decay_count   = start_ep

    last_states = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]
    last_acts   = [[None] * ENVS_PER_WORKER for _ in range(NUM_WORKERS)]

    pbar = tqdm(
        total=MAX_EPISODES,
        initial=start_ep,
        desc=f"Training ({TOTAL_ENVS} envs)",
        dynamic_ncols=True,
    )

    try:
        while episodes_done < MAX_EPISODES:

            # ── Choose actions ────────────────────────────────────────────────
            all_states = flat_states()
            all_acts   = [agent.act(states, train=True) for states in all_states]

            for w in range(NUM_WORKERS):
                worker_acts = []
                for e in range(ENVS_PER_WORKER):
                    flat_idx         = w * ENVS_PER_WORKER + e
                    last_states[w][e] = state_store[w][e]
                    last_acts[w][e]   = all_acts[flat_idx]
                    worker_acts.append(all_acts[flat_idx])
                action_qs[w].put(worker_acts)

            # ── Collect results ───────────────────────────────────────────────
            transitions    = []
            episode_events = []

            for _ in range(TOTAL_ENVS):
                r    = result_q.get()
                w, e = r["worker_id"], r["env_idx"]

                prev_states      = last_states[w][e]
                actions          = last_acts[w][e]
                next_states      = r["states"]
                live_next_states = r["next_start_states"]

                if prev_states is not None and actions is not None:
                    transitions.append((
                        prev_states, actions, r["rewards"],
                        next_states, [r["done"]] * N_SALPS,
                    ))

                # Timestep logging
                current_ep = ep_idx_store[w][e]
                ts_logger.log_step(current_ep, r["per_salp_log"])

                state_store[w][e] = live_next_states
                last_states[w][e] = None
                last_acts[w][e]   = None

                if r["done"]:
                    episode_events.append((w, e, r))

            # ── Q-table updates ───────────────────────────────────────────────
            for prev_states, actions, rewards, next_states, dones in transitions:
                agent.learn(prev_states, actions, rewards, next_states, dones)

            # ── Process completed episodes ────────────────────────────────────
            for w, e, r in episode_events:
                if episodes_done >= MAX_EPISODES:
                    break

                success = r["success"]
                if success:
                    success_count += 1

                # Episode logger (detailed CSV)
                ep_logger.log(episodes_done, r, agent.eps)

                # Legacy log_data dict (for backward-compat plotting/checkpoint)
                log_data["episode"].append(episodes_done)
                log_data["reward"].append(r["total_reward"])
                log_data["success"].append(int(success))
                log_data["timeout"].append(int(r.get("timeout", False)))
                log_data["collisions"].append(r["collision_sum"])
                log_data["steps"].append(r["episode_steps"])

                ep_idx_store[w][e] = episodes_done
                episodes_done += 1
                pbar.update(1)

                if decay_count < episodes_done:
                    agent.decay()
                    decay_count += 1

                pbar.set_postfix(
                    eps=f"{agent.eps:.3f}",
                    goals=success_count,
                    steps=r["episode_steps"],
                )

                if episodes_done % CHECKPOINT_INTERVAL == 0:
                    ep_logger.flush()
                    ts_logger.flush()
                    save_checkpoint(agent, log_data, episodes_done)
                    tqdm.write(f"  Checkpoint @ ep {episodes_done}")

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
        save_checkpoint(agent, log_data, episodes_done)
        agent.save()
        print("All saved.")
        analyze_and_plot(log_data)


if __name__ == "__main__":
    main()