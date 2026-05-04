"""
demo1.py  —  Live Demo with Live Visualization
================================================
Runs 1 episode per algorithm with live matplotlib visualization showing:
- Salps moving through the environment
- Obstacles
- Goal position
- Real-time trajectory

Run from repository root:
    python demo1.py
"""

import importlib.util
import math
import os
import pickle
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")  # Interactive backend for live display
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.animation import FuncAnimation

REPO = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Terminal colors
GRN = "\033[92m"
RED = "\033[91m"
YLW = "\033[93m"
CYN = "\033[96m"
BLD = "\033[1m"
RST = "\033[0m"


def load_module(path, name):
    """Load a Python module from file without executing __main__ blocks."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def visualize_live(env, env_mod, get_actions_fn, algo_name):
    """Live visualization using matplotlib animation."""
    print(f"\n{BLD}{CYN}Running {algo_name}...{RST}")
    
    env.reset()
    
    # Get initial observation
    if hasattr(env, 'get_obs'):
        obs = env.get_obs()
    elif hasattr(env, 'get_agent_states'):
        obs = env.get_agent_states()
    else:
        obs = None
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, env.W)
    ax.set_ylim(0, env.H)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"{algo_name} - Live Episode", fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Draw obstacles
    if hasattr(env, 'land_polys') and env.land_polys:
        for poly in env.land_polys:
            if isinstance(poly, np.ndarray) and len(poly) > 0:
                poly_patch = Polygon(poly, closed=True, alpha=0.3, 
                                    facecolor='gray', edgecolor='black', linewidth=1)
                ax.add_patch(poly_patch)
    
    # Draw goal
    goal_circle = Circle(env.goal_pos, env_mod.GOAL_RADIUS, 
                         alpha=0.5, facecolor='green', edgecolor='darkgreen', 
                         linewidth=2, label='Goal')
    ax.add_patch(goal_circle)
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'g*', markersize=20, label='Goal Center')
    
    # Salp colors
    colors = plt.cm.Set3(np.linspace(0, 1, env_mod.N_SALPS))
    
    # Initialize scatter plots for salps
    salp_scatters = []
    for i in range(env_mod.N_SALPS):
        scatter = ax.scatter([], [], s=150, color=colors[i], marker='o', 
                            label=f'Salp {i}', zorder=10, edgecolors='black', linewidth=1)
        salp_scatters.append(scatter)
    
    # Initialize line plots for trajectories
    trajectory_lines = []
    for i in range(env_mod.N_SALPS):
        line, = ax.plot([], [], color=colors[i], alpha=0.4, linewidth=1.5)
        trajectory_lines.append(line)
    
    # Initialize link line (connections between salps)
    link_line, = ax.plot([], [], 'k--', alpha=0.5, linewidth=1.5, label='Links')
    
    # Storage for trajectories
    trajectories = [[] for _ in range(env_mod.N_SALPS)]
    
    # Metrics display
    metrics_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Episode state
    episode_data = {
        'steps': 0,
        'total_reward': 0.0,
        'min_dist': float('inf'),
        'done': False,
        'obs': obs,
    }
    
    def get_salp_positions():
        """Extract salp positions from environment."""
        if hasattr(env, 'pos'):  # FastEnv (MADDPG/MAPPO)
            return env.pos.copy()
        elif hasattr(env, 'salps'):  # Env (IQL)
            return np.array([s.pos for s in env.salps], dtype=np.float32)
        return None
    
    def update_frame(frame):
        if episode_data['done']:
            return
        
        # Get actions
        actions = get_actions_fn(episode_data['obs'])
        
        # Step environment
        env.step(actions)
        episode_data['steps'] += 1
        
        # Get next observation
        if hasattr(env, 'get_obs'):
            episode_data['obs'] = env.get_obs()
        elif hasattr(env, 'get_agent_states'):
            episode_data['obs'] = env.get_agent_states()
        
        # Compute rewards
        if hasattr(env, 'reward'):
            rewards, team_reward, step_min_dist, done, success, timeout, per_salp_log = env.reward()
            episode_data['total_reward'] += team_reward
            episode_data['min_dist'] = min(episode_data['min_dist'], step_min_dist)
            episode_data['done'] = done
        
        # Get current positions
        positions = get_salp_positions()
        
        if positions is not None:
            # Update salp positions
            for i in range(env_mod.N_SALPS):
                salp_scatters[i].set_offsets(positions[i:i+1])
                trajectories[i].append(positions[i].copy())
            
            # Update trajectories
            for i in range(env_mod.N_SALPS):
                if len(trajectories[i]) > 1:
                    traj = np.array(trajectories[i])
                    trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
            
            # Update links between salps
            if len(positions) > 1:
                link_x = [positions[i, 0] for i in range(len(positions))]
                link_y = [positions[i, 1] for i in range(len(positions))]
                link_line.set_data(link_x, link_y)
        
        # Update metrics text
        metrics = (
            f"Step: {episode_data['steps']:4d} | "
            f"Reward: {episode_data['total_reward']:8.1f} | "
            f"Min Dist: {episode_data['min_dist']:7.1f}"
        )
        metrics_text.set_text(metrics)
        
        # Check termination
        if hasattr(env, "done"):
            episode_data['done'] = env.done or episode_data['done']
        
        if episode_data['done'] or episode_data['steps'] >= 700:
            print(f"  {GRN}✓ Episode Complete!{RST}")
            print(f"    Steps: {episode_data['steps']} | "
                  f"Reward: {episode_data['total_reward']:.2f} | "
                  f"Min Dist: {episode_data['min_dist']:.2f} | "
                  f"Status: {'SUCCESS' if episode_data['min_dist'] < env_mod.GOAL_RADIUS else 'TIMEOUT'}")
    
    # Create animation
    anim = FuncAnimation(fig, update_frame, frames=1000, interval=50, repeat=False)
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# IQL AGENTS
# ═════════════════════════════════════════════════════════════════════════════

ANGLE_BINS = 8
ACTION_DIRS = [
    (0.0, 0.0),
    (0.0,-1.0),(1.0,-1.0),(1.0,0.0),(1.0,1.0),
    (0.0, 1.0),(-1.0,1.0),(-1.0,0.0),(-1.0,-1.0),
]


def demo_iql(label: str, script_path: str, checkpoint_path: str):
    """Run IQL demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        safe_name = label.replace(" ", "_").replace("-", "_")
        env_mod = load_module(script_path, f"iql_{safe_name}")
        
        # Create env with headless=False for visualization
        env = env_mod.Env(headless=False) if hasattr(env_mod, "Env") else env_mod.FastEnv()
        
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        
        if isinstance(ckpt, dict):
            q_tables = ckpt.get("q_tables", ckpt.get("qtables", ckpt))
        else:
            q_tables = ckpt
        
        def iql_get_obs_key(obs: np.ndarray) -> tuple:
            """Discretise observation into hashable key."""
            x_bin   = int(obs[0] * 5)
            y_bin   = int(obs[1] * 5)
            ang     = math.atan2(float(obs[5]), float(obs[6]))
            ang_bin = int((ang + math.pi) / (2 * math.pi) * ANGLE_BINS) % ANGLE_BINS
            dist_b  = min(int(obs[4] * 5), 4)
            obs_b   = min(int(obs[7] * 5), 4)
            return (x_bin, y_bin, ang_bin, dist_b, obs_b)
        
        def get_actions(obs_list):
            actions = []
            for i in range(env_mod.N_SALPS):
                obs = obs_list[i]
                if isinstance(obs, (np.ndarray, tuple, list)) and len(obs) >= 7:
                    if isinstance(obs, np.ndarray):
                        key = iql_get_obs_key(obs)
                    else:
                        key = obs  # Already discretized
                    q = q_tables[i].get(key, [0.0] * len(ACTION_DIRS))
                    a = int(np.argmax(q))
                    actions.append(list(ACTION_DIRS[a]))
                else:
                    actions.append([0.0, 0.0])
            return actions
        
        visualize_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")
        import traceback
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# MADDPG AGENTS
# ═════════════════════════════════════════════════════════════════════════════

class MaddpgActor(nn.Module):
    def __init__(self, obs_dim: int = 10, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)
    
    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self.net(t).squeeze(0).cpu().numpy()


def load_maddpg_actors(checkpoint_path: str, n_salps: int, obs_dim: int = 10, hidden: int = 64) -> List:
    """Load MADDPG actors from checkpoint with correct architecture."""
    data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MaddpgActor(obs_dim=obs_dim, hidden=hidden).to(DEVICE) for _ in range(n_salps)]
    actor_states = data["actors"]
    for i, actor in enumerate(actors):
        actor.load_state_dict(actor_states[i])
        actor.eval()
    return actors


def demo_maddpg(label: str, script_path: str, checkpoint_path: str):
    """Run MADDPG demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        env_mod = load_module(script_path, f"maddpg_{label}")
        env = env_mod.FastEnv() if hasattr(env_mod, "FastEnv") else env_mod.Env()
        
        # Get hidden layer size from module constants if available
        hidden = getattr(env_mod, 'HIDDEN_DIM', getattr(env_mod, 'ACTOR_HIDDEN', 64))
        
        actors = load_maddpg_actors(checkpoint_path, env_mod.N_SALPS, hidden=hidden)
        
        def get_actions(obs_list):
            return [actors[i].act(obs_list[i]) for i in range(env_mod.N_SALPS)]
        
        visualize_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")
        import traceback
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# MAPPO AGENTS
# ═════════════════════════════════════════════════════════════════════════════

class MappoActor(nn.Module):
    def __init__(self, obs_dim: int = 10, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        h = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        return mean
    
    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self(t).squeeze(0).cpu().numpy()


def load_mappo_actors(checkpoint_path: str, n_salps: int, obs_dim: int = 10, hidden: int = 64) -> List:
    """Load MAPPO actors from checkpoint."""
    data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MappoActor(obs_dim=obs_dim, hidden=hidden).to(DEVICE) for _ in range(n_salps)]
    saved = data["actors"]
    share = data.get("share_params", False)
    for i, actor in enumerate(actors):
        idx = 0 if share else min(i, len(saved) - 1)
        actor.load_state_dict(saved[idx])
        actor.eval()
    return actors


def demo_mappo(label: str, script_path: str, checkpoint_path: str):
    """Run MAPPO demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        env_mod = load_module(script_path, f"mappo_{label}")
        env = env_mod.FastEnv() if hasattr(env_mod, "FastEnv") else env_mod.Env()
        
        # Get hidden layer size from module constants if available
        hidden = getattr(env_mod, 'HIDDEN_DIM', getattr(env_mod, 'ACTOR_HIDDEN', 64))
        
        actors = load_mappo_actors(checkpoint_path, env_mod.N_SALPS, hidden=hidden)
        
        def get_actions(obs_list):
            return [actors[i].act(obs_list[i]) for i in range(env_mod.N_SALPS)]
        
        visualize_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")
        import traceback
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BLD}{CYN}{'═'*70}{RST}")
    print(f"{BLD}{CYN}   Salp Chain Navigation · Live Algorithm Demo{RST}")
    print(f"{BLD}{CYN}   1 Episode per Algorithm | Live Visualization{RST}")
    print(f"{BLD}{CYN}{'═'*70}{RST}\n")
    print(f"  Device : {DEVICE}\n")
    
    # IQL Static
    demo_iql(
        label="IQL Static",
        script_path=os.path.join(REPO, "python_codes", "salp_iql_static_final.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "checkpoint_iql_static.pkl"),
    )
    
    # IQL Parallel
    iql_par_ckpt = os.path.join(REPO, "pickled_models", "checkpoint_iql_parallel.pkl")
    if not os.path.exists(iql_par_ckpt):
        iql_par_ckpt = os.path.join(REPO, "pickled_models", "iql_parallel.pkl")
    demo_iql(
        label="IQL Parallel",
        script_path=os.path.join(REPO, "python_codes", "salp_iql_parallel.py"),
        checkpoint_path=iql_par_ckpt,
    )
    
    # MADDPG Static
    demo_maddpg(
        label="MADDPG Static",
        script_path=os.path.join(REPO, "python_codes", "salp_maddpg_static.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "maddpg_static_checkpoint.pt"),
    )
    
    # MADDPG Dynamic
    demo_maddpg(
        label="MADDPG Dynamic",
        script_path=os.path.join(REPO, "python_codes", "salp_maddpg_parallel.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "maddpg_dynamic_checkpoint.pt"),
    )
    
    # MAPPO Static
    demo_mappo(
        label="MAPPO Static",
        script_path=os.path.join(REPO, "python_codes", "salp_mappo_static.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "mappo_checkpoint_mappo_static.pt"),
    )
    
    # MAPPO Dynamic
    demo_mappo(
        label="MAPPO Dynamic",
        script_path=os.path.join(REPO, "python_codes", "salp_mappo_parallel.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "mappo_dynamic_checkpoint.pt"),
    )
    
    print(f"\n{BLD}{CYN}{'═'*70}{RST}")
    print(f"{BLD}{GRN}All demos complete!{RST}")
    print(f"{BLD}{CYN}{'═'*70}{RST}\n")


if __name__ == "__main__":
    main()


# ═════════════════════════════════════════════════════════════════════════════
# IQL AGENTS
# ═════════════════════════════════════════════════════════════════════════════

ANGLE_BINS = 8
ACTION_DIRS = [
    (0.0, 0.0),
    (0.0,-1.0),(1.0,-1.0),(1.0,0.0),(1.0,1.0),
    (0.0, 1.0),(-1.0,1.0),(-1.0,0.0),(-1.0,-1.0),
]


def demo_iql(label: str, script_path: str, checkpoint_path: str):
    """Run IQL demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        safe_name = label.replace(" ", "_").replace("-", "_")
        env_mod = load_module(script_path, f"iql_{safe_name}")
        
        # Create env with headless=False for visualization
        env = env_mod.Env(headless=False) if hasattr(env_mod, "Env") else env_mod.FastEnv()
        
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        
        if isinstance(ckpt, dict):
            q_tables = ckpt.get("q_tables", ckpt.get("qtables", ckpt))
        else:
            q_tables = ckpt
        
        def iql_get_obs_key(obs: np.ndarray) -> tuple:
            """Discretise observation into hashable key."""
            x_bin   = int(obs[0] * 5)
            y_bin   = int(obs[1] * 5)
            ang     = math.atan2(float(obs[5]), float(obs[6]))
            ang_bin = int((ang + math.pi) / (2 * math.pi) * ANGLE_BINS) % ANGLE_BINS
            dist_b  = min(int(obs[4] * 5), 4)
            obs_b   = min(int(obs[7] * 5), 4)
            return (x_bin, y_bin, ang_bin, dist_b, obs_b)
        
        def get_actions(obs_list):
            actions = []
            for i in range(env_mod.N_SALPS):
                obs = obs_list[i]
                if isinstance(obs, np.ndarray) and len(obs) >= 7:
                    key = iql_get_obs_key(obs)
                    q = q_tables[i].get(key, [0.0] * len(ACTION_DIRS))
                    a = int(np.argmax(q))
                    actions.append(list(ACTION_DIRS[a]))
                else:
                    actions.append([0.0, 0.0])
            return actions
        
        run_episode_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")


# ═════════════════════════════════════════════════════════════════════════════
# MADDPG AGENTS
# ═════════════════════════════════════════════════════════════════════════════

class MaddpgActor(nn.Module):
    def __init__(self, obs_dim: int = 10, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)
    
    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self.net(t).squeeze(0).cpu().numpy()


def load_maddpg_actors(checkpoint_path: str, n_salps: int, obs_dim: int = 10, hidden: int = 64) -> List:
    """Load MADDPG actors from checkpoint with correct architecture."""
    data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MaddpgActor(obs_dim=obs_dim, hidden=hidden).to(DEVICE) for _ in range(n_salps)]
    actor_states = data["actors"]
    for i, actor in enumerate(actors):
        actor.load_state_dict(actor_states[i])
        actor.eval()
    return actors


def demo_maddpg(label: str, script_path: str, checkpoint_path: str):
    """Run MADDPG demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        env_mod = load_module(script_path, f"maddpg_{label}")
        env = env_mod.FastEnv() if hasattr(env_mod, "FastEnv") else env_mod.Env()
        
        # Get hidden layer size from module constants if available
        hidden = getattr(env_mod, 'HIDDEN_DIM', getattr(env_mod, 'ACTOR_HIDDEN', 64))
        
        actors = load_maddpg_actors(checkpoint_path, env_mod.N_SALPS, hidden=hidden)
        
        def get_actions(obs_list):
            return [actors[i].act(obs_list[i]) for i in range(env_mod.N_SALPS)]
        
        run_episode_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")


# ═════════════════════════════════════════════════════════════════════════════
# MAPPO AGENTS
# ═════════════════════════════════════════════════════════════════════════════

class MappoActor(nn.Module):
    def __init__(self, obs_dim: int = 10, action_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        h = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        return mean
    
    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
        return self(t).squeeze(0).cpu().numpy()


def load_mappo_actors(checkpoint_path: str, n_salps: int, obs_dim: int = 10, hidden: int = 64) -> List:
    """Load MAPPO actors from checkpoint."""
    data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    actors = [MappoActor(obs_dim=obs_dim, hidden=hidden).to(DEVICE) for _ in range(n_salps)]
    saved = data["actors"]
    share = data.get("share_params", False)
    for i, actor in enumerate(actors):
        idx = 0 if share else min(i, len(saved) - 1)
        actor.load_state_dict(saved[idx])
        actor.eval()
    return actors


def demo_mappo(label: str, script_path: str, checkpoint_path: str):
    """Run MAPPO demo with live visualization."""
    if not os.path.exists(script_path) or not os.path.exists(checkpoint_path):
        print(f"  {YLW}⊘ SKIP{RST}  {label} — missing files")
        return
    
    try:
        env_mod = load_module(script_path, f"mappo_{label}")
        env = env_mod.FastEnv() if hasattr(env_mod, "FastEnv") else env_mod.Env()
        
        # Get hidden layer size from module constants if available
        hidden = getattr(env_mod, 'HIDDEN_DIM', getattr(env_mod, 'ACTOR_HIDDEN', 64))
        
        actors = load_mappo_actors(checkpoint_path, env_mod.N_SALPS, hidden=hidden)
        
        def get_actions(obs_list):
            return [actors[i].act(obs_list[i]) for i in range(env_mod.N_SALPS)]
        
        run_episode_live(env, env_mod, get_actions, label)
        
    except Exception as e:
        print(f"  {RED}✗ Error in {label}: {e}{RST}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BLD}{CYN}{'═'*70}{RST}")
    print(f"{BLD}{CYN}   Salp Chain Navigation · Live Algorithm Demo{RST}")
    print(f"{BLD}{CYN}   1 Episode per Algorithm | Live Visualization{RST}")
    print(f"{BLD}{CYN}{'═'*70}{RST}\n")
    print(f"  Device : {DEVICE}\n")
    
    # IQL Static
    demo_iql(
        label="IQL Static",
        script_path=os.path.join(REPO, "python_codes", "salp_iql_static_final.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "checkpoint_iql_static.pkl"),
    )
    
    # IQL Parallel
    iql_par_ckpt = os.path.join(REPO, "pickled_models", "checkpoint_iql_parallel.pkl")
    if not os.path.exists(iql_par_ckpt):
        iql_par_ckpt = os.path.join(REPO, "pickled_models", "iql_parallel.pkl")
    demo_iql(
        label="IQL Parallel",
        script_path=os.path.join(REPO, "python_codes", "salp_iql_parallel.py"),
        checkpoint_path=iql_par_ckpt,
    )
    
    # MADDPG Static
    demo_maddpg(
        label="MADDPG Static",
        script_path=os.path.join(REPO, "python_codes", "salp_maddpg_static.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "maddpg_static_checkpoint.pt"),
    )
    
    # MADDPG Dynamic
    demo_maddpg(
        label="MADDPG Dynamic",
        script_path=os.path.join(REPO, "python_codes", "salp_maddpg_parallel.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "maddpg_dynamic_checkpoint.pt"),
    )
    
    # MAPPO Static
    demo_mappo(
        label="MAPPO Static",
        script_path=os.path.join(REPO, "python_codes", "salp_mappo_static.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "mappo_checkpoint_mappo_static.pt"),
    )
    
    # MAPPO Dynamic
    demo_mappo(
        label="MAPPO Dynamic",
        script_path=os.path.join(REPO, "python_codes", "salp_mappo_parallel.py"),
        checkpoint_path=os.path.join(REPO, "pickled_models", "mappo_dynamic_checkpoint.pt"),
    )
    
    print(f"\n{BLD}{CYN}{'═'*70}{RST}")
    print(f"{BLD}{GRN}All demos complete!{RST}")
    print(f"{BLD}{CYN}{'═'*70}{RST}\n")


if __name__ == "__main__":
    main()

