

# Salp Chain Navigation using Multi-Agent Reinforcement Learning

## Overview

This project implements a **multi-agent reinforcement learning (MARL)** system where a *salp chain* (a sequence of linked agents) learns to navigate toward a goal while avoiding obstacles.

The chain behaves like a **rigid structure (solid rod constraint)**, requiring coordinated motion between agents while interacting with a dynamic environment.

---

## Objectives

* Learn cooperative navigation in a constrained multi-agent system
* Maintain rigid link constraints between agents
* Avoid collisions with obstacles
* Evaluate multiple MARL algorithms

---

## Algorithms Implemented

* **Independent Q-Learning (IQL)**
* **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**
* **MAPPO (Multi-Agent Proximal Policy Optimization)**

Each algorithm is trained and evaluated under different environment configurations.

---

## Repository Structure

```id="structure01"
.
├── python_codes/          # Training implementations
├── pickled_models/        # Pretrained models (used for demo)
├── csv_outputs/           # Training logs
├── outputs/               # Training curves
├── assets/                # Demo GIFs
├── demo.py                # Runs pretrained models
├── run.sh                 # Fully automated execution script
└── README.md
```

---

## Demo



The demo visualizes a trained salp chain navigating toward a goal while maintaining structure and avoiding obstacles.

---

## Run Instructions (IMPORTANT)

### Single Command Execution

```bash id="run01"
bash run.sh
```

---

### What `run.sh` Does

The script **fully automates the entire pipeline**:

* Creates a virtual environment
* Installs all required dependencies
* Loads pretrained models
* Runs the demo environment
* Generates outputs (if applicable)

**No manual setup is required.**

---

## Demo Description

The demo:

* Loads pretrained models from `pickled_models/`
* Runs them inside the **original training environment**
* Displays:

  * Salp chain movement
  * Rigid link constraints
  * Obstacles
  * Goal position

---

## Key Features

### Rigid Chain Constraint

Maintains fixed distances between agents, ensuring realistic chain behavior.

### Physics-Based Dynamics

Includes velocity, drag, and thrust-based movement.

### Obstacle Avoidance

Polygon-based collision detection enables safe navigation.

### Multi-Agent Coordination

Agents act independently while optimizing a shared objective.

---

## Outputs

* Logs → `csv_outputs/`
* Plots → `outputs/`
* Models → `pickled_models/`

---

## Authors

* Anandita Garg
* Avantika Bansal
* Puneet Madan
* Trusha Maheshwari

---

## Summary

This project demonstrates how multi-agent reinforcement learning can be applied to control a physically constrained system, requiring coordination, obstacle avoidance, and goal-directed behavior.

---
