"""
Train all models
"""

import sys
import os
from pathlib import Path

# Add python_codes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_codes'))


def run_train_all():
    """Train all models."""
    training_scripts = [
        'salp_iql_static_final',
        'salp_iql_parallel',
        'salp_maddpg_static',
        'salp_maddpg_parallel',
        'salp_mappo_static',
        'salp_mappo_parallel',
    ]

    for script in training_scripts:
        print(f"\n{'=' * 60}")
        print(f"  Training: {script}")
        print(f"{'=' * 60}\n")
        try:
            module = __import__(script)
            # Call the main training function if it exists
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'train'):
                module.train()
            else:
                print(f"  Warning: No main() or train() function found in {script}")
        except Exception as e:
            print(f"\n  Error training {script}: {e}\n")

    print(f"\n{'=' * 60}")
    print("  All training completed!")
    print(f"{'=' * 60}\n")
