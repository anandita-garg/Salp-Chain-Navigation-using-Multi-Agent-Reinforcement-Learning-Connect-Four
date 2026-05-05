#!/usr/bin/env python3
"""
Salp Chain MARL - Interactive Menu
"""

import sys
from demo_all import run_demo
from train_all import run_train_all


def display_menu():
    """Display the main menu."""
    print("\n" + "=" * 60)
    print("  Salp Chain Navigation - Multi-Agent RL")
    print("=" * 60)
    print("\n  Select an option:\n")
    print("    1. Run training (VERY INTENSIVE - will take considerable time)")
    print("    2. Run demo again")
    print("    3. Exit\n")


def main():
    """Main menu loop."""
    # Run demo immediately on startup
    print("\n" + "=" * 60)
    print("  Salp Chain Navigation - Multi-Agent RL")
    print("=" * 60)
    print("\n  Running demo for all models...\n")
    try:
        run_demo()
    except Exception as e:
        print(f"\n  Error running demo: {e}\n")

    # Show menu for further actions
    while True:
        display_menu()
        choice = input("  Enter your choice (1-3): ").strip()

        if choice == "1":
            confirm = input("\nTraining is VERY INTENSIVE and will take considerable time.\n  Continue? (y/n): ").strip().lower()
            if confirm == "y":
                print("\n  Training all models...\n")
                try:
                    run_train_all()
                except Exception as e:
                    print(f"\n  Error during training: {e}\n")
            else:
                print("\n  Training cancelled.\n")

        elif choice == "2":
            print("\n  Running demo for all models...\n")
            try:
                run_demo()
            except Exception as e:
                print(f"\n  Error running demo: {e}\n")

        elif choice == "3":
            print("\n  Exiting...\n")
            sys.exit(0)

        else:
            print("\n  Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
