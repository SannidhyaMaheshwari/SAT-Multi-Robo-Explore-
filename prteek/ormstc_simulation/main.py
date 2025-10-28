#!/usr/bin/env python3

import sys
import os
import random
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ormstc_simulation.simulation import ORMSTCSimulation


def main():
    """Main function with improved interface"""
    print("ORMSTC Multi-Robot Coverage Simulation")
    print("=" * 50)

    # Get grid size
    print("\nChoose grid size:")
    print("1. 16x16 grid (18 obstacles)")
    print("2. 50x50 grid (175 obstacles)")
    try:
        grid_choice = input("Enter choice (1-2, default 1): ").strip() or "1"
        if grid_choice == "2":
            grid_size = 50
            num_obstacles = 175
        else:
            grid_size = 16
            num_obstacles = 18
    except ValueError:
        grid_size = 16
        num_obstacles = 18

    # Get number of robots
    try:
        num_robots = int(input("Number of robots (1-10, default 3): ") or "3")
        num_robots = max(1, min(10, num_robots))
    except ValueError:
        num_robots = 3

    # Get seed for robot placement
    try:
        seed = int(input("Random seed for robot placement (default 42): ") or "42")
    except ValueError:
        seed = 42

    print("\nChoose simulation mode:")
    print("1. Animated simulation (GUI)")
    print("2. Console simulation (text-based)")

    try:
        choice = input("Enter choice (1-2): ").strip()

        # Set random seed for reproducible obstacle generation (always fixed)
        random.seed(42)
        np.random.seed(42)

        print(f"\nSimulation settings:")
        print(f"  Grid: {grid_size}x{grid_size}")
        print(f"  Robots: {num_robots}")
        print(f"  Robot placement seed: {seed}")
        print(f"  Obstacles: {num_obstacles}")

        if choice == "1":
            # Animated simulation
            sim = ORMSTCSimulation(grid_size, num_robots, num_obstacles, robot_seed=seed, use_gui=True)
            results = sim.run_animated_simulation()

        elif choice == "2":
            # Console simulation
            sim = ORMSTCSimulation(grid_size, num_robots, num_obstacles, robot_seed=seed, use_gui=False)
            results = sim.run_console_simulation(show_grid=True)

        else:
            print("Invalid choice!")
            return

        # Display final results
        print("\n" + "="*60)
        print("FINAL SIMULATION RESULTS")
        print("="*60)
        print(f"Total steps (simulation time): {results['total_steps']}")

        # Display coverage completion time
        if results['coverage_complete_step'] is not None:
            print(f"Time to 100% coverage: {results['coverage_complete_step']} steps")
        else:
            print(f"Coverage not completed (reached max steps)")

        print(f"Robots completed: {results['robots_completed']}/{num_robots}")
        print(f"Total coverage: {results['total_coverage']} cells")

        # Calculate coverage efficiency
        total_free_cells = grid_size * grid_size - num_obstacles
        coverage_percentage = (results['total_coverage'] / total_free_cells) * 100
        print(f"Total free cells: {total_free_cells}")
        print(f"Coverage percentage: {coverage_percentage:.1f}%")

        if coverage_percentage >= 99.0:
            print("🎉 Excellent coverage achieved!")
        elif coverage_percentage >= 95.0:
            print("✅ Good coverage achieved!")

        print("\n" + "-"*60)
        print("PER-ROBOT STATISTICS")
        print("-"*60)
        for robot_data in results['robot_coverage']:
            robot_id = robot_data['robot_id']
            cells_covered = robot_data['cells_covered']
            path_length = robot_data['path_length']
            print(f"Robot {robot_id}:")
            print(f"  Area covered: {cells_covered} cells")
            print(f"  Path length: {path_length} steps")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()