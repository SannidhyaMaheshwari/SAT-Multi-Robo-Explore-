#!/usr/bin/env python3
"""
Quick test script for data collection - one sample iteration
"""

import sys
import os
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ormstc_simulation.simulation import ORMSTCSimulation


def run_sample_test():
    """Run a single test iteration for 50x50 grid"""
    print("="*70)
    print("SAMPLE DATA COLLECTION TEST")
    print("="*70)
    print("\nConfiguration:")
    print("  Grid: 50x50")
    print("  Obstacles: 175 (fixed, seed=42)")
    print("  Robots: 50")
    print("  Robot placement seed: 1")
    print("\nRunning simulation...")
    print("-"*70)

    # Set obstacle seed to 42 (fixed)
    random.seed(42)
    np.random.seed(42)

    # Run simulation
    sim = ORMSTCSimulation(
        grid_size=50,
        num_robots=50,
        num_obstacles=175,
        robot_seed=1,
        use_gui=False
    )

    # Run in silent mode
    results = sim.run_console_simulation(show_grid=False, silent=True)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Calculate coverage percentage
    total_free_cells = 50 * 50 - 175
    coverage_percentage = (results['total_coverage'] / total_free_cells) * 100

    print(f"\nGeneral Stats:")
    print(f"  Coverage time: {results['coverage_complete_step']} steps")
    print(f"  Total steps: {results['total_steps']} steps")
    print(f"  Coverage: {results['total_coverage']}/{total_free_cells} cells ({coverage_percentage:.1f}%)")
    print(f"  Robots completed: {results['robots_completed']}/50")

    print(f"\nPer-Robot Coverage (cells covered by each robot):")
    print("-"*70)
    print("Robot ID | Cells Covered | Path Length")
    print("-"*70)
    for robot_data in results['robot_coverage'][:10]:  # Show first 10 robots
        print(f"   {robot_data['robot_id']:3d}   |      {robot_data['cells_covered']:3d}      |     {robot_data['path_length']:3d}")
    if len(results['robot_coverage']) > 10:
        print(f"   ...   |      ...      |     ...")

    # Show last robot
    last_robot = results['robot_coverage'][-1]
    print(f"   {last_robot['robot_id']:3d}   |      {last_robot['cells_covered']:3d}      |     {last_robot['path_length']:3d}")

    print("\n" + "="*70)
    print("Sample data in CSV format (for Excel):")
    print("="*70)

    # Generate CSV-like output
    print("\ngrid_size,num_robots,iteration,seed,coverage_time,total_steps,coverage_pct,robots_completed,total_coverage," +
          ",".join([f"r{i}" for i in range(50)]))

    csv_row = f"50,50,1,1,{results['coverage_complete_step']},{results['total_steps']},{coverage_percentage:.1f}," + \
              f"{results['robots_completed']},{results['total_coverage']}"

    # Add individual robot coverage
    for robot_data in results['robot_coverage']:
        csv_row += f",{robot_data['cells_covered']}"

    print(csv_row)

    print("\n" + "="*70)
    print("Test completed successfully!")
    print("You can now run the full data collection with:")
    print("  python3 data_collection.py")
    print("="*70)


if __name__ == "__main__":
    run_sample_test()
