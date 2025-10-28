#!/usr/bin/env python3
"""
Data Collection Script for ORMSTC Multi-Robot Coverage Simulation
Automated testing for research data collection
"""

import sys
import os
import csv
import time
import random
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ormstc_simulation.simulation import ORMSTCSimulation


class DataCollector:
    def __init__(self):
        self.detailed_results = []
        self.start_time = None

    def run_single_test(self, grid_size, num_robots, num_obstacles, seed):
        """Run a single simulation test"""
        # Set obstacle seed to 42 (fixed)
        random.seed(42)
        np.random.seed(42)

        # Run simulation in console mode (no GUI for speed)
        sim = ORMSTCSimulation(
            grid_size=grid_size,
            num_robots=num_robots,
            num_obstacles=num_obstacles,
            robot_seed=seed,
            use_gui=False
        )

        # Run the simulation in silent mode (no console output)
        results = sim.run_console_simulation(show_grid=False, silent=True)

        return results

    def calculate_stats(self, results_list):
        """Calculate statistics from list of results"""
        coverage_times = [r['coverage_complete_step'] for r in results_list if r['coverage_complete_step'] is not None]
        total_steps = [r['total_steps'] for r in results_list]
        coverage_pcts = [r['coverage_percentage'] for r in results_list]

        stats = {
            'avg_coverage_time': np.mean(coverage_times) if coverage_times else None,
            'min_coverage_time': np.min(coverage_times) if coverage_times else None,
            'max_coverage_time': np.max(coverage_times) if coverage_times else None,
            'std_coverage_time': np.std(coverage_times) if coverage_times else None,
            'success_rate': (len(coverage_times) / len(results_list)) * 100,
            'avg_total_steps': np.mean(total_steps),
            'avg_coverage_pct': np.mean(coverage_pcts),
        }

        return stats

    def run_test_suite(self, grid_size, robot_counts, num_obstacles, num_iterations=25):
        """Run full test suite for a grid size"""
        print(f"\n{'='*70}")
        print(f"Starting tests for {grid_size}x{grid_size} grid")
        print(f"{'='*70}")

        summary_results = []

        for robot_count in robot_counts:
            print(f"\nTesting with {robot_count} robots...")
            iteration_results = []

            for iteration in range(1, num_iterations + 1):
                seed = iteration  # Use iteration number as seed

                # Progress indicator
                progress = (iteration / num_iterations) * 100
                bar_length = 30
                filled = int(bar_length * iteration / num_iterations)
                bar = '█' * filled + '-' * (bar_length - filled)

                print(f"\r  Iteration {iteration}/{num_iterations} [{bar}] {progress:.0f}%", end='', flush=True)

                try:
                    # Run test
                    results = self.run_single_test(grid_size, robot_count, num_obstacles, seed)

                    # Calculate total free cells for coverage percentage
                    total_free_cells = grid_size * grid_size - num_obstacles
                    coverage_percentage = (results['total_coverage'] / total_free_cells) * 100

                    # Calculate average stats per robot
                    avg_cells_per_robot = np.mean([r['cells_covered'] for r in results['robot_coverage']])
                    avg_path_per_robot = np.mean([r['path_length'] for r in results['robot_coverage']])

                    # Store detailed results with individual robot data
                    detailed_data = {
                        'grid_size': grid_size,
                        'num_robots': robot_count,
                        'iteration': iteration,
                        'seed': seed,
                        'coverage_time': results['coverage_complete_step'],
                        'total_steps': results['total_steps'],
                        'coverage_pct': coverage_percentage,
                        'robots_completed': results['robots_completed'],
                        'total_coverage': results['total_coverage']
                    }

                    # Add individual robot coverage data (r1, r2, r3, ...)
                    for robot_data in results['robot_coverage']:
                        robot_id = robot_data['robot_id']
                        detailed_data[f'r{robot_id}'] = robot_data['cells_covered']

                    self.detailed_results.append(detailed_data)

                    # Store for aggregation
                    iteration_results.append({
                        'coverage_complete_step': results['coverage_complete_step'],
                        'total_steps': results['total_steps'],
                        'coverage_percentage': coverage_percentage,
                        'avg_cells_per_robot': avg_cells_per_robot,
                        'avg_path_per_robot': avg_path_per_robot,
                    })

                except Exception as e:
                    print(f"\n  Error in iteration {iteration}: {e}")
                    continue

            print()  # New line after progress bar

            # Calculate aggregate statistics
            stats = self.calculate_stats(iteration_results)

            # Calculate average per-robot metrics
            avg_cells = np.mean([r['avg_cells_per_robot'] for r in iteration_results])
            avg_path = np.mean([r['avg_path_per_robot'] for r in iteration_results])

            summary_data = {
                'grid_size': grid_size,
                'num_robots': robot_count,
                'avg_coverage_time': stats['avg_coverage_time'],
                'min_coverage_time': stats['min_coverage_time'],
                'max_coverage_time': stats['max_coverage_time'],
                'std_coverage_time': stats['std_coverage_time'],
                'success_rate': stats['success_rate'],
                'avg_total_steps': stats['avg_total_steps'],
                'avg_coverage_pct': stats['avg_coverage_pct'],
                'avg_cells_per_robot': avg_cells,
                'avg_path_per_robot': avg_path,
            }

            summary_results.append(summary_data)

            print(f"  ✓ Completed: Avg coverage time = {stats['avg_coverage_time']:.1f} steps, Success rate = {stats['success_rate']:.0f}%")

        return summary_results

    def save_to_csv(self, data, filename, fieldnames=None):
        """Save data to CSV file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)

        if fieldnames is None:
            # Auto-detect fieldnames from first row
            if data:
                fieldnames = list(data[0].keys())

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)

        print(f"\n✓ Saved: {filename}")
        return filepath

    def save_all_detailed_results(self):
        """Save all detailed results into one combined CSV file"""
        if not self.detailed_results:
            print("No detailed results to save.")
            return 0

        # Find maximum number of robots across all tests
        max_robots = max(result['num_robots'] for result in self.detailed_results)

        # Create fieldnames with robot columns up to max
        base_fields = ['grid_size', 'num_robots', 'iteration', 'seed',
                      'coverage_time', 'total_steps', 'coverage_pct',
                      'robots_completed', 'total_coverage']
        robot_fields = [f'r{i}' for i in range(max_robots)]
        fieldnames = base_fields + robot_fields

        # Fill in missing robot columns with None for configurations with fewer robots
        for result in self.detailed_results:
            for i in range(max_robots):
                if f'r{i}' not in result:
                    result[f'r{i}'] = None

        filename = 'all_detailed_results.csv'
        self.save_to_csv(self.detailed_results, filename, fieldnames)

        return max_robots

    def run_all_tests(self):
        """Run all data collection tests"""
        self.start_time = time.time()

        print("\n" + "="*70)
        print("ORMSTC DATA COLLECTION - RESEARCH AUTOMATION")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_summary_results = []

        # Test configuration
        configs = [
            {
                'grid_size': 16,
                'robot_counts': [5, 10, 15, 20, 25, 30, 35, 40],
                'num_obstacles': 18,
            },
            {
                'grid_size': 50,
                'robot_counts': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                'num_obstacles': 175,
            }
        ]

        # Run tests for each configuration
        for config in configs:
            summary = self.run_test_suite(
                grid_size=config['grid_size'],
                robot_counts=config['robot_counts'],
                num_obstacles=config['num_obstacles'],
                num_iterations=25
            )
            all_summary_results.extend(summary)

        # Save summary results (combined)
        summary_fields = [
            'grid_size', 'num_robots', 'avg_coverage_time', 'min_coverage_time',
            'max_coverage_time', 'std_coverage_time', 'success_rate',
            'avg_total_steps', 'avg_coverage_pct', 'avg_cells_per_robot', 'avg_path_per_robot'
        ]
        self.save_to_csv(all_summary_results, 'summary_results.csv', summary_fields)

        # Save all detailed results into one combined file
        print("\n" + "="*70)
        print("Saving detailed results (all configurations combined)...")
        print("="*70)
        max_robots = self.save_all_detailed_results()

        # Also save separate summary files for each grid size
        grid_16_summary = [r for r in all_summary_results if r['grid_size'] == 16]
        grid_50_summary = [r for r in all_summary_results if r['grid_size'] == 50]

        self.save_to_csv(grid_16_summary, '16x16_summary.csv', summary_fields)
        self.save_to_csv(grid_50_summary, '50x50_summary.csv', summary_fields)

        # Print completion message
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE!")
        print("="*70)
        print(f"Total tests run: {len(self.detailed_results)}")
        print(f"Time elapsed: {minutes}m {seconds}s")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nSummary files (aggregated statistics):")
        print("  - summary_results.csv (all configurations combined)")
        print("  - 16x16_summary.csv (16x16 grid summary)")
        print("  - 50x50_summary.csv (50x50 grid summary)")
        print("\nDetailed file (all 450 test runs with individual robot coverage):")
        print("  - all_detailed_results.csv")
        print(f"\n    Columns: grid_size, num_robots, iteration, seed,")
        print(f"             coverage_time, total_steps, coverage_pct,")
        print(f"             robots_completed, total_coverage,")
        print(f"             r0, r1, r2, ... r{max_robots-1} (cells covered by each robot)")
        print("\n    Note: For configurations with fewer robots, extra columns will be empty")
        print("\nYou can now copy these CSV files into Excel!")
        print("="*70)


def main():
    """Main entry point"""
    collector = DataCollector()
    collector.run_all_tests()


if __name__ == "__main__":
    main()
