#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import time

from .core import CellState
from .grid import Grid
from .communication import CommunicationSystem
from .robot import Robot


class ORMSTCSimulation:
    def __init__(self, grid_size=16, num_robots=3, num_obstacles=18, robot_seed=42, use_gui=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.robot_seed = robot_seed
        self.grid = Grid(grid_size, num_obstacles)
        self.comm_system = CommunicationSystem()
        self.robots = []
        self.step_count = 0
        self.use_gui = use_gui
        self.running = False
        # Dynamically set max_steps based on grid size
        # 16x16: 500 steps, 50x50: 3000 steps (scales with grid area)
        self.max_steps = 500 if grid_size <= 16 else 3000

        # Track coverage completion time
        self.coverage_complete_step = None  # Step at which 100% coverage was achieved

        # Cache total free cells (this never changes)
        self.total_free_cells = self._calculate_total_free_cells()

        # Place robots
        self._place_robots()

        if use_gui:
            # Setup visualization
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_xlim(-0.5, grid_size - 0.5)
            self.ax.set_ylim(-0.5, grid_size - 0.5)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.animation = None

    def _calculate_total_free_cells(self):
        """Calculate total number of free cells (cache this as it never changes)"""
        total = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.grid.is_obstacle(x, y):
                    total += 1
        return total

    def _place_robots(self):
        """Place robots at random non-obstacle positions using robot_seed"""
        # Set seed for robot placement only
        random.seed(self.robot_seed)
        np.random.seed(self.robot_seed)

        positions = []

        for i in range(self.num_robots):
            attempts = 0
            while attempts < 100:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)

                if not self.grid.is_obstacle(x, y) and (x, y) not in positions:
                    positions.append((x, y))
                    robot = Robot(i, x, y, self.grid, self.comm_system)
                    self.robots.append(robot)
                    break
                attempts += 1

    def _check_global_completion(self):
        """
        Check if all robots have either completed, finalized, or died,
        AND all free cells are covered (based on reported coverage).
        """
        total_free_cells = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.grid.is_obstacle(x, y):
                    total_free_cells += 1

        # Get all reported (shared) coverage
        reported_union = self.comm_system.get_global_reported_union()

        # --- Check 1: Coverage complete ---
        coverage_complete = len(reported_union) >= total_free_cells

        # --- Check 2: All robots done or dead ---
        all_done = True
        for r in self.robots:
            # If robot still alive and not finalized or completed, not done yet
            if r.alive and not (r.completed or getattr(r, "final_comp", False)):
                all_done = False
                break

        return coverage_complete and all_done


    def step(self):
        """Execute one simulation step"""
        # Check if already complete before stepping
        all_covered = self._check_global_completion()
        if all_covered:
            # Check if all robots are done (completed state)
            all_completed = all(robot.completed for robot in self.robots)
            # If all covered and all robots know they're completed, we're done
            if all_completed:
                # Check if they're all home or almost home
                all_home_or_close = all(
                    (robot.x, robot.y) == (robot.start_x, robot.start_y) or
                    robot.checking_connections or robot.returning_home
                    for robot in self.robots
                )
                if all_home_or_close:
                    return False  # Simulation is complete

        # Update global coverage for all robots
        for robot in self.robots:
            robot.update_global_coverage(self.robots)

        # Execute one step for each robot
        active_robots = 0
        exploring_robots = 0  # Count robots that are actively exploring (not just returning)
        for robot in self.robots:
            if robot.step():
                active_robots += 1
                # Only count as exploring if robot hasn't completed yet
                if not robot.completed:
                    exploring_robots += 1

        # Only increment step count if at least one robot is actively exploring
        # Don't count steps when robots are just returning home
        if exploring_robots > 0:
            self.step_count += 1

        self.comm_system.clear_old_messages()

        # Track when coverage is first completed
        if all_covered and self.coverage_complete_step is None:
            self.coverage_complete_step = self.step_count

        # Check completion (strict check)
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)

        return not (all_truly_done and all_covered)

    def update_visualization(self, frame):
        """Update function for animation"""
        # frame parameter required by matplotlib animation but not used
        if not self.running or self.step_count >= self.max_steps:
            return []

        # Check completion BEFORE calling step to prevent extra increments
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)
        all_covered = self._check_global_completion()

        if all_truly_done and all_covered:
            self.running = False
            if self.use_gui:
                print(f"\n{'='*60}")
                print(f"✓ SIMULATION COMPLETED at step {self.step_count}")
                print(f"{'='*60}")
            if self.animation:
                self.animation.event_source.stop()
            return []

        if not self.step():
            self.running = False
            if self.use_gui:
                print(f"\n{'='*60}")
                print(f"✓ SIMULATION COMPLETED at step {self.step_count}")
                print(f"{'='*60}")
            if self.animation:
                self.animation.event_source.stop()
            return []

        # Clear and redraw
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        # Draw grid lines
        for i in range(self.grid_size + 1):
            self.ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)
            self.ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.3)

        # Draw obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid.is_obstacle(x, y):
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                           linewidth=1, facecolor='black', edgecolor='gray')
                    self.ax.add_patch(rect)

        # Draw covered areas for each robot
        for robot in self.robots:
            color = robot.color
            for (x, y) in robot.local_map.covered_cells:
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                       linewidth=0, facecolor=color, alpha=0.2)
                self.ax.add_patch(rect)

        # Draw spanning tree edges
        for robot in self.robots:
            color = robot.color
            for (x, y), parent in robot.spanning_tree.items():
                px, py = parent
                self.ax.plot([px, x], [py, y], color=color, linewidth=1.5, alpha=0.7)

        # Draw robots
        for robot in self.robots:
            color = robot.color

            # Robot body
            circle = patches.Circle((robot.x, robot.y), 0.25,
                                  facecolor=color, edgecolor='black', linewidth=2)
            self.ax.add_patch(circle)

            # Robot ID
            self.ax.text(robot.x, robot.y, str(robot.robot_id),
                        ha='center', va='center', fontweight='bold',
                        color='white', fontsize=8)

            # Status indicator
            if robot.completed:
                if robot.checking_connections:
                    self.ax.text(robot.x + 0.35, robot.y + 0.35, '🔍',
                               ha='center', va='center', fontsize=10)
                else:
                    self.ax.text(robot.x + 0.35, robot.y + 0.35, '✓',
                               ha='center', va='center', color='green',
                               fontweight='bold', fontsize=12)

        # Calculate and display statistics
        all_covered = set()
        for robot in self.robots:
            all_covered.update(robot.local_map.covered_cells)

        coverage_percentage = (len(all_covered) / self.total_free_cells) * 100
        # Count robots that have completed their coverage (even if still returning home)
        completed_count = sum(1 for robot in self.robots if robot.completed)

        # Check if simulation is complete
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)
        sim_complete = all_truly_done and self._check_global_completion()

        # Check if coverage is complete (even if robots still returning home)
        coverage_complete = self._check_global_completion() and all(robot.completed for robot in self.robots)

        # Title with statistics
        title_text = f'ORMSTC Multi-Robot Coverage Simulation\n'
        if sim_complete:
            title_text += f'✓ SIMULATION COMPLETE - '
        elif coverage_complete:
            title_text += f'✓ COVERAGE COMPLETE (robots returning) - '
        title_text += f'Step: {self.step_count} | Robots: {self.num_robots} | '
        title_text += f'Completed: {completed_count} | Coverage: {coverage_percentage:.1f}%'

        # Add coverage completion time if available
        if self.coverage_complete_step is not None:
            title_text += f'\nTime to 100% coverage: {self.coverage_complete_step} steps'

        self.ax.set_title(
            title_text,
            fontsize=14, fontweight='bold',
            color='green' if sim_complete else 'black'
        )

        # Legend
        legend_elements = []
        for robot in self.robots:
            # Determine status symbol and text
            if robot.completed and robot.checking_connections:
                status = "🔍"
                status_text = "checking"
            elif robot.completed and robot.returning_home:
                status = "↩"
                status_text = "returning"
            elif robot.completed:
                status = "✓"
                status_text = "completed"
            else:
                status = "●"
                status_text = "exploring"

            legend_elements.append(
                patches.Patch(color=robot.color, label=f'Robot {robot.robot_id} {status}')
            )

        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        return []

    def run_animated_simulation(self):
        """Run simulation with real-time animation"""
        if not self.use_gui:
            print("GUI not available for this simulation")
            return self.run_console_simulation(silent=True)

        self.running = True
        print(f"Starting animated ORMSTC simulation...")
        print(f"Grid: {self.grid_size}x{self.grid_size}, Robots: {self.num_robots}")

        self.animation = FuncAnimation(
            self.fig, self.update_visualization,
            interval=600, blit=False, repeat=False
        )

        plt.show()
        return self.get_results()

    def run_console_simulation(self, show_grid=True, silent=False):
        """Run simulation in console mode"""
        if not silent:
            print(f"Starting ORMSTC simulation...")
            print(f"Grid: {self.grid_size}x{self.grid_size}, Robots: {self.num_robots}")

        if show_grid:
            print("Initial grid state:")
            self.grid.print_grid(self.robots)

        while self.step() and self.step_count < self.max_steps:
            if show_grid and self.step_count % 20 == 0:
                print(f"Step {self.step_count}:")
                self.grid.print_grid(self.robots)
            if not silent:
                time.sleep(0.05)

        if not silent:
            print(f"Simulation completed at step {self.step_count}")
        return self.get_results()

    def get_results(self):
        """Get simulation results"""
        results = {
            "total_steps": self.step_count,
            "coverage_complete_step": self.coverage_complete_step,  # Time to achieve 100% coverage
            "robots_completed": sum(1 for robot in self.robots if robot.completed),
            "total_coverage": 0,
            "robot_paths": [],
            "robot_coverage": []  # Per-robot area coverage (number of cells)
        }

        all_covered = set()
        for robot in self.robots:
            all_covered.update(robot.local_map.covered_cells)
            results["robot_paths"].append(robot.path)
            # Track how many cells each robot covered
            results["robot_coverage"].append({
                "robot_id": robot.robot_id,
                "cells_covered": len(robot.local_map.covered_cells),
                "path_length": len(robot.path)
            })

        results["total_coverage"] = len(all_covered)
        return results