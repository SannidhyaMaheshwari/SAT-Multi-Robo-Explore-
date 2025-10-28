#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import random
import time

from .core import CellState
from .grid import Grid
from .communication import CommunicationSystem
from .robot import Robot


class ORMSTCSimulation:
    def __init__(self, grid_size=16, num_robots=3, num_obstacles=18, robot_seed=42, use_gui=True, flush_trigger_cell=None, flush_batch_size=None):
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

        # NEW (Step 5): attack-related metrics
        self.re_explores = 0     # number of times fair robots stepped into hidden cells
        self.collisions = 0      # (will be used in Step 7)
        self.flush_trigger_cell = None      # e.g., (5, 2)
        self.flush_batch_size = None        # e.g., 10
        
        # Place robots
        self._place_robots()


        # stability detection for global reported coverage
        self.prev_reported_union = set()
        self.stable_reported_ticks = 0
        self.stability_threshold = 3   # require 3 ticks of identical reported coverage; tune as needed


                # --- Idle detection setup ---
        self.prev_positions = {}           # Track previous robot positions
        self.consecutive_idle_ticks = 0    # Number of consecutive ticks with no movement
        self.idle_threshold = 3            # Stop if no movement for this many ticks
        self.require_coverage_on_idle = False  # If True, stop only when idle + full coverage


        # NEW: mark highest-ID robot as malicious
        if self.robots:
            malicious_id = max(r.robot_id for r in self.robots)
            self.comm_system.set_malicious_robot(malicious_id)
        
        # NEW (Step 8): remember last seen flush epoch for UI/logging
        self.prev_flush_epoch = self.comm_system.flush_epoch
        
        # NEW: configure when the attacker will flush
        self.comm_system.set_flush_trigger_cell(None)   # e.g., (5, 2)
        self.comm_system.set_flush_batch_size(10)       # e.g., 10

        if use_gui:
            # Setup visualization
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_xlim(-0.5, grid_size - 0.5)
            self.ax.set_ylim(-0.5, grid_size - 0.5)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.animation = None


    def _count_free_cells(self):
        """Utility to count non-obstacle cells for coverage comparison."""
        total_free = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.grid.is_obstacle(x, y):
                    total_free += 1
        return total_free

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
        """Check if all accessible cells have been covered (based on REPORTED coverage).
        Uses a stability guard: reported coverage must be unchanged for
        self.stability_threshold consecutive ticks before we consider global completion true.
        """
        # Count free cells
        total_free_cells = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.grid.is_obstacle(x, y):
                    total_free_cells += 1

        # Current reported union (only reported, not hidden)
        reported_union = self.comm_system.get_global_reported_union()

        # If reported coverage has changed since last tick, reset stability counter
        if reported_union != self.prev_reported_union:
            self.prev_reported_union = set(reported_union)  # copy
            self.stable_reported_ticks = 0
        else:
            # unchanged since last tick -> increment stability counter
            self.stable_reported_ticks += 1

        # If coverage equals or exceeds free cells AND it's stable for threshold ticks -> done
        if len(reported_union) >= total_free_cells and self.stable_reported_ticks >= self.stability_threshold:
            # finalize robots so they won't flip back to active
            for r in self.robots:
                r.completed = True
                # if you use completed_finalized flag, set it too:
                if hasattr(r, "completed_finalized"):
                    r.completed_finalized = True
            return True

        return False


    def step(self):
        """
        Perform one simulation step: update all robots, handle communication, and check for completion.
        """

        if self._check_global_completion():
            print("[SIM] Global completion detected.")
            return False

        moved_this_tick = False
        self.flush_detected = False

        # Update all robots' global coverage before movement
        for r in self.robots:
            r.update_global_coverage(self.robots)

        # --- Robot Step Execution ---
        for r in self.robots:
            if not r.alive:
                continue

            prev_pos = (r.x, r.y)
            prev_reported = set(self.comm_system.get_reported_coverage(r.robot_id))

            result = r.step()  # perform one robot tick

            new_pos = (r.x, r.y)
            if new_pos != prev_pos:
                moved_this_tick = True

            # Track re-explores
            if not self.comm_system.is_malicious(r.robot_id):
                hidden = self.comm_system.hidden_coverage.get(self.malicious_robot_id, set())
                if new_pos in hidden:
                    self.re_explores += 1
                    r.overlap_candidates.add(new_pos)

            # Track collisions with malicious robot
            if not self.comm_system.is_malicious(r.robot_id):
                mal_pos = self.comm_system.robot_states.get(self.malicious_robot_id, {}).get("position")
                if mal_pos and new_pos == mal_pos and new_pos not in self.comm_system.reported_coverage.get(self.malicious_robot_id, set()):
                    print(f"[COLLISION] Fair Robot {r.robot_id} collided with malicious robot at {new_pos}.")
                    r.alive = False
                    self.collisions += 1

        # --- Flush detection (visual / metric) ---
        if self.comm_system.flush_count > self.prev_flush_count:
            self.flush_detected = True
            print(f"[SIM] Flush detected — new flush count = {self.comm_system.flush_count}")
            self.prev_flush_count = self.comm_system.flush_count

        # --- Handle collisions after all moves (multi-robot overlaps) ---
        positions = {}
        for r in self.robots:
            if not r.alive:
                continue
            pos = (r.x, r.y)
            if pos in positions:
                print(f"[COLLISION] Robots {r.robot_id} and {positions[pos]} occupy the same cell {pos}.")
                self.collisions += 1
                r.alive = False
            else:
                positions[pos] = r.robot_id

        # --- Clear old messages ---
        self.comm_system.clear_old_messages()

        # --- IDLE / NO-MOVEMENT DETECTION ---
        any_backtracking = any(getattr(r, "backtracking", False) for r in self.robots)
        moved_any = moved_this_tick

        if not moved_any:
            self.consecutive_idle_ticks += 1
        else:
            self.consecutive_idle_ticks = 0

        # Debug hint if robots are intentionally backtracking
        if any_backtracking:
            count = sum(1 for r in self.robots if r.backtracking)
            print(f"[SIM] {count} robot(s) backtracking — pausing idle-stop check.")

        # --- Stop only if idle AND no robots backtracking ---
        if self.consecutive_idle_ticks >= self.idle_threshold and not any_backtracking:
            reported = self.comm_system.get_global_reported_union()
            if (not self.require_coverage_on_idle) or (len(reported) >= self._count_free_cells()):
                for r in self.robots:
                    r.final_comp = True
                    try:
                        if hasattr(self.comm_system, "finalize_robot_completion"):
                            self.comm_system.finalize_robot_completion(r.robot_id)
                    except Exception:
                        pass
                print(f"[SIM] No movement for {self.consecutive_idle_ticks} ticks — stopping simulation.")
                return False

        return True


    def update_visualization(self, frame):
        """Update function for animation"""
        # frame parameter required by matplotlib animation but not used
        flush_happened = (self.comm_system.flush_epoch > self.prev_flush_epoch)
        if not self.running or self.step_count >= self.max_steps:
            return []

        # Check completion BEFORE calling step to prevent extra increments
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)
        all_covered = self._check_global_completion()

        if all_truly_done and all_covered:
            self.running = False
            print(f"\n{'='*60}")
            print(f"✓ SIMULATION COMPLETED at step {self.step_count}")
            print(f"{'='*60}")
            if self.animation:
                self.animation.event_source.stop()
            return []

        if not self.step():
            self.running = False
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

        # --- draw malicious hidden (buffered) cells as dashed magenta boxes ---
        mal_id = self.comm_system.malicious_id
        hidden_cells = self.comm_system.get_hidden_for_malicious() if mal_id is not None else set()
        for (hx, hy) in hidden_cells:
            rect = patches.Rectangle(
                (hx - 0.5, hy - 0.5), 1, 1,
                linewidth=1.5, edgecolor='magenta', facecolor='none', linestyle='--', alpha=0.8
            )
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
            # Robot body style: greyed out if dead
            face = color if robot.alive else 'lightgray'
            edge = 'black'

            # Robot body
            circle = patches.Circle((robot.x, robot.y), 0.25,
                                  facecolor=color, edgecolor='black', linewidth=2)
            self.ax.add_patch(circle)

            # Robot ID
            id_suffix = ''
            if self.comm_system.is_malicious(robot.robot_id):
                id_suffix += '☠'
            if not robot.alive:
                id_suffix += '✖'
            self.ax.text(robot.x, robot.y, f'{robot.robot_id}{id_suffix}',
                        ha='center', va='center', fontweight='bold',
                        color='white' if robot.alive else 'black', fontsize=8)

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
        true_covered = set()
        for robot in self.robots:
            true_covered.update(robot.local_map.covered_cells)

        reported_union = self.comm_system.get_global_reported_union()

        total_free_cells = int(np.sum(self.grid.grid != CellState.OBSTACLE.value))
        coverage_percentage = (len(reported_union) / total_free_cells) * 100
        # Count robots that have completed their coverage (even if still returning home)
        completed_count = sum(1 for robot in self.robots if robot.completed)

        # Check if simulation is complete
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)
        sim_complete = all_truly_done and self._check_global_completion()

        # Title with statistics
        title_text = f'ORMSTC Multi-Robot Coverage Simulation\n'
        if sim_complete:
            title_text += f'✓ COMPLETED - '
        title_text += f'Step: {self.step_count} | Robots: {self.num_robots} | '
        title_text += f'Completed: {completed_count} | Coverage: {coverage_percentage:.1f}%'
        # NEW: live attack metrics
        title_text += f' | Re-explores: {self.re_explores} | Collisions: {self.collisions} | Flushes: {self.comm_system.flush_count}'
        if flush_happened:
            title_text += ' | ⚠ FLUSH'

        self.ax.set_title(
            title_text,
            fontsize=14, fontweight='bold',
            color=('green' if sim_complete else 'black')
        )

        # Legend
        legend_elements = []
        for robot in self.robots:
            label = f'Robot {robot.robot_id}'
            if self.comm_system.is_malicious(robot.robot_id):
                label += ' (malicious)'
            if not robot.alive:
                label += ' [dead]'
            # status = "✓" if robot.completed else "●"
            # if robot.checking_connections:
            #     status = "🔍"
            legend_elements.append(
                patches.Patch(color=robot.color, label=label)
            )
        # Extra legend entries
        legend_elements.append(
            patches.Patch(facecolor='none', edgecolor='magenta', linestyle='--', label='Hidden (buffered) cells')
        )
        legend_elements.append(
            patches.Patch(color='lightgray', label='Dead/Collided robot')
        )

        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        # NEW: update last-seen epoch after drawing
        if flush_happened:
            self.prev_flush_epoch = self.comm_system.flush_epoch
        return []

    def run_animated_simulation(self):
        """Run simulation with real-time animation"""
        if not self.use_gui:
            print("GUI not available for this simulation")
            return self.run_console_simulation()

        self.running = True
        print(f"Starting animated ORMSTC simulation...")
        print(f"Grid: {self.grid_size}x{self.grid_size}, Robots: {self.num_robots}")

        self.animation = FuncAnimation(
            self.fig, self.update_visualization,
            interval=600, blit=False, repeat=False
        )

        plt.show()
        return self.get_results()

    def run_console_simulation(self, show_grid=True):
        """Run simulation in console mode"""
        print(f"Starting ORMSTC simulation...")
        print(f"Grid: {self.grid_size}x{self.grid_size}, Robots: {self.num_robots}")
        print(f"Metrics: re-explores={self.re_explores}, collisions={self.collisions}, flushes={self.comm_system.flush_count}")

        if show_grid:
            print("Initial grid state:")
            self.grid.print_grid(self.robots)

        while self.step() and self.step_count < self.max_steps:
            # NEW: if a flush just happened, log it
            if self.comm_system.flush_epoch > self.prev_flush_epoch:
                print(f"[Step {self.step_count}] ⚠ FLUSH detected (total flushes: {self.comm_system.flush_count})")
                self.prev_flush_epoch = self.comm_system.flush_epoch
            if show_grid and self.step_count % 20 == 0:
                print(f"Step {self.step_count}:")
                self.grid.print_grid(self.robots)
            time.sleep(0.05)

        print(f"Simulation completed at step {self.step_count}")
        print(f"Final metrics: re-explores={self.re_explores}, collisions={self.collisions}, flushes={self.comm_system.flush_count}")
        return self.get_results()

    def get_results(self):
        """Get simulation results"""
        results = {
            "total_steps": self.step_count,
            "robots_completed": sum(1 for robot in self.robots if robot.completed),
            "total_coverage": 0,
            "robot_paths": [],
            "re_explorations": self.re_explores,
            "collisions": self.collisions,   # placeholder for Step 7
            # Optional: useful comms-side metrics
            "flush_count": self.comm_system.flush_count,
            "delayed_cells": self.comm_system.delayed_cells_count
        }

        all_covered = set()
        for robot in self.robots:
            all_covered.update(robot.local_map.covered_cells)
            results["robot_paths"].append(robot.path)

        results["total_coverage"] = len(all_covered)
        return results