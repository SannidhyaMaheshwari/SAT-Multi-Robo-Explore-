#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import time
from enum import Enum
from copy import deepcopy


class CellState(Enum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    COVERED = 3
    ROBOT = 4
    TREE_EDGE = 5


class Direction(Enum):
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    UP = (0, -1)


class Message:
    def __init__(self, sender_id, msg_type, data, timestamp):
        self.sender_id = sender_id
        self.msg_type = msg_type
        self.data = data
        self.timestamp = timestamp


class Grid:
    def __init__(self, size=16, obstacle_density=0.2):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.obstacle_density = obstacle_density
        self._generate_obstacles()

    def _generate_obstacles(self):
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.grid[y, x] = CellState.OBSTACLE.value

    def is_valid_position(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def is_obstacle(self, x, y):
        if not self.is_valid_position(x, y):
            return True
        return self.grid[y, x] == CellState.OBSTACLE.value

    def get_neighbors(self, x, y):
        """Get neighbors in clockwise order: right, down, left, up"""
        neighbors = []
        # Clockwise order: right, down, left, up
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        for direction in directions:
            dx, dy = direction.value
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_cell_state(self, x, y):
        """Return the CellState of the given cell in the global grid."""
        if not self.is_valid_position(x, y):
            return CellState.OBSTACLE
        return CellState(self.grid[y, x])


    def print_grid(self, robots=None):
        """Print ASCII representation of the grid"""
        print("\n" + "="*40)
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if robots:
                    robot_here = None
                    for robot in robots:
                        if robot.x == x and robot.y == y:
                            robot_here = robot
                            break

                    if robot_here:
                        row += f"R{robot_here.robot_id}"
                        continue

                if self.is_obstacle(x, y):
                    row += "██"
                else:
                    row += "  "
            print(f"|{row}|")
        print("="*40)


class LocalMap:
    def __init__(self, size=16):
        self.size = size
        self.map = np.full((size, size), CellState.UNKNOWN.value)
        self.tree_edges = {}  # (x, y) -> robot_id
        self.covered_cells = set()

    def update_cell(self, x, y, state):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.map[y, x] = state.value

    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return CellState(self.map[y, x])
        return CellState.OBSTACLE

    def add_tree_edge(self, x, y, robot_id):
        self.tree_edges[(x, y)] = robot_id
        self.update_cell(x, y, CellState.TREE_EDGE)

    def mark_covered(self, x, y):
        self.covered_cells.add((x, y))
        self.update_cell(x, y, CellState.COVERED)


class CommunicationSystem:
    def __init__(self):
        self.messages = []
        self.robot_states = {}
        self.active_robots = set()

    def register_robot(self, robot_id):
        self.active_robots.add(robot_id)
        self.robot_states[robot_id] = {
            "alive": True,
            "position": None,
            "last_activity": time.time()
        }

    def broadcast_message(self, sender_id, msg_type, data):
        message = Message(sender_id, msg_type, data, time.time())
        self.messages.append(message)

        # Update last activity time for alive tracking (paper-compliant)
        if sender_id in self.robot_states:
            self.robot_states[sender_id]["last_activity"] = time.time()
            self.robot_states[sender_id]["alive"] = True

    def get_messages_for_robot(self, robot_id):
        return [msg for msg in self.messages if msg.sender_id != robot_id]

    def clear_old_messages(self):
        current_time = time.time()
        self.messages = [msg for msg in self.messages if current_time - msg.timestamp < 1.0]

    def update_robot_state(self, robot_id, state):
        if robot_id in self.robot_states:
            self.robot_states[robot_id].update(state)

    def is_robot_alive(self, robot_id):
        """Check if robot is alive based on recent activity (paper-compliant method)"""
        if robot_id not in self.robot_states:
            return False

        # Robot is considered alive if it has sent any message recently
        current_time = time.time()
        last_activity = self.robot_states[robot_id].get("last_activity", 0)

        # Consider robot dead if no activity for 5 simulation steps (reasonable timeout)
        timeout_threshold = 5.0
        is_alive = (current_time - last_activity) < timeout_threshold

        # Update alive status
        self.robot_states[robot_id]["alive"] = is_alive
        return is_alive


class Robot:
    def __init__(self, robot_id, start_x, start_y, grid, comm_system):
        self.robot_id = robot_id
        self.x, self.y = start_x, start_y
        self.start_x, self.start_y = start_x, start_y
        self.grid = grid
        self.comm_system = comm_system
        self.local_map = LocalMap(grid.size)

        # Robot state
        self.alive = True
        self.completed = False
        self.returning_home = False
        self.checking_connections = False
        self.path = [(start_x, start_y)]
        self.spanning_tree = {}  # parent relationships
        self.connections = {}  # connections to other robots' trees
        self.global_coverage = set()

        # Define unique color for each robot
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.color = colors[robot_id % len(colors)]

        # Initialize starting position
        self.local_map.update_cell(start_x, start_y, CellState.FREE)
        self.local_map.mark_covered(start_x, start_y)
        self.comm_system.register_robot(robot_id)
        self.comm_system.update_robot_state(robot_id, {
            "position": (start_x, start_y),
            "alive": True
        })

    def update_global_coverage(self, all_robots):
        self.global_coverage.clear()
        for robot in all_robots:
            if not robot.alive:
                continue

            if getattr(robot, "is_malicious", False):
                # Only trust what malicious robot has actually reported
                if hasattr(robot, "reported_cells"):
                    self.global_coverage.update(robot.reported_cells)
            else:
                # Normal robots share everything they’ve covered
                self.global_coverage.update(robot.local_map.covered_cells)

    def is_globally_complete(self):
        """Check if entire grid is covered by all robots"""
        total_free_cells = 0
        for y in range(self.grid.size):
            for x in range(self.grid.size):
                if not self.grid.is_obstacle(x, y):
                    total_free_cells += 1
        return len(self.global_coverage) >= total_free_cells

    def sense_environment(self):
        """Sense neighboring cells and update local map"""
        neighbors = self.grid.get_neighbors(self.x, self.y)
        sensed_data = []

        for nx, ny in neighbors:
            if self.grid.is_obstacle(nx, ny):
                self.local_map.update_cell(nx, ny, CellState.OBSTACLE)
                sensed_data.append((nx, ny, CellState.OBSTACLE))
            else:
                current_state = self.local_map.get_cell(nx, ny)
                if current_state == CellState.UNKNOWN:
                    self.local_map.update_cell(nx, ny, CellState.FREE)
                    sensed_data.append((nx, ny, CellState.FREE))

        return sensed_data

    def _check_cell_conflict(self, x, y):
        """Check if another robot is also trying to claim this cell"""
        messages = self.comm_system.get_messages_for_robot(self.robot_id)

        conflicting_robots = []
        for message in messages:
            if message.msg_type == "claiming":
                claim_data = message.data
                if claim_data.get("target") == (x, y):
                    conflicting_robots.append(message.sender_id)

        # If multiple robots want the same cell, return the one with highest ID
        if conflicting_robots:
            return max(conflicting_robots)
        return None

    def find_next_cell(self):
        """Find next uncovered free cell with tie-breaking for conflicts"""
        if self.returning_home or self.checking_connections:
            return None, None

        neighbors = self.grid.get_neighbors(self.x, self.y)
        candidates = []

        for nx, ny in neighbors:
            cell_state = self.local_map.get_cell(nx, ny)

            # Skip if obstacle
            if cell_state == CellState.OBSTACLE:
                continue

            # Skip if already covered globally
            if (nx, ny) in self.global_coverage:
                continue

            # Check if another robot owns this cell
            if (nx, ny) in self.local_map.tree_edges:
                other_robot = self.local_map.tree_edges[(nx, ny)]
                if self.comm_system.is_robot_alive(other_robot):
                    # Save connection for potential takeover
                    if other_robot not in self.connections:
                        self.connections[other_robot] = []
                    self.connections[other_robot].append((self.x, self.y, nx, ny))
                    continue
                else:
                    # Dead robot, can take over
                    self.local_map.tree_edges.pop((nx, ny), None)

            # Check for conflicts with other robots trying to claim same cell
            conflicting_robot = self._check_cell_conflict(nx, ny)
            if conflicting_robot is not None:
                # Tie-breaking: higher robot ID wins
                if self.robot_id > conflicting_robot:
                    # We win the conflict, proceed to check if cell is valid
                    if cell_state == CellState.FREE and (nx, ny) not in self.local_map.covered_cells:
                        candidates.append((nx, ny))
                # If we lose, skip this cell
                continue

            # No conflict - found valid next cell
            if cell_state == CellState.FREE and (nx, ny) not in self.local_map.covered_cells:
                candidates.append((nx, ny))

        if candidates:
            return candidates[0]
        return None, None

    def move_to_cell(self, target_x, target_y):
        """
        Move to target cell with conflict resolution.

        - Collision: if malicious robot is physically in the target cell → stop immediately.
        - Re-exploration: if the cell was already in malicious.unreported_covered_cells → log it,
        but don't stop here. Fair robot will stop later when malicious releases.
        """

        if target_x is None or target_y is None:
            return False

        sim = getattr(self, 'simulation', None)

        # --- Collision detection BEFORE broadcasting claim ---
        if sim is not None:
            for other in sim.robots:
                if getattr(other, 'is_malicious', False):
                    if (other.x, other.y) == (target_x, target_y):
                        # ✅ Physical collision
                        self.x, self.y = target_x, target_y
                        self.path.append((self.x, self.y))
                        self.local_map.mark_covered(self.x, self.y)

                        self.completed = True
                        self.checking_connections = False
                        self.returning_home = False

                        sim.attack_metrics.setdefault('collisions', 0)
                        sim.attack_metrics['collisions'] += 1
                        self.metrics = getattr(self, 'metrics', {})
                        self.metrics['collisions'] = self.metrics.get('collisions', 0) + 1

                        sim.comm_system.broadcast_message(
                            self.robot_id, "collision", {"robot_id": self.robot_id, "at": (self.x, self.y)}
                        )
                        return True

        # --- Normal move ---
        self.comm_system.broadcast_message(
            self.robot_id,
            "claiming",
            {"target": (target_x, target_y), "position": (self.x, self.y)}
        )

        self.local_map.add_tree_edge(target_x, target_y, self.robot_id)
        self.spanning_tree[(target_x, target_y)] = (self.x, self.y)

        self.x, self.y = target_x, target_y
        self.path.append((self.x, self.y))
        self.local_map.mark_covered(self.x, self.y)

        self.comm_system.broadcast_message(
            self.robot_id,
            "move",
            {"position": (self.x, self.y), "tree_edge": (target_x, target_y)}
        )
        self.comm_system.update_robot_state(self.robot_id, {"position": (self.x, self.y)})

        # --- Re-exploration detection ---
        if sim is not None:
            for other in sim.robots:
                if getattr(other, 'is_malicious', False) and hasattr(other, 'unreported_covered_cells'):
                    if (self.x, self.y) in other.unreported_covered_cells:
                        sim.attack_metrics.setdefault('reexplorations', 0)
                        sim.attack_metrics['reexplorations'] += 1

                        self.metrics = getattr(self, 'metrics', {})
                        self.metrics['reexplorations'] = self.metrics.get('reexplorations', 0) + 1

                        # 🚫 Do NOT stop here. Robot will stop when malicious releases.
                        return True

        return True

    def perform_hilling(self):
        """Perform hilling procedure as described in the paper to extend spanning tree path"""
        if len(self.path) < 2:
            return False

        current_pos = (self.x, self.y)

        # Look for two joint unoccupied cells adjacent to the current path
        # This creates "hills" that help stretch the spanning tree optimally
        neighbors = self.grid.get_neighbors(self.x, self.y)

        # Find pairs of adjacent free cells that can form a hill
        for nx1, ny1 in neighbors:
            if (nx1, ny1) in self.local_map.covered_cells or self.grid.get_cell_state(nx1, ny1) != CellState.FREE:
                continue

            # Check if this cell has an adjacent free cell (forming a hill)
            hill_neighbors = self.grid.get_neighbors(nx1, ny1)
            for nx2, ny2 in hill_neighbors:
                if ((nx2, ny2) not in self.local_map.covered_cells and
                    self.grid.get_cell_state(nx2, ny2) == CellState.FREE and
                    (nx2, ny2) != current_pos and (nx2, ny2) != (nx1, ny1)):

                    # Found a valid hill - two adjacent free cells
                    # Add them to the spanning tree path
                    if self.move_to_cell(nx1, ny1):
                        if self.move_to_cell(nx2, ny2):
                            return True

        return False

    def backtrack(self):
        """Backtrack along spanning tree when no forward moves available"""
        if (self.x, self.y) in self.spanning_tree:
            parent_x, parent_y = self.spanning_tree[(self.x, self.y)]
            self.x, self.y = parent_x, parent_y
            # Don't add to path during backtracking to avoid counting unnecessary steps
            return True
        return False

    def move_towards_start(self):
        """Move towards starting position for connection checking"""
        if (self.x, self.y) == (self.start_x, self.start_y):
            return False

        # Simple movement towards start (Manhattan distance)
        dx = self.start_x - self.x
        dy = self.start_y - self.y

        if abs(dx) > abs(dy):
            next_x = self.x + (1 if dx > 0 else -1)
            next_y = self.y
        else:
            next_x = self.x
            next_y = self.y + (1 if dy > 0 else -1)

        if self.grid.is_valid_position(next_x, next_y) and not self.grid.is_obstacle(next_x, next_y):
            self.x, self.y = next_x, next_y
            # Don't count return journey steps in path
            return True

        return False

    def check_failed_robots(self):
        """Check connections for failed robots and take over if needed"""
        for robot_id, connection_list in self.connections.items():
            if not self.comm_system.is_robot_alive(robot_id):
                # Robot failed, need to take over its area
                self.comm_system.broadcast_message(
                    self.robot_id,
                    "takeover",
                    {"failed_robot": robot_id, "connections": connection_list}
                )
                # Move to connection point and start coverage
                if connection_list:
                    # Reset state to continue coverage
                    self.completed = False
                    self.returning_home = False
                    self.checking_connections = False
                    return True
        return False

    def monitor_connected_robots_after_completion(self):
        """Monitor connected robots after completion as per Algorithm 3 lines 25-36"""
        # Check if any connected robot has failed
        for robot_id in list(self.connections.keys()):
            if not self.comm_system.is_robot_alive(robot_id):
                # Robot failed - withdraw completion and take over its area
                self.comm_system.broadcast_message(
                    self.robot_id,
                    "withdrawal",
                    {"taking_over": robot_id}
                )

                # Reset state to continue coverage of failed robot's area
                self.completed = False
                self.returning_home = False
                self.checking_connections = False

                # Mark failed robot's cells as empty for coverage
                # (This would involve more complex state management in full implementation)
                break

    def process_messages(self):
        """Process messages from other robots"""
        messages = self.comm_system.get_messages_for_robot(self.robot_id)

        for message in messages:
            if message.msg_type == "move":
                data = message.data
                tree_edge = data.get("tree_edge")
                if tree_edge:
                    tx, ty = tree_edge
                    self.local_map.add_tree_edge(tx, ty, message.sender_id)

            elif message.msg_type == "completed":
                pass

            elif message.msg_type == "takeover":
                # Another robot is taking over a failed robot's area
                pass

    def check_completion(self):
        """Enhanced completion check"""
        # Check if there are any uncovered cells accessible to this robot
        next_cell = self.find_next_cell()
        if next_cell[0] is not None:
            return False

        # Check if global coverage is complete
        return self.is_globally_complete()

    def step(self):
        """Execute one step of the ORMSTC algorithm"""
        if not self.alive:
            return False

        # If completed and checking connections
        if self.completed and self.checking_connections:
            if (self.x, self.y) == (self.start_x, self.start_y):
                # Check for failed robots
                if self.check_failed_robots():
                    return True  # Found failed robot, resume coverage
                else:
                    # All connections checked, truly done
                    return False
            else:
                # Move towards start to check connections
                return self.move_towards_start()

        # If completed but need to return home
        if self.completed and not self.checking_connections:
            if (self.x, self.y) == (self.start_x, self.start_y):
                # Reached start, now check connections
                self.checking_connections = True
                return True
            else:
                # Return to starting position
                self.returning_home = True
                return self.move_towards_start()

        # Normal operation
        self.process_messages()
        self.sense_environment()

        # Try to find next cell
        next_x, next_y = self.find_next_cell()

        if next_x is not None and next_y is not None:
            # Move forward
            return self.move_to_cell(next_x, next_y)
        else:
            # Try hilling first (paper algorithm step)
            if self.perform_hilling():
                return True

            # Then try to backtrack
            if self.backtrack():
                return True
            else:
                # Check if completed
                if self.check_completion():
                    self.completed = True
                    self.comm_system.broadcast_message(
                        self.robot_id,
                        "completed",
                        {"robot_id": self.robot_id}
                    )

                    # Paper Algorithm 3: Monitor connected robots after completion
                    self.monitor_connected_robots_after_completion()
                    return True  # Continue to return home
                
                # If robot is at start and has nothing left to do, mark completed
                if not self.completed and (self.x, self.y) == (self.start_x, self.start_y):
                    if self.is_globally_complete() or not self.find_next_cell()[0]:
                        self.completed = True
                        self.checking_connections = False
                        self.returning_home = False
                        return False

                return False


class MaliciousRobot(Robot):
    """
    MaliciousRobot buffers its own coverage/claiming messages locally (self.outbox)
    and releases them according to attack_cfg policy, while still moving normally.
    Minimal, non-invasive: mirrors Robot.move_to_cell but buffers instead of broadcasting.
    """

    def __init__(self, robot_id, start_x, start_y, grid, comm_system, attack_cfg=None):
        super().__init__(robot_id, start_x, start_y, grid, comm_system)
        # Mark malicious
        self.is_malicious = True

        # Outbox: list of dicts {'msg_type': str, 'data': dict, 'orig_step': int, 'orig_time': float}
        self.outbox = []

        # Cells that malicious has covered but not yet announced
        self.unreported_covered_cells = set()

        # Attack configuration
        # Examples:
        # {'mode':'fixed_cells_delay', 'threshold':3}
        # {'mode':'fixed_steps_delay', 'threshold':5}
        # {'mode':'time_delay', 'threshold':2.0}  # seconds
        # {'mode':'reach_cell_release', 'release_cell': (5,2)}
        self.attack_cfg = attack_cfg or {'mode': 'fixed_cells_delay', 'threshold': 3}

        # bookkeeping for step-based delay (use simulation.step_count if available)
        self._buffer_first_step = None
        self.metrics = {
            'delayed_msgs_sent': 0,
            'delays': [],   # per-message delay in sim-steps (or seconds if time-based)
            'collisions_caused': 0
        }
        self.reported_cells = set()

    # --- Buffering helpers ---
    def _current_step(self):
        if hasattr(self, 'simulation') and getattr(self.simulation, 'step_count', None) is not None:
            return self.simulation.step_count
        return None

    def buffer_message(self, msg_type, data):
        """Append message to local outbox (records orig step/time)."""
        entry = {
            'msg_type': msg_type,
            'data': deepcopy(data),
            'orig_step': self._current_step(),
            'orig_time': time.time()
        }
        self.outbox.append(entry)

        # Track covered cells that are not yet reported (so re-exploration detection / collisions use this)
        if msg_type == 'move':
            pos = data.get('position')
            if pos:
                self.unreported_covered_cells.add(tuple(pos))

        # record when buffering started
        if self._buffer_first_step is None:
            self._buffer_first_step = entry['orig_step']

    def release_outbox_if_needed(self):
        """Check attack_cfg and release whole outbox (FIFO) if policy triggers."""
        mode = self.attack_cfg.get('mode', 'fixed_cells_delay')
        thr = self.attack_cfg.get('threshold', 3)
        now = time.time()
        cur_step = self._current_step()

        should_release = False

        if not self.outbox:
            return  # nothing to release

        if mode == 'fixed_cells_delay':
            if len(self.unreported_covered_cells) >= int(thr):
                should_release = True

        elif mode == 'fixed_steps_delay':
            # release if we have buffered for >= threshold simulation steps
            if self._buffer_first_step is not None and cur_step is not None:
                if (cur_step - self._buffer_first_step) >= int(thr):
                    should_release = True

        elif mode == 'time_delay':
            oldest = self.outbox[0]['orig_time']
            if (now - oldest) >= float(thr):
                should_release = True

        elif mode == 'reach_cell_release':
            release_cell = self.attack_cfg.get('release_cell')
            if release_cell and (self.x, self.y) == tuple(release_cell):
                should_release = True

        elif mode == 'always_release':
            should_release = True

        # You can add other modes here (burst_replay, random_delay, etc.)
        if should_release:
            released_cells = []   # ✅ track all cells being released this round
            while self.outbox:
                entry = self.outbox.pop(0)
                if entry['msg_type'] == 'move':
                    pos = entry['data']['position']
                    released_cells.append(pos)   # ✅ save for later fair-robot check
                    if pos in self.unreported_covered_cells:
                        self.unreported_covered_cells.remove(pos)
                        self.reported_cells.add(pos)

                # compute delay, broadcast as before
                if cur_step is not None and entry['orig_step'] is not None:
                    delay_metric = cur_step - entry['orig_step']
                else:
                    delay_metric = time.time() - entry['orig_time']
                self.metrics['delays'].append(delay_metric)

                self.comm_system.broadcast_message(self.robot_id, entry['msg_type'], entry['data'])
                self.metrics['delayed_msgs_sent'] += 1
                if hasattr(self, "simulation") and self.simulation is not None:
                    self.simulation.attack_metrics['delayed_messages_total'] += 1

            # ✅ After releasing, stop fair robots that unknowingly re-explored these cells
            if hasattr(self, "simulation") and self.simulation is not None:
                for r in self.simulation.robots:
                    if not getattr(r, 'is_malicious', False):
                        if any(c in r.local_map.covered_cells for c in released_cells):
                            r.completed = True
                            r.checking_connections = False
                            r.returning_home = False

            # Clear bookkeeping
            self.unreported_covered_cells.clear()
            self._buffer_first_step = None


    # --- Movement override (buffers messages instead of broadcasting them immediately) ---
    def move_to_cell(self, target_x, target_y):
        """Move very similarly to Robot.move_to_cell but buffer the claim & move messages."""
        if target_x is None or target_y is None:
            return False

        # Buffer the claiming message instead of broadcasting instantly
        self.buffer_message(
            'claiming',
            {"target": (target_x, target_y), "position": (self.x, self.y)}
        )

        # Add tree edge locally (same as base robot)
        self.local_map.add_tree_edge(target_x, target_y, self.robot_id)
        self.spanning_tree[(target_x, target_y)] = (self.x, self.y)

        # Move robot
        self.x, self.y = target_x, target_y
        self.path.append((self.x, self.y))
        self.local_map.mark_covered(self.x, self.y)

        # Buffer the move message
        self.buffer_message(
            'move',
            {"position": (self.x, self.y), "tree_edge": (target_x, target_y)}
        )

        # Update own robot state in comm_system (this keeps alive-tracking accurate)
        # NOTE: We still update robot_state last_activity here as the robot is active.
        self.comm_system.update_robot_state(self.robot_id, {
            "position": (self.x, self.y)
        })

        return True

    # --- per-step tick routine to be called from Robot.step() or simulation loop ---
    def malicious_tick(self):
        """
        Called each simulation step (e.g. from Robot.step() or after robots are created
        set robot.simulation = sim and call this from sim loop if desired).
        """
        # if we've started buffering, consider incrementing step-based counter implicitly via simulation.step_count
        # release according to policy
        self.release_outbox_if_needed()


class ORMSTCSimulation:
    def __init__(self, grid_size=16, num_robots=3, obstacle_density=0.15, use_gui=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.grid = Grid(grid_size, obstacle_density)
        self.comm_system = CommunicationSystem()
        self.robots = []
        self.step_count = 0
        self.use_gui = use_gui
        self.running = False
        self.max_steps = 500
        # metrics for attacks & measurements
        self.attack_metrics = {
            'collisions': 0,
            'reexplorations': 0,
            'delayed_messages_total': 0,   # The malicious robot should increment this when releasing messages
        }

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

    def _place_robots(self):
        """Place robots at random non-obstacle positions"""
        # inside ORMSTCSimulation._place_robots() where robots are appended
        positions = []
        for i in range(self.num_robots):
            attempts = 0
            while attempts < 100:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if not self.grid.is_obstacle(x, y) and (x, y) not in positions:
                    positions.append((x, y))
                    # create robot normally for now
                    robot = Robot(i, x, y, self.grid, self.comm_system)
                    robot.simulation = self
                    self.robots.append(robot)
                    break
                attempts += 1
        # ✅ Ensure every robot has a pointer back to the simulation
        for r in self.robots:
            r.simulation = self

        # AFTER creating all robots, replace the highest-ID robot with malicious
        if self.robots:
            mal_id = max(r.robot_id for r in self.robots)
            for idx, r in enumerate(self.robots):
                # assign simulation pointer used by malicious (and useful for others)
                r.simulation = self
                if r.robot_id == mal_id:
                    attack_cfg = {'mode':'fixed_cells_delay', 'threshold': 10}  # adjust per-experiment
                    mal = MaliciousRobot(r.robot_id, r.x, r.y, self.grid, self.comm_system, attack_cfg=attack_cfg)
                    # keep any existing local-map/path if you want, or replace cleanly:
                    mal.path = r.path
                    mal.spanning_tree = r.spanning_tree
                    # set simulation reference
                    mal.simulation = self
                    self.robots[idx] = mal
                    self.malicious_id = mal.robot_id  # record id for convenience
                    break

    def _check_global_completion(self):
        """Check if all accessible cells have been covered"""
        all_covered = set()
        for robot in self.robots:
            all_covered.update(robot.local_map.covered_cells)

        total_free_cells = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.grid.is_obstacle(x, y):
                    total_free_cells += 1

        return len(all_covered) >= total_free_cells

    def step(self):
        """Execute one simulation step"""
        # Update global coverage for all robots
        for robot in self.robots:
            robot.update_global_coverage(self.robots)

        # Execute one step for each robot
        active_robots = 0
        for robot in self.robots:
            if robot.completed and not robot.checking_connections:
                continue
            if robot.step():
                active_robots += 1
            if getattr(robot, 'is_malicious', False):
                robot.malicious_tick()

        # --- ✅ Post-step collision resolution (order-independent) ---
        mal = None
        for r in self.robots:
            if getattr(r, 'is_malicious', False):
                mal = r
                break

        if mal is not None:
            for r in self.robots:
                if r is mal or not r.alive:
                    continue
                if (r.x, r.y) == (mal.x, mal.y) and not r.completed:
                    r.completed = True
                    r.checking_connections = False
                    r.returning_home = False

                    # metrics/logs
                    self.attack_metrics.setdefault('collisions', 0)
                    self.attack_metrics['collisions'] += 1
                    r.metrics = getattr(r, 'metrics', {})
                    r.metrics['collisions'] = r.metrics.get('collisions', 0) + 1

                    self.comm_system.broadcast_message(
                        r.robot_id, "collision", {"robot_id": r.robot_id, "at": (r.x, r.y)}
                    )
                    print(f"[COLLISION] Fair R{r.robot_id} stopped on Malicious R{mal.robot_id} at {(r.x, r.y)} (step {self.step_count})")

        # Only increment step count if robots are actually moving
        if active_robots > 0:
            self.step_count += 1

        self.comm_system.clear_old_messages()

        # Check completion
        all_truly_done = all(robot.completed and robot.checking_connections and
                           (robot.x, robot.y) == (robot.start_x, robot.start_y)
                           for robot in self.robots)

        all_covered = self._check_global_completion()

        return not (all_truly_done and all_covered)

    def update_visualization(self, frame):
        """Update function for animation"""
        # frame parameter required by matplotlib animation but not used
        if not self.running or self.step_count >= self.max_steps:
            return []

        if not self.step():
            self.running = False
            print(f"Simulation completed at step {self.step_count}")
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

        total_free_cells = np.sum(self.grid.grid != CellState.OBSTACLE.value)
        coverage_percentage = (len(all_covered) / total_free_cells) * 100
        completed_count = sum(1 for robot in self.robots if robot.completed)

        # Title with statistics
        self.ax.set_title(
            f'ORMSTC Multi-Robot Coverage Simulation\n'
            f'Step: {self.step_count} | Robots: {self.num_robots} | '
            f'Completed: {completed_count} | Coverage: {coverage_percentage:.1f}%',
            fontsize=14, fontweight='bold'
        )

        # Legend
        legend_elements = []
        for robot in self.robots:
            status = "✓" if robot.completed else "●"
            if robot.checking_connections:
                status = "🔍"
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

        if show_grid:
            print("Initial grid state:")
            self.grid.print_grid(self.robots)

        while self.step() and self.step_count < self.max_steps:
            if show_grid and self.step_count % 20 == 0:
                print(f"Step {self.step_count}:")
                self.grid.print_grid(self.robots)
            time.sleep(0.05)

        print(f"Simulation completed at step {self.step_count}")
        return self.get_results()

    def get_results(self):
        """Get simulation results"""
        results = {
            "total_steps": self.step_count,
            "robots_completed": sum(1 for robot in self.robots if robot.completed),
            "total_coverage": 0,
            "robot_paths": []
        }

        all_covered = set()
        for robot in self.robots:
            all_covered.update(robot.local_map.covered_cells)
            results["robot_paths"].append(robot.path)

        results["total_coverage"] = len(all_covered)
        results['attack_metrics'] = self.attack_metrics.copy()
        return results


def main():
    """Main function with improved interface"""
    print("ORMSTC Multi-Robot Coverage Simulation")
    print("=" * 50)

    # Get number of robots
    try:
        num_robots = int(input("Number of robots (1-10, default 3): ") or "3")
        num_robots = max(1, min(10, num_robots))
    except ValueError:
        num_robots = 3

    print("\nChoose simulation mode:")
    print("1. Animated simulation (GUI)")
    print("2. Console simulation (text-based)")

    try:
        choice = input("Enter choice (1-2): ").strip()

        # Default parameters
        grid_size = 16  # Default 16x16 as requested
        obstacle_density = 0.15

        # Set random seed for reproducible results
        random.seed(42)
        np.random.seed(42)

        print(f"\nSimulation settings:")
        print(f"  Grid: {grid_size}x{grid_size}")
        print(f"  Robots: {num_robots}")
        print(f"  Obstacles: {obstacle_density*100:.0f}%")

        if choice == "1":
            # Animated simulation
            sim = ORMSTCSimulation(grid_size, num_robots, obstacle_density, use_gui=True)
            results = sim.run_animated_simulation()

        elif choice == "2":
            # Console simulation
            sim = ORMSTCSimulation(grid_size, num_robots, obstacle_density, use_gui=False)
            results = sim.run_console_simulation(show_grid=True)

        else:
            print("Invalid choice!")
            return

        # Display final results
        print("\n" + "="*60)
        print("FINAL SIMULATION RESULTS")
        print("="*60)
        print(f"Total steps: {results['total_steps']}")
        print(f"Robots completed: {results['robots_completed']}/{num_robots}")
        print(f"Total coverage: {results['total_coverage']} cells")

        # Calculate coverage efficiency
        total_free_cells = grid_size * grid_size - int(grid_size * grid_size * obstacle_density)
        coverage_percentage = (results['total_coverage'] / total_free_cells) * 100
        print(f"Total free cells: {total_free_cells}")
        print(f"Coverage percentage: {coverage_percentage:.1f}%")

        if coverage_percentage >= 99.0:
            print("🎉 Excellent coverage achieved!")
        elif coverage_percentage >= 95.0:
            print("✅ Good coverage achieved!")

        print("\nRobot path lengths:")
        for i, path in enumerate(results['robot_paths']):
            print(f"  Robot {i}: {len(path)} steps")
        
        print("\nAttack metrics:")
        for k, v in results['attack_metrics'].items():
            print(f"  {k}: {v}")


    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()