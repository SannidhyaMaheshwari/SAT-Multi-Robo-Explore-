#!/usr/bin/env python3

from .core import CellState, MessageType
from .local_map import LocalMap


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

        # Position broadcasting state
        self.known_robot_positions = {}  # {robot_id: (x, y)}
        self.position_broadcasted = False
        self.position_acks = set()  # Set of robot IDs that acknowledged our position

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
        
        # NEW (Step 3): report/queue the start cell via comms
        self.comm_system.publish_coverage(self.robot_id, (start_x, start_y))
        # NEW (Step 7): track the last flush epoch we've observed
        self.last_seen_flush_epoch = self.comm_system.flush_epoch
        self.overlap_candidates = set()      # NEW: cells we stepped on that were hidden by attacker at that moment
        self.last_seen_flush_epoch = self.comm_system.flush_epoch  # already added earlier; keep it
        self.flush_replan_pending = False    # ok to keep from earlier step, not strictly required here
        self.stop_in_place = False           # NEW: when true, robot never moves again
        self.stalled_due_to_overlap = False  # NEW: label for metrics

        # Backtrack-after-overlap state (NEW)
        self.backtracking = False
        self.replans_after_overlap = 0
        self.overlap_candidates = set()
        self.suppress_backtrack_once = False


    def broadcast_initial_position(self):
        """Broadcast initial position to other robots"""
        if not self.position_broadcasted:
            self.comm_system.broadcast_message(
                self.robot_id,
                MessageType.POSITION_BROADCAST.value,
                {
                    "robot_id": self.robot_id,
                    "position": (self.start_x, self.start_y)
                }
            )
            self.position_broadcasted = True

    def acknowledge_position(self, sender_id):
        """Send acknowledgment for received position broadcast"""
        self.comm_system.broadcast_message(
            self.robot_id,
            MessageType.POSITION_ACK.value,
            {
                "acknowledging": sender_id,
                "from_robot": self.robot_id
            }
        )

    def update_global_coverage(self, all_robots):
        """Update global coverage knowledge from all robots"""
        self.global_coverage.clear()
        self.global_coverage.update(self.comm_system.get_global_reported_union())

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
        """Move to target cell with conflict resolution"""
        if target_x is None or target_y is None:
            return False

        is_mal = self.comm_system.is_malicious(self.robot_id)

        # Broadcast claim before moving
        if not is_mal:
            self.comm_system.broadcast_message(
                self.robot_id,
                "claiming",
                {"target": (target_x, target_y), "position": (self.x, self.y)}
            )

        # Add tree edge
        self.local_map.add_tree_edge(target_x, target_y, self.robot_id)
        self.spanning_tree[(target_x, target_y)] = (self.x, self.y)

        # Move robot
        self.x, self.y = target_x, target_y
        self.path.append((self.x, self.y))
        self.local_map.mark_covered(self.x, self.y)

        # Track cells visited since the last flush (fair robots only)
        if not is_mal:
            self.overlap_candidates.add((self.x, self.y))

        # NEW: report/queue the newly covered cell via comms (fair → reported, malicious → hidden)
        self.comm_system.publish_coverage(self.robot_id, (self.x, self.y))

         # NEW (Step 4): if I'm the malicious robot, check triggers and flush if needed
        if is_mal:
            # cell-based trigger (e.g., flush when reaching (5, 2))
            if self.comm_system._should_flush_on_cell((self.x, self.y)) or self.comm_system._should_flush_on_batch():
                self.comm_system.flush_hidden()

        # Broadcast move
        move_payload = {"position": (self.x, self.y)}
        if not is_mal:
            move_payload["tree_edge"] = (target_x, target_y)
        
        self.comm_system.broadcast_message(
            self.robot_id,
            "move", move_payload
        )

        self.comm_system.update_robot_state(self.robot_id, {
            "position": (self.x, self.y)
        })

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
            if (nx1, ny1) in self.local_map.covered_cells or self.grid.is_obstacle(nx1, ny1):
                continue

            # Check if this cell has an adjacent free cell (forming a hill)
            hill_neighbors = self.grid.get_neighbors(nx1, ny1)
            for nx2, ny2 in hill_neighbors:
                if ((nx2, ny2) not in self.local_map.covered_cells and
                    not self.grid.is_obstacle(nx2, ny2) and
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
            if message.msg_type == MessageType.POSITION_BROADCAST.value:
                # Another robot is broadcasting its position
                data = message.data
                sender_id = message.sender_id
                position = data.get("position")

                if sender_id not in self.known_robot_positions:
                    self.known_robot_positions[sender_id] = position
                    # Send acknowledgment
                    self.acknowledge_position(sender_id)

            elif message.msg_type == MessageType.POSITION_ACK.value:
                # Another robot acknowledged our position
                data = message.data
                if data.get("acknowledging") == self.robot_id:
                    from_robot = data.get("from_robot")
                    self.position_acks.add(from_robot)

            elif message.msg_type == "move":
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
        # Local completion (others may still be exploring)
        return True

    def step(self):
        """Execute one step of the ORMSTC algorithm"""
        if not self.alive:
            return False

        # # NEW: if we decided to stop in place (overlap realized), do nothing forever
        # if self.completed and getattr(self, "stop_in_place", False):
        #     return False
        
        # First, broadcast position if not done yet (non-blocking)
        if not self.position_broadcasted:
            self.broadcast_initial_position()

        # Process messages to learn about other robots (continuously)
        self.process_messages()

        # One-tick anti-thrash latch
        suppress_now = False
        if getattr(self, "suppress_backtrack_once", False):
            suppress_now = True
            self.suppress_backtrack_once = False


        # If a flush happened since last tick, check if it revealed any cell we overlapped earlier.
        if not self.comm_system.is_malicious(self.robot_id):
            if self.comm_system.flush_epoch > self.last_seen_flush_epoch:
                self.last_seen_flush_epoch = self.comm_system.flush_epoch
                flushed = self.comm_system.get_last_flushed_cells()

                # If any of our previously overlapped cells is in this flushed batch -> stop in place
                flushed_set = set(flushed)  # harmless if already a set
                overlap = self.overlap_candidates & flushed_set
                if overlap and not suppress_now:
                    # self.completed = True
                    self.backtracking = True
                    self.replans_after_overlap += 1
                    self.overlap_candidates.clear()
                    # self.stop_in_place = True
                    # self.stalled_due_to_overlap = True
                    # Optional: announce why we stopped
                    # self.comm_system.broadcast_message(
                    #     self.robot_id, "completed",
                    #     {"robot_id": self.robot_id, "reason": "overlap_revealed"}
                    # )
                    return False  # do not move this tick (and never again)
                self.overlap_candidates.clear()

        # --- Backtrack-to-Replan mode (NEW) ---
        if self.backtracking and not self.completed:
            # Keep perception fresh before deciding
            self.sense_environment()

            # Try to resume exploration from here (post-flush coverage is already in self.global_coverage)
            nx, ny = self.find_next_cell()
            if nx is not None and ny is not None:
                # Found a valid frontier; exit backtracking mode and move forward
                self.backtracking = False
                self.suppress_backtrack_once = True
                return self.move_to_cell(nx, ny)

            # No frontier here → step one parent up along the DFS tree
            if self.backtrack():
                # We moved one step toward parent; next tick we'll check again
                return True

            # Can't backtrack further (likely at root). If no frontier from root, finish locally.
            if (self.x, self.y) == (self.start_x, self.start_y) and self.check_completion():
                self.completed = True
                self.checking_connections = True   # same semantics as your "done at start" branch
                self.returning_home = False
                self.comm_system.broadcast_message(
                    self.robot_id, "completed",
                    {"robot_id": self.robot_id, "reason": "local_done_after_backtrack"}
                )
                return False  # idle at start while connection-check logic runs

            # Nothing to do this tick (stuck but not at start yet); wait for next tick
            return False

        # If completed and checking connections
        if self.completed and self.checking_connections:
            if (self.x, self.y) == (self.start_x, self.start_y):
                # Check for failed robots
                if self.check_failed_robots():
                    return True  # Found failed robot, resume coverage
                else:
                    # All connections checked, truly done - not active anymore
                    return False
            else:
                # Move towards start to check connections
                return self.move_towards_start()

        # If completed but need to return home
        if self.completed and not self.checking_connections:
            if (self.x, self.y) == (self.start_x, self.start_y):
                # Reached start, now check connections - not moving anymore
                self.checking_connections = True
                return False  # Changed from True to False - not actively moving
            else:
                # Return to starting position
                self.returning_home = True
                return self.move_towards_start()

        # Normal operation (process_messages already called at the beginning of step)
        self.sense_environment()

        # Try to find next cell
        next_x, next_y = self.find_next_cell()

        if next_x is not None and next_y is not None:
            # Move forward
            return self.move_to_cell(next_x, next_y)
        else:
            # Try to backtrack
            if self.backtrack():
                return True
            else:
                # ---- Key fix: if we're already at START and have no frontier, finish here ----
                if (self.x, self.y) == (self.start_x, self.start_y):
                    # No frontier from start → immediately completed (no return-home needed)
                    self.completed = True
                    self.checking_connections = True
                    self.returning_home = False
                    self.comm_system.broadcast_message(
                        self.robot_id, "completed",
                        {"robot_id": self.robot_id, "reason": "local_done_at_start"}
                    )
                    return False  # idle

                # General local-complete case (away from start → begin return-home)
                if self.check_completion():
                    self.completed = True
                    self.returning_home = True
                    self.comm_system.broadcast_message(
                        self.robot_id, "completed", {"robot_id": self.robot_id}
                    )
                    return True  # will start heading home next ticks
                
                return False