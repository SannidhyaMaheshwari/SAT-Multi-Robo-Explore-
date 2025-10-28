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
        self.final_comp = False
        self.returning_home = False
        self.checking_connections = False
        self.path = [(start_x, start_y)]
        self.spanning_tree = {}
        self.connections = {}
        self.global_coverage = set()

        # Position broadcast
        self.known_robot_positions = {}
        self.position_broadcasted = False
        self.position_acks = set()

        # Color
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.color = colors[robot_id % len(colors)]

        # Init position
        self.local_map.update_cell(start_x, start_y, CellState.FREE)
        self.local_map.mark_covered(start_x, start_y)
        self.comm_system.register_robot(robot_id)
        self.comm_system.update_robot_state(robot_id, {"position": (start_x, start_y), "alive": True})
        self.comm_system.publish_coverage(self.robot_id, (start_x, start_y))

        # Tracking flush
        self.last_seen_flush_epoch = self.comm_system.flush_epoch
        self.overlap_candidates = set()
        self.flush_replan_pending = False
        self.stop_in_place = False
        self.stalled_due_to_overlap = False

        # --- New backtrack attributes ---
        self.backtracking = False
        self.bt_epoch = 0
        self.bt_target = None
        self.bt_path = []
        self.bt_pending_acks = set()
        self.bt_ack_received = set()
        self.bt_wait_ticks = 0
        self.ACK_TIMEOUT = 10
        self.ACK_RETRIES = 2
        self.bt_retries = 0

    # -------------------------------------------------
    # Normal functions (unchanged)
    # -------------------------------------------------

    def broadcast_initial_position(self):
        if not self.position_broadcasted:
            self.comm_system.broadcast_message(
                self.robot_id, MessageType.POSITION_BROADCAST.value,
                {"robot_id": self.robot_id, "position": (self.start_x, self.start_y)}
            )
            self.position_broadcasted = True

    def acknowledge_position(self, sender_id):
        self.comm_system.broadcast_message(
            self.robot_id, MessageType.POSITION_ACK.value,
            {"acknowledging": sender_id, "from_robot": self.robot_id}
        )

    def update_global_coverage(self, all_robots):
        self.global_coverage.clear()
        self.global_coverage.update(self.comm_system.get_global_reported_union())

    def is_globally_complete(self):
        total_free_cells = 0
        for y in range(self.grid.size):
            for x in range(self.grid.size):
                if not self.grid.is_obstacle(x, y):
                    total_free_cells += 1
        return len(self.global_coverage) >= total_free_cells

    def sense_environment(self):
        neighbors = self.grid.get_neighbors(self.x, self.y)
        for nx, ny in neighbors:
            if self.grid.is_obstacle(nx, ny):
                self.local_map.update_cell(nx, ny, CellState.OBSTACLE)
            elif self.local_map.get_cell(nx, ny) == CellState.UNKNOWN:
                self.local_map.update_cell(nx, ny, CellState.FREE)
        return True

    def _check_cell_conflict(self, x, y):
        messages = self.comm_system.get_messages_for_robot(self.robot_id)
        conflicting_robots = []
        for message in messages:
            if message.msg_type == "claiming" and message.data.get("target") == (x, y):
                conflicting_robots.append(message.sender_id)
        return max(conflicting_robots) if conflicting_robots else None

    def find_next_cell(self):
        if self.returning_home or self.checking_connections:
            return None, None
        neighbors = self.grid.get_neighbors(self.x, self.y)
        for nx, ny in neighbors:
            cell_state = self.local_map.get_cell(nx, ny)
            if cell_state != CellState.FREE or (nx, ny) in self.local_map.covered_cells:
                continue
            if (nx, ny) in self.global_coverage:
                continue
            conflict = self._check_cell_conflict(nx, ny)
            if conflict is not None and conflict > self.robot_id:
                continue
            return nx, ny
        return None, None

    def move_to_cell(self, tx, ty):
        if tx is None or ty is None:
            return False
        is_mal = self.comm_system.is_malicious(self.robot_id)
        if not is_mal:
            self.comm_system.broadcast_message(
                self.robot_id, "claiming", {"target": (tx, ty), "position": (self.x, self.y)}
            )
        self.local_map.add_tree_edge(tx, ty, self.robot_id)
        self.spanning_tree[(tx, ty)] = (self.x, self.y)
        self.x, self.y = tx, ty
        self.path.append((tx, ty))
        self.local_map.mark_covered(tx, ty)
        self.comm_system.publish_coverage(self.robot_id, (tx, ty))
        if is_mal and (self.comm_system._should_flush_on_cell((tx, ty)) or self.comm_system._should_flush_on_batch()):
            self.comm_system.flush_hidden()
        move_payload = {"position": (self.x, self.y)}
        if not is_mal:
            move_payload["tree_edge"] = (tx, ty)
        self.comm_system.broadcast_message(self.robot_id, "move", move_payload)
        self.comm_system.update_robot_state(self.robot_id, {"position": (self.x, self.y)})
        return True

    def backtrack(self):
        if (self.x, self.y) in self.spanning_tree:
            self.x, self.y = self.spanning_tree[(self.x, self.y)]
            return True
        return False

    # -------------------------------------------------
    # Backtrack message handling and step logic
    # -------------------------------------------------

    def process_messages(self):
        messages = self.comm_system.get_messages_for_robot(self.robot_id)
        for message in messages:
            mtype, data = message.msg_type, message.data
            if mtype == MessageType.POSITION_BROADCAST.value:
                sender = message.sender_id
                if sender not in self.known_robot_positions:
                    self.known_robot_positions[sender] = data.get("position")
                    self.acknowledge_position(sender)
            elif mtype == MessageType.MOVE.value:
                edge = data.get("tree_edge")
                if edge:
                    self.local_map.add_tree_edge(edge[0], edge[1], message.sender_id)
            elif mtype == MessageType.BACKTRACK_START.value:
                edges = data.get("edges", [])
                removed = 0
                for child, parent in edges:
                    if child in self.spanning_tree and self.spanning_tree[child] == parent:
                        del self.spanning_tree[child]
                        removed += 1
                self.comm_system.send_edge_removed_ack(self.robot_id, data.get("bt_epoch"))
                print(f"[Robot {self.robot_id}] Removed {removed} edges due to BACKTRACK_START from Robot {message.sender_id}.")
                # ensure normal robots keep exploring
                if self.backtracking and message.sender_id != self.robot_id:
                    self.backtracking = False
            elif mtype == MessageType.EDGE_REMOVED_ACK.value:
                if self.backtracking and data.get("bt_epoch") == self.bt_epoch:
                    self.bt_ack_received.add(message.sender_id)

    def initiate_backtrack(self, flushed_cells):
        if self.backtracking:
            return
        safe = None
        for cell in reversed(self.path):
            if cell not in flushed_cells:
                safe = cell
                break
        if safe is None:
            self.stop_in_place = True
            return
        self.bt_target = safe
        self.bt_path = []
        cur = (self.x, self.y)
        while cur in self.spanning_tree:
            parent = self.spanning_tree[cur]
            self.bt_path.append(cur)
            if parent == safe:
                self.bt_path.append(parent)
                break
            cur = parent
        self.bt_epoch += 1
        self.bt_pending_acks = set(self.comm_system.robot_states.keys()) - {self.robot_id}
        self.bt_ack_received.clear()
        self.bt_wait_ticks = 0
        self.bt_retries = 0
        self.backtracking = True
        edges = [(c, self.spanning_tree.get(c)) for c in self.bt_path if c in self.spanning_tree]
        self.comm_system.broadcast_message(
            self.robot_id, MessageType.BACKTRACK_START.value,
            {"bt_epoch": self.bt_epoch, "bt_target": self.bt_target, "edges": edges}
        )
        print(f"[Robot {self.robot_id}] Initiating backtrack to {self.bt_target} with {len(edges)} edges.")

    def perform_backtrack_step(self):
        if self.bt_wait_ticks < self.ACK_TIMEOUT and len(self.bt_ack_received) < max(1, len(self.bt_pending_acks)//2 + 1):
            self.bt_wait_ticks += 1
            return
        if len(self.bt_ack_received) < max(1, len(self.bt_pending_acks)//2 + 1):
            if self.bt_retries < self.ACK_RETRIES:
                self.bt_retries += 1
                self.bt_wait_ticks = 0
                self.comm_system.broadcast_message(
                    self.robot_id, MessageType.BACKTRACK_START.value,
                    {"bt_epoch": self.bt_epoch, "bt_target": self.bt_target,
                     "edges": [(c, self.spanning_tree.get(c)) for c in self.bt_path if c in self.spanning_tree]}
                )
                print(f"[Robot {self.robot_id}] Retry {self.bt_retries}: rebroadcasting BACKTRACK_START.")
                return
            else:
                print(f"[Robot {self.robot_id}] Timeout waiting for ACKs, proceeding anyway.")
        if self.bt_path:
            nxt = self.bt_path.pop(0)
            self.spanning_tree.pop((self.x, self.y), None)
            self.x, self.y = nxt
            self.local_map.mark_covered(self.x, self.y)
            print(f"[Robot {self.robot_id}] Backtracking step to {nxt}.")
        else:
            print(f"[Robot {self.robot_id}] Backtrack complete at {self.bt_target}.")
            self.backtracking = False
            self.bt_target = None
            self.bt_path = []
            self.comm_system.broadcast_message(
                self.robot_id, MessageType.BACKTRACK_COMPLETE.value,
                {"bt_epoch": self.bt_epoch}
            )

    def step(self):
        if self.backtracking:
            self.perform_backtrack_step()
            return False
        if not self.alive:
            return False
        if self.completed and self.stop_in_place:
            return False
        if not self.position_broadcasted:
            self.broadcast_initial_position()
        self.process_messages()
        if not self.backtracking and not self.completed:
            self.stop_in_place = False
        if not self.comm_system.is_malicious(self.robot_id):
            if self.comm_system.flush_epoch > self.last_seen_flush_epoch:
                self.last_seen_flush_epoch = self.comm_system.flush_epoch
                flushed = self.comm_system.get_last_flushed_cells()
                if self.overlap_candidates and (self.overlap_candidates & flushed):
                    self.initiate_backtrack(flushed)
                    return False
        self.sense_environment()
        nx, ny = self.find_next_cell()
        if nx is not None:
            return self.move_to_cell(nx, ny)
        else:
            if self.backtrack():
                return True
            return False
