#!/usr/bin/env python3

import time
from .core import Message
from .core import MessageType


class CommunicationSystem:
    def __init__(self):
        self.messages = []
        self.robot_states = {}
        self.active_robots = set()
        self.malicious_id = None
        
        # Officially known coverage per robot (what's been reported/broadcast)
        self.reported_coverage = {}         # Dict[int, Set[tuple[int,int]]]
        # Buffered coverage for the malicious robot (not yet shared)
        self.hidden_coverage = {}           # Dict[int, Set[tuple[int,int]]]
        # A monotonic counter to let robots detect a recent flush (we'll use it in Step 8)
        self.flush_epoch = 0
        # Simple attack metrics
        self.flush_count = 0
        self.delayed_cells_count = 0
        self.delayed_messages_count = 0     # if you count per-cell as a “message”, you can increment both

        # NEW: configurable flush triggers
        self.flush_trigger_cell = None      # e.g., (5, 2)
        self.flush_batch_size = None        # e.g., 10
        self.last_flushed_cells = set()


    # --- NEW (Step 2) tiny helpers ---
    def _ensure_robot_slots(self, robot_id: int):
        """Lazily create per-robot coverage sets."""
        if robot_id not in self.reported_coverage:
            self.reported_coverage[robot_id] = set()
        if robot_id not in self.hidden_coverage:
            self.hidden_coverage[robot_id] = set()

    def register_robot(self, robot_id):
        self.active_robots.add(robot_id)
        self.robot_states[robot_id] = {
            "alive": True,
            "position": None,
            "last_activity": time.time()
        }
        self._ensure_robot_slots(robot_id)
    
    # Mark a robot as malicious
    def set_malicious_robot(self, robot_id: int):
        self.malicious_id = robot_id
    
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

     # Helper to query maliciousness
    def is_malicious(self, robot_id: int) -> bool:
        return self.malicious_id is not None and robot_id == self.malicious_id
    
    # --- NEW (Step 2) main API the robots will use in Step 3 ---
    def publish_coverage(self, robot_id: int, cell: tuple[int, int]):
        """
        Called by a robot AFTER it enters a cell.
        Fair robots: immediately add to reported coverage.
        Malicious robot: buffer the cell in hidden coverage (delay).
        """
        self._ensure_robot_slots(robot_id)

        # Touch liveness timestamp (reusing your state)
        state = self.robot_states.get(robot_id)
        if state is not None:
            state["last_activity"] = time.time()

        if self.is_malicious(robot_id):
            # Buffer instead of reporting
            if cell not in self.hidden_coverage[robot_id]:
                self.hidden_coverage[robot_id].add(cell)
                self.delayed_cells_count += 1
                self.delayed_messages_count += 1
        else:
            # Share immediately
            self.reported_coverage[robot_id].add(cell)

        # --- NEW (Step 2) helper getters, used later by robots/simulator ---
    def get_robot_reported(self, robot_id: int):
        self._ensure_robot_slots(robot_id)
        return self.reported_coverage[robot_id]

    def get_hidden_for_malicious(self):
        """Convenience accessor for the attacker’s hidden set."""
        if self.malicious_id is None:
            return set()
        self._ensure_robot_slots(self.malicious_id)
        return self.hidden_coverage[self.malicious_id]

    def get_global_reported_union(self):
        """Union of all robots' reported coverage (the only 'truth' fair robots should use)."""
        union = set()
        for s in self.reported_coverage.values():
            union |= s
        return union
    
    # --- NEW (Step 2) flushing API (we'll trigger this in a later step) ---
    def flush_hidden(self) -> int:
        """
        Move all hidden cells of the malicious robot into reported coverage,
        bump metrics, and advance a 'flush epoch' that fair robots can see.
        Returns the number of cells flushed.
        """
        if self.malicious_id is None:
            return 0
        self._ensure_robot_slots(self.malicious_id)

        hidden = self.hidden_coverage[self.malicious_id]
        if not hidden:
            return 0

        # Record which cells are being flushed this epoch
        self.last_flushed_cells = set(hidden)          # NEW  <<<

        # Move hidden -> reported
        self.reported_coverage[self.malicious_id] |= hidden
        flushed = len(hidden)

        # Clear buffer
        self.hidden_coverage[self.malicious_id].clear()

        # Update metrics and notify via epoch
        self.flush_count += 1
        self.flush_epoch += 1
        return flushed
    
        # --- configure triggers ---
    def set_flush_trigger_cell(self, cell):
        """Set a coordinate (x,y). When attacker enters this cell, flush."""
        self.flush_trigger_cell = cell

    def set_flush_batch_size(self, k):
        """Set an integer K. When hidden buffer reaches K cells, flush."""
        self.flush_batch_size = k

    # --- helpers used by robot.py to decide when to flush ---
    def _should_flush_on_cell(self, pos):
        return (self.malicious_id is not None
                and self.flush_trigger_cell is not None
                and pos == self.flush_trigger_cell)

    def _should_flush_on_batch(self):
        if self.malicious_id is None or self.flush_batch_size is None:
            return False
        return len(self.get_hidden_for_malicious()) >= int(self.flush_batch_size)
    
    def get_last_flushed_cells(self):
        """Cells that were made public in the most recent flush."""
        return self.last_flushed_cells

    def send_edge_removed_ack(self, sender_id, bt_epoch):
        """Send ACK to confirm edge removal during backtrack"""
        self.broadcast_message(sender_id, MessageType.EDGE_REMOVED_ACK.value, {"bt_epoch": bt_epoch})
