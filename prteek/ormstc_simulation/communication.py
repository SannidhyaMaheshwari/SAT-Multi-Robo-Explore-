#!/usr/bin/env python3

import time
from .core import Message


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