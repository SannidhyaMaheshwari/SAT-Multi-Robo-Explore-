#!/usr/bin/env python3

from enum import Enum


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

class MessageType(Enum):
    """Message types for robot communication"""
    POSITION_BROADCAST = "position_broadcast"
    POSITION_ACK = "position_ack"
    CLAIMING = "claiming"
    MOVE = "move"
    COMPLETED = "completed"
    TAKEOVER = "takeover"
    WITHDRAWAL = "withdrawal"

    # --- NEW for Reliable Backtrack Protocol ---
    BACKTRACK_START = "backtrack_start"
    EDGE_REMOVED_ACK = "edge_removed_ack"
    BACKTRACK_COMPLETE = "backtrack_complete"


class Message:
    def __init__(self, sender_id, msg_type, data, timestamp):
        self.sender_id = sender_id
        self.msg_type = msg_type
        self.data = data
        self.timestamp = timestamp