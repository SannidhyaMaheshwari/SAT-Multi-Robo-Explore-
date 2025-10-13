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
    POSITION_BROADCAST = "position_broadcast"  # Robot broadcasts its initial position
    POSITION_ACK = "position_ack"  # Robot acknowledges another robot's position
    CLAIMING = "claiming"  # Robot claims a cell
    MOVE = "move"  # Robot has moved
    COMPLETED = "completed"  # Robot has completed its coverage
    TAKEOVER = "takeover"  # Robot is taking over failed robot's area
    WITHDRAWAL = "withdrawal"  # Robot withdraws completion


class Message:
    def __init__(self, sender_id, msg_type, data, timestamp):
        self.sender_id = sender_id
        self.msg_type = msg_type
        self.data = data
        self.timestamp = timestamp