#!/usr/bin/env python3
"""
ORMSTC Multi-Robot Coverage Simulation Package

This package implements the Optimized-Region-Based Multi-robot Spanning Tree Coverage (ORMSTC)
algorithm for multi-robot coverage tasks.

Modules:
    - core: Core data structures (enums and messages)
    - grid: Grid environment representation
    - local_map: Local mapping for robots
    - communication: Communication system for robots
    - robot: Robot implementation with ORMSTC algorithm
    - simulation: Main simulation engine
    - main: Entry point for running simulations
"""

from .core import CellState, Direction, Message, MessageType
from .grid import Grid
from .local_map import LocalMap
from .communication import CommunicationSystem
from .robot import Robot



# Import simulation conditionally (requires matplotlib)
try:
    from .simulation import ORMSTCSimulation
    __all__ = [
        'CellState', 'Direction', 'Message', 'MessageType',
        'Grid', 'LocalMap', 'CommunicationSystem',
        'Robot', 'ORMSTCSimulation'
    ]
except ImportError:
    __all__ = [
        'CellState', 'Direction', 'Message', 'MessageType',
        'Grid', 'LocalMap', 'CommunicationSystem',
        'Robot'
    ]

__version__ = "1.0.0"
__author__ = "Multi-Robot Systems Team"