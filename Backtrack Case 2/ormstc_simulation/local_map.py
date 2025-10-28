#!/usr/bin/env python3

import numpy as np
from .core import CellState


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
    
    def get_tree_owner(self, x, y):
        return self.tree_edges.get((x, y))

    def remove_tree_edge(self, x, y):
        """Remove local DFS-ownership mark (does NOT change global coverage)."""
        if (x, y) in self.tree_edges:
            del self.tree_edges[(x, y)]
            # If the cell was previously drawn as TREE_EDGE, render it as COVERED now.
            self.update_cell(x, y, CellState.COVERED)
