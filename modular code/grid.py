#!/usr/bin/env python3

import numpy as np
import random
from .core import CellState, Direction


class Grid:
    def __init__(self, size=16, num_obstacles=18):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.num_obstacles = num_obstacles
        self._generate_obstacles()

    def _generate_obstacles(self):
        """Generate fixed number of obstacles using seed 42 for reproducibility"""
        # Use fixed seed for obstacle generation (always the same obstacles)
        random.seed(42)
        np.random.seed(42)

        placed = 0
        attempts = 0
        max_attempts = self.num_obstacles * 100

        while placed < self.num_obstacles and attempts < max_attempts:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if self.grid[y, x] != CellState.OBSTACLE.value:
                self.grid[y, x] = CellState.OBSTACLE.value
                placed += 1
            attempts += 1

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