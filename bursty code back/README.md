# ORMSTC Multi-Robot Coverage Simulation

This is a modularized implementation of the Optimized-Region-Based Multi-robot Spanning Tree Coverage (ORMSTC) algorithm for multi-robot coverage tasks.

## Project Structure

```
ormstc_simulation/
├── __init__.py          # Package initialization with conditional imports
├── core.py             # Core data structures (CellState, Direction, Message)
├── grid.py             # Grid environment representation
├── local_map.py        # Local mapping for individual robots
├── communication.py    # Communication system for inter-robot messaging
├── robot.py           # Robot implementation with ORMSTC algorithm
├── simulation.py      # Main simulation engine (requires matplotlib)
├── main.py           # Entry point for running simulations
└── README.md         # This file
```

## Module Descriptions

### core.py
- **CellState**: Enum for cell states (UNKNOWN, FREE, OBSTACLE, COVERED, ROBOT, TREE_EDGE)
- **Direction**: Enum for movement directions (RIGHT, DOWN, LEFT, UP) in clockwise order
- **Message**: Class for inter-robot communication messages

### grid.py
- **Grid**: Represents the environment grid with obstacles
  - Generates random obstacles based on density
  - Provides neighbor finding in clockwise order
  - Validates positions and checks for obstacles

### local_map.py
- **LocalMap**: Maintains robot's local knowledge of the environment
  - Tracks cell states and coverage
  - Manages spanning tree edges
  - Handles covered cells tracking

### communication.py
- **CommunicationSystem**: Manages inter-robot communication
  - Broadcasts messages between robots
  - Tracks robot states and activity
  - Implements failure detection mechanism

### robot.py
- **Robot**: Implements the ORMSTC algorithm
  - Performs environment sensing
  - Implements conflict resolution for cell claiming
  - Handles backtracking and spanning tree construction
  - Manages completion detection and connection monitoring

### simulation.py
- **ORMSTCSimulation**: Main simulation engine
  - Manages multiple robots
  - Provides both GUI and console simulation modes
  - Handles visualization and statistics
  - **Note**: Requires matplotlib for GUI functionality

### main.py
- Entry point with user interface for running simulations
- Supports both animated and console modes
- Configurable parameters (grid size, robot count, obstacle density)

## Usage

### Basic Usage (Core Components Only)
```python
from ormstc_simulation import Grid, Robot, CommunicationSystem
import random

# Set up environment
grid = Grid(16, 0.15)  # 16x16 grid with 15% obstacles
comm = CommunicationSystem()

# Create robot
robot = Robot(0, 5, 5, grid, comm)

# Run simulation steps
for i in range(100):
    if not robot.step():
        break
```

### Full Simulation (Requires matplotlib)
```python
from ormstc_simulation.simulation import ORMSTCSimulation

# Create and run simulation
sim = ORMSTCSimulation(
    grid_size=16,
    num_robots=3,
    obstacle_density=0.15,
    use_gui=True
)
results = sim.run_animated_simulation()
```

### Console Mode (No GUI dependencies)
```python
from ormstc_simulation.simulation import ORMSTCSimulation

sim = ORMSTCSimulation(
    grid_size=16,
    num_robots=3,
    obstacle_density=0.15,
    use_gui=False
)
results = sim.run_console_simulation()
```

## Algorithm Features

1. **Spanning Tree Coverage**: Robots build spanning trees to ensure complete coverage
2. **Conflict Resolution**: Higher robot ID wins when multiple robots compete for same cell
3. **Failure Handling**: Robots monitor connections and take over failed robots' areas
4. **Backtracking**: Robots backtrack when no forward moves are available
5. **Completion Detection**: Global coverage tracking and completion signaling
6. **Communication**: Message-based coordination between robots

## Dependencies

### Core Components (Always Available)
- numpy
- enum (built-in)
- time (built-in)
- random (built-in)

### GUI Components (Optional)
- matplotlib
- matplotlib.pyplot
- matplotlib.patches
- matplotlib.animation

## Running the Simulation

1. **With GUI** (requires matplotlib):
   ```bash
   python -m ormstc_simulation.main
   # or
   python run_simulation.py
   ```

2. **Console Only**:
   ```python
   from ormstc_simulation import Robot, Grid, CommunicationSystem
   # Use core components directly
   ```

## Key Algorithm Components

- **ORMSTC Algorithm**: Optimized region-based multi-robot spanning tree coverage
- **Tie-Breaking**: Deterministic conflict resolution using robot IDs
- **Global Coverage**: Shared knowledge of covered areas between robots
- **Connection Monitoring**: Post-completion monitoring of connected robots
- **Failure Recovery**: Automatic takeover of failed robots' coverage areas

## Verification

The implementation preserves all logic from the original monolithic version:
- ✅ Grid neighbor finding (clockwise order)
- ✅ Cell state management
- ✅ Robot behavior and algorithm logic
- ✅ Communication and message handling
- ✅ Spanning tree construction
- ✅ Conflict resolution mechanisms
- ✅ Completion detection and monitoring