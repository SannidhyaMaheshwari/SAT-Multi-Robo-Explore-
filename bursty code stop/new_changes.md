Summary of Changes

1. Grid Size Selection ✓
   Added option to choose between 16x16 grid (18 obstacles) or 50x50 grid (175 obstacles) in main.py
   User is prompted at the start of the simulation
2. Fixed Obstacles ✓
   Modified grid.py to use a fixed number of obstacles instead of density
   Obstacles are always generated with seed 42, ensuring they are fixed and reproducible
   18 obstacles for 16x16 grid, 175 obstacles for 50x50 grid
3. Random Robot Placement with Seed ✓
   Updated simulation.py to accept a robot_seed parameter
   Robot placement is now randomized based on the seed provided by the user
   Changing the seed will change robot positions, but obstacles remain fixed
   User is prompted to enter a seed (default: 42) in main.py
4. Position Broadcasting Protocol ✓
   Created new MessageType enum in core.py with POSITION_BROADCAST and POSITION_ACK message types
   Modified robot.py to implement position broadcasting:
   Each robot broadcasts its initial position at the start
   Other robots acknowledge received positions
   Robots wait for position exchanges before starting traversal
   Robots maintain a known_robot_positions dictionary learned from broadcasts
   Now robots communicate their positions through messages instead of having pre-knowledge of each other's locations! You can test the changes by running:
   python3 main.py

Perfect! Now robots will:
Broadcast their initial position as a message
Other robots will acknowledge when they receive it
Robots start traversal immediately without waiting for acknowledgments
They continuously learn about other robots' positions through message processing during traversal
The position broadcasting now happens in parallel with the traversal, not blocking the robot from starting its coverage task.
