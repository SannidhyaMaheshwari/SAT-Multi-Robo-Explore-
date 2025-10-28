# ORMSTC Data Collection Script

## Overview
Automated data collection script for research analysis of the ORMSTC multi-robot coverage algorithm.

## Test Configuration

### 16x16 Grid
- **Grid size**: 16×16
- **Obstacles**: 18 (fixed positions, seed=42)
- **Robot counts**: [5, 10, 15, 20, 25, 30, 35, 40]
- **Iterations per robot count**: 25
- **Seeds**: 1-25 (different robot placements)
- **Total tests**: 200

### 50x50 Grid
- **Grid size**: 50×50
- **Obstacles**: 175 (fixed positions, seed=42)
- **Robot counts**: [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
- **Iterations per robot count**: 25
- **Seeds**: 1-25 (different robot placements)
- **Total tests**: 250

**Grand Total**: 450 simulations

## How to Run

```bash
cd /path/to/ormstc_simulation
python3 data_collection.py
```

## Expected Runtime
- Approximately 30-90 minutes depending on system performance
- Progress bars will show real-time status
- Estimated time remaining is displayed

## Output Files

After completion, 4 CSV files will be created:

### 1. `summary_results.csv` (Combined Summary)
All grid sizes combined. Columns:
- `grid_size`: Grid dimension (16 or 50)
- `num_robots`: Number of robots
- `avg_coverage_time`: Average steps to 100% coverage
- `min_coverage_time`: Minimum coverage time
- `max_coverage_time`: Maximum coverage time
- `std_coverage_time`: Standard deviation
- `success_rate`: Percentage of successful runs
- `avg_total_steps`: Average total simulation steps
- `avg_coverage_pct`: Average coverage percentage achieved
- `avg_cells_per_robot`: Average cells covered per robot
- `avg_path_per_robot`: Average path length per robot

### 2. `16x16_summary.csv`
Summary statistics for 16×16 grid only (same columns as above)

### 3. `50x50_summary.csv`
Summary statistics for 50×50 grid only (same columns as above)

### 4. `detailed_results.csv` (All Individual Runs)
Every single test run. Columns:
- `grid_size`: Grid dimension
- `num_robots`: Number of robots
- `seed`: Random seed used
- `coverage_time`: Steps to 100% coverage
- `total_steps`: Total simulation steps
- `coverage_pct`: Coverage percentage achieved
- `robots_completed`: Number of robots that completed
- `avg_cells_per_robot`: Average cells per robot
- `avg_path_per_robot`: Average path length per robot
- `total_coverage`: Total cells covered

## Using Results in Excel

1. Open Excel
2. Go to **Data → From Text/CSV**
3. Select the CSV file
4. Click **Import**
5. Data will be formatted in columns automatically

Or simply:
1. Open the CSV file in Excel directly
2. Copy and paste into your analysis spreadsheet

## Metrics Explanation

### Coverage Time
- Number of steps required to achieve 100% coverage of free cells
- Does NOT include return journey time
- Lower is better (more efficient)

### Total Steps
- Total simulation steps including return journey
- Only counts exploration steps, not return steps

### Success Rate
- Percentage of runs that achieved 100% coverage within max_steps limit
- Should be 100% for properly configured tests

### Cells Per Robot
- Average number of unique cells covered by each robot
- Note: cells may be counted by multiple robots if overlapping

### Path Length Per Robot
- Total number of moves each robot made
- Includes exploration and backtracking

## Troubleshooting

### Script takes too long
- This is normal! 450 tests take time
- Each simulation runs up to max_steps (500 for 16×16, 3000 for 50×50)
- Progress bars show estimated time remaining

### Out of memory
- Close other applications
- Run tests for one grid size at a time by commenting out one config in `run_all_tests()`

### Results look wrong
- Check that obstacles are fixed (seed=42)
- Verify robot placement seeds are different (1-25)
- Check max_steps limits in simulation.py

## Contact
For issues or questions about the data collection script, check the main simulation code or contact the research team.
