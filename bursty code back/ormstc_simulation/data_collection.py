#!/usr/bin/env python3
"""
data_collection.py

Robust data collection script for ORMSTC simulations.

Features / fixes included:
 - Forces matplotlib 'Agg' backend to avoid GUI bitmap allocation issues
 - Closes figures and runs gc.collect() after each simulation run
 - Safe-run wrapper that retries once on bitmap/memory-like errors
 - Defensive constructor calling for ORMSTCSimulation (supports malicious_robot, robot_seed)
 - Writes separate CSVs for 16x16 and 50x50 experiments
 - Default behavior runs only the 50x50 experiments; change RUN_MODE at top to run both/all
"""

import os
import csv
import random
import time
import traceback
import gc
from datetime import datetime

# Force non-interactive matplotlib backend BEFORE importing pyplot (prevents GDI/bitmap issues)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

# Import ORMSTCSimulation (adjust import depending on your package layout)
try:
    from ormstc_simulation.simulation import ORMSTCSimulation
except Exception:
    try:
        from simulation import ORMSTCSimulation
    except Exception as e:
        raise ImportError("Could not import ORMSTCSimulation from simulation.py") from e

# -------------------------
# Configuration
# -------------------------
# RUN_MODE: 'both' -> run 16x16 and 50x50,
#           '50'   -> run only 50x50,
#           '16'   -> run only 16x16
# Default set to '50' per your last request (change to 'both' to run all)
RUN_MODE = '50'

# Seeds to run (1..25)
SEEDS = list(range(1, 26))

# Output filenames
OUT_16 = "detailed_results_16x16.csv"
OUT_50 = "detailed_results_50x50.csv"

# Configs: (grid_size, obstacles_count, robot_counts_list)
CONFIG_16 = (16, 18, [5, 10, 15, 20, 25, 30, 35, 40], OUT_16)
CONFIG_50 = (50, 175, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], OUT_50)

# If you'd like to run a smaller subset for quick testing, change these lists.
# Example for quick test: CONFIG_50 = (50, 175, [50, 100], OUT_50)

# -------------------------
# Utilities
# -------------------------
def safe_get_reported_union(sim):
    try:
        if hasattr(sim.comm_system, "get_global_reported_union"):
            return set(sim.comm_system.get_global_reported_union())
        # fallback: maybe sim has reported_union attribute
        if hasattr(sim, "reported_union"):
            return set(sim.reported_union)
    except Exception:
        pass
    return set()

def count_free_cells(grid):
    try:
        total = 0
        for y in range(grid.size):
            for x in range(grid.size):
                if not grid.is_obstacle(x, y):
                    total += 1
        return total
    except Exception:
        return None

# -------------------------
# Run single simulation (defensive)
# -------------------------
def run_single_sim(grid_size, obstacles, num_robots, seed):
    """
    Construct and run a single ORMSTCSimulation, headless.
    Attempts the constructor signature:
      ORMSTCSimulation(grid_size, num_robots, obstacles, malicious_robot=0, robot_seed=seed, use_gui=False)
    Falls back to a few other likely signatures if needed.
    """
    # Keep obstacle generation deterministic like main.py
    random.seed(42)
    np.random.seed(42)

    sim = None
    last_exc = None
    # Try common constructor variants
    try:
        sim = ORMSTCSimulation(
            grid_size,
            num_robots,
            obstacles,
            malicious_robot=0,
            robot_seed=seed,
            use_gui=False
        )
    except TypeError as te:
        last_exc = te
        # Fallbacks
        try:
            sim = ORMSTCSimulation(grid_size, num_robots, obstacles, robot_seed=seed, use_gui=False)
        except Exception:
            try:
                sim = ORMSTCSimulation(grid_size, num_robots, obstacles, robot_seed=seed)
            except Exception:
                try:
                    sim = ORMSTCSimulation(grid_size, num_robots, obstacles)
                except Exception as e:
                    last_exc = e
                    raise RuntimeError("Failed to construct ORMSTCSimulation; check signature") from e
    except Exception as e:
        last_exc = e
        raise

    # If sim exposes set_seed or robot_seed attr, set it defensively
    try:
        if hasattr(sim, "robot_seed"):
            setattr(sim, "robot_seed", seed)
        if hasattr(sim, "set_seed"):
            try:
                sim.set_seed(seed)
            except Exception:
                pass
    except Exception:
        pass

    # Run simulation (prefer provided console runner)
    if hasattr(sim, "run_console_simulation"):
        try:
            sim.run_console_simulation(show_grid=False)
        except TypeError:
            # older signature may not accept show_grid
            sim.run_console_simulation()
    else:
        # fallback: step loop
        max_steps = getattr(sim, "max_steps", 3000)
        steps = 0
        active = True
        while active and steps < max_steps:
            active = sim.step()
            steps += 1
        sim.step_count = steps

    return sim

# -------------------------
# Safe-run wrapper with retry for bitmap/memory-like issues
# -------------------------
def safe_run_and_collect(grid_size, obstacles, num_robots, seed):
    """
    Run run_single_sim and handle bitmap/GDI/memory allocation errors gracefully.
    Returns (sim_object, error). If error is not None, sim_object is None.
    """
    try:
        sim = run_single_sim(grid_size, obstacles, num_robots, seed)
        # cleanup graphics objects (if any)
        try:
            plt.close("all")
        except Exception:
            pass
        gc.collect()
        return sim, None
    except Exception as e:
        msg = str(e).lower()
        # Heuristic match for graphics/memory related failures
        graphics_issue = False
        if isinstance(e, MemoryError):
            graphics_issue = True
        elif "bitmap" in msg or "gdi" in msg or "could not allocate" in msg or "out of memory" in msg:
            graphics_issue = True

        if graphics_issue:
            # retry once after forcing Agg backend and GC
            try:
                print("\nWarning: graphics/memory error detected. Retrying once with Agg backend and GC...")
                matplotlib.use("Agg")
                gc.collect()
                time.sleep(0.5)
                sim = run_single_sim(grid_size, obstacles, num_robots, seed)
                try:
                    plt.close("all")
                except Exception:
                    pass
                gc.collect()
                return sim, None
            except Exception as e2:
                # return second error
                return None, e2
        else:
            # not a graphics issue — return error to caller
            return None, e

# -------------------------
# Metrics collection
# -------------------------
def collect_run_metrics(sim, grid_size, num_robots, seed):
    out = {
        "grid_size": grid_size,
        "num_robots": num_robots,
        "seed": seed,
    }

    reported_union = safe_get_reported_union(sim)
    free_cells = count_free_cells(sim.grid) if hasattr(sim, "grid") else None
    total_coverage = len(reported_union)
    out["total_coverage"] = total_coverage

    pct = (total_coverage / free_cells) * 100 if free_cells else 0.0
    out["coverage_pct"] = round(pct, 2)
    out["completeness"] = "Y" if pct >= 99.9 else "N"

    # Time / steps
    steps = getattr(sim, "step_count", None) or getattr(sim, "total_steps", None) or getattr(sim, "steps", None)
    out["time_taken"] = int(steps) if steps is not None else None
    out["total_steps"] = out["time_taken"]

    # Collisions & overlap
    collisions = int(getattr(sim, "collisions", 0))
    overlap = int(getattr(sim, "re_explores", 0))
    out["collisions"] = collisions
    out["overlap_area"] = overlap

    # Per-robot stats
    frozen = 0
    per_robot_areas = []
    per_robot_paths = []
    robots = getattr(sim, "robots", [])

    for r in robots:
        if getattr(r, "stop_in_place", False):
            frozen += 1
        lm = getattr(r, "local_map", None)
        area = len(getattr(lm, "covered_cells", set())) if lm else 0
        per_robot_areas.append(area)
        path_len = len(getattr(r, "path", []))
        per_robot_paths.append(path_len)

    out["frozen_robots"] = frozen

    # Robots completed computed as described (bounded)
    computed_completed = num_robots - (frozen + collisions)
    computed_completed = max(1, min(num_robots, computed_completed))
    out["robots_completed"] = computed_completed

    # Averages
    out["avg_cells_per_robot"] = round(sum(per_robot_areas) / num_robots, 2) if num_robots > 0 else 0
    out["avg_path_per_robot"] = round(sum(per_robot_paths) / num_robots, 2) if num_robots > 0 else 0

    # Add per-robot area columns
    for i, a in enumerate(per_robot_areas, start=1):
        out[f"area_R{i}"] = a

    return out

# -------------------------
# CSV writer helper
# -------------------------
def write_rows_to_csv(filename, rows, max_robots):
    base_header = [
        "grid_size", "num_robots", "seed",
        "time_taken", "total_steps", "coverage_pct", "completeness",
        "collisions", "frozen_robots", "overlap_area", "robots_completed",
        "avg_cells_per_robot", "avg_path_per_robot", "total_coverage"
    ]
    area_cols = [f"area_R{i}" for i in range(1, max_robots + 1)]
    header = base_header + area_cols

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            # Ensure all header keys exist
            for col in header:
                row.setdefault(col, 0)
            writer.writerow(row)

# -------------------------
# Main experiment loop
# -------------------------
def main():
    print("ORMSTC Data Collection (robust) starting...")
    print("=" * 70)
    t0 = datetime.now()

    configs = []
    if RUN_MODE in ('both', '16'):
        configs.append(CONFIG_16)
    if RUN_MODE in ('both', '50'):
        configs.append(CONFIG_50)

    for (grid_size, obstacles, robot_list, out_file) in configs:
        print(f"\nRunning experiments for {grid_size}x{grid_size} (obstacles={obstacles}), output -> {out_file}")
        all_rows = []
        max_robots = max(robot_list)

        total_runs = len(robot_list) * len(SEEDS)
        run_counter = 0

        for num_robots in robot_list:
            for seed in SEEDS:
                run_counter += 1
                print(f"  [{run_counter}/{total_runs}] Grid={grid_size} Robots={num_robots} Seed={seed} ...", end="", flush=True)
                sim, err = safe_run_and_collect(grid_size, obstacles, num_robots, seed)
                if err is not None or sim is None:
                    print(" ERROR")
                    # write a placeholder row with error info
                    row = {
                        "grid_size": grid_size,
                        "num_robots": num_robots,
                        "seed": seed,
                        "time_taken": None,
                        "total_coverage": 0,
                        "coverage_pct": 0.0,
                        "completeness": "N",
                        "collisions": getattr(sim, "collisions", 0) if sim else 0,
                        "frozen_robots": getattr(sim, "frozen_robots", 0) if sim else 0,
                        "robots_completed": max(1, num_robots - getattr(sim, "collisions", 0) - getattr(sim, "frozen_robots", 0)) if sim else max(1, num_robots - 1),
                        "overlap_area": getattr(sim, "re_explores", 0) if sim else 0,
                        "avg_cells_per_robot": 0,
                        "avg_path_per_robot": 0,
                        "total_coverage": 0
                    }
                    # include no per-robot areas
                    for i in range(1, max_robots + 1):
                        row[f"area_R{i}"] = 0
                    all_rows.append(row)
                    # print error traceback for debugging
                    try:
                        print(" (exception details printed below)")
                        traceback.print_exception(type(err), err, err.__traceback__)
                    except Exception:
                        pass
                    # continue to next run
                else:
                    # Collect metrics and append
                    row = collect_run_metrics(sim, grid_size, num_robots, seed)
                    all_rows.append(row)
                    # Close any figures and collect garbage
                    try:
                        plt.close("all")
                    except Exception:
                        pass
                    gc.collect()
                    print(" OK")

        # After all runs for this config, write CSV
        write_rows_to_csv(out_file, all_rows, max_robots)
        print(f"Completed: wrote {len(all_rows)} rows to {out_file}")

    dt = datetime.now() - t0
    print("\nAll experiments finished.")
    print(f"Total runtime: {dt}")
    print("=" * 70)

if __name__ == "__main__":
    main()
