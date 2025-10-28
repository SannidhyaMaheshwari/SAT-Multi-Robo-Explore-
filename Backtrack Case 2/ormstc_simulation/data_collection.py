#!/usr/bin/env python3
"""
data_collection.py

Robust data collection script for ORMSTC simulations.

Features:
 - Forces matplotlib 'Agg' backend to avoid GUI bitmap allocation issues
 - Closes figures and runs gc.collect() after each simulation run
 - Safe-run wrapper that retries once on bitmap/memory-like errors
 - Defensive constructor calling for ORMSTCSimulation (supports malicious_robot, robot_seed)
 - Appends CSV after every experiment/run completed (safer for long runs)
 - Adds three new metrics: num_replans, pruned_cells_total, avg_backtrack_length
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
RUN_MODE = 'both'

# Seeds to run (1..25)
SEEDS = list(range(1, 26))

# Output filenames
OUT_16 = "detailed_results_16x16.csv"
OUT_50 = "detailed_results_50x50.csv"

# Configs: (grid_size, obstacles_count, robot_counts_list, out_file)
CONFIG_16 = (16, 18, [5, 10, 15, 20, 25, 30, 35, 40], OUT_16)
CONFIG_50 = (50, 175, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], OUT_50)

# Bounded stepping parameters (to avoid infinite runs)
DEFAULT_MAX_STEPS = 5000
DEFAULT_MAX_WALL_SECONDS = 300  # 5 minutes per run

# -------------------------
# Utilities
# -------------------------
def safe_get_reported_union(sim):
    try:
        if hasattr(sim.comm_system, "get_global_reported_union"):
            return set(sim.comm_system.get_global_reported_union())
        if hasattr(sim, "reported_union"):
            return set(sim.reported_union)
    except Exception:
        pass
    return set()

# Robust import for CellState (works whether run as package or standalone)
try:
    # when running as package (python -m ormstc_simulation.data_collection)
    from ormstc_simulation.core import CellState
except Exception:
    try:
        # when running as module inside package (relative)
        from .core import CellState
    except Exception:
        # last-resort fallback (if you run the file directly and PYTHONPATH includes package folder)
        from core import CellState

def count_free_cells(grid):
    """
    Robustly count non-obstacle cells.
    Tries several possible grid representations.
    """
    # Try: grid.size + is_obstacle(x,y)
    try:
        if hasattr(grid, "size") and hasattr(grid, "is_obstacle"):
            total = 0
            for y in range(grid.size):
                for x in range(grid.size):
                    if not grid.is_obstacle(x, y):
                        total += 1
            return total
    except Exception:
        pass

    # Try: numpy array grid.grid where obstacle==CellState.OBSTACLE.value
    try:
        if hasattr(grid, "grid"):
            arr = grid.grid
            try:
                return int((arr != CellState.OBSTACLE.value).sum())
            except Exception:
                # fallback to treat 0 as free
                return int((arr == 0).sum())
    except Exception:
        pass

    # Try: grid.grid_size + is_obstacle
    try:
        if hasattr(grid, "grid_size") and hasattr(grid, "is_obstacle"):
            total = 0
            for y in range(grid.grid_size):
                for x in range(grid.grid_size):
                    if not grid.is_obstacle(x, y):
                        total += 1
            return total
    except Exception:
        pass

    return None

# -------------------------
# Run single simulation (defensive, bounded)
# -------------------------
def run_single_sim(grid_size, obstacles, num_robots, seed):
    """
    Construct and run a single ORMSTCSimulation, headless.
    Uses a bounded step loop to avoid infinite runs.
    """
    # Keep obstacle generation deterministic like main.py
    random.seed(42)
    np.random.seed(42)

    sim = None
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
    except TypeError:
        # Fallback constructor attempts
        tried = False
        for ctor_args in [
            (grid_size, num_robots, obstacles, {'robot_seed': seed, 'use_gui': False}),
            (grid_size, num_robots, obstacles, {'robot_seed': seed}),
            (grid_size, num_robots, obstacles, {})
        ]:
            try:
                args = ctor_args[:3]
                kwargs = ctor_args[3] if len(ctor_args) > 3 and isinstance(ctor_args[3], dict) else {}
                sim = ORMSTCSimulation(*args, **kwargs)
                tried = True
                break
            except Exception:
                continue
        if not tried or sim is None:
            raise RuntimeError("Failed to construct ORMSTCSimulation; check signature")
    except Exception as e:
        raise

    # defensive: set seed if attribute exists
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

    # --- RUN: bounded manual stepping loop (safe) ---
    max_steps = getattr(sim, "max_steps", DEFAULT_MAX_STEPS)
    max_steps = max_steps if max_steps is not None else DEFAULT_MAX_STEPS
    MAX_WALL_SECONDS = DEFAULT_MAX_WALL_SECONDS

    start_time = time.time()
    steps = 0
    active = True
    while active and steps < max_steps and (time.time() - start_time) < MAX_WALL_SECONDS:
        try:
            active = sim.step()
        except Exception as e:
            # If step raises, bail out but return sim for partial metrics
            print(f"[WARN] sim.step() raised exception: {e}")
            traceback.print_exc()
            break
        steps += 1

    sim.step_count = steps
    if steps >= max_steps or (time.time() - start_time) >= MAX_WALL_SECONDS:
        print(f"[RUN TIMEOUT] stopped at step {steps}, wall_time={(time.time()-start_time):.1f}s")

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
                return None, e2
        else:
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

    # --- NEW: three additional metrics ---
    # Prefer sim.replans_after_overlap (what GUI shows). Fallback to sim.backtracks if present.
    num_replans_val = None
    if hasattr(sim, "replans_after_overlap"):
        try:
            num_replans_val = int(getattr(sim, "replans_after_overlap", 0))
        except Exception:
            num_replans_val = int(getattr(sim, "backtracks", 0) if hasattr(sim, "backtracks") else 0)
    else:
        num_replans_val = int(getattr(sim, "backtracks", 0))

    out["num_replans"] = num_replans_val
    # pruned_cells_total: total number of cells removed/unclaimed globally (comm system counter)
    out["pruned_cells_total"] = int(getattr(getattr(sim, "comm_system", None), "pruned_cells_count", 0))
    # avg_backtrack_length: pruned_cells_total / num_replans (guard division)
    try:
        if out["num_replans"] > 0:
            out["avg_backtrack_length"] = round(out["pruned_cells_total"] / out["num_replans"], 2)
        else:
            out["avg_backtrack_length"] = 0.0
    except Exception:
        out["avg_backtrack_length"] = 0.0

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
# CSV writer helper (supports append)
# -------------------------
def write_rows_to_csv(filename, rows, max_robots, append=False):
    base_header = [
        "grid_size", "num_robots", "seed",
        "time_taken", "total_steps", "coverage_pct", "completeness",
        "collisions", "frozen_robots", "overlap_area", "robots_completed",
        "avg_cells_per_robot", "avg_path_per_robot", "total_coverage",
        # new metrics (kept near summary)
        "num_replans", "pruned_cells_total", "avg_backtrack_length"
    ]
    area_cols = [f"area_R{i}" for i in range(1, max_robots + 1)]
    header = base_header + area_cols

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(filename)
    mode = "a" if append and file_exists else "w"
    write_header = not (append and file_exists)

    with open(filename, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            # Ensure all header keys exist
            for col in header:
                row.setdefault(col, 0)
            writer.writerow(row)

# -------------------------
# Main experiment loop (appends after each run)
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
        max_robots = max(robot_list)

        total_runs = len(robot_list) * len(SEEDS)
        run_counter = 0

        # If file exists from previous runs, we will append. If not, header will be written on first write.
        for num_robots in robot_list:
            for seed in SEEDS:
                run_counter += 1
                print(f"  [{run_counter}/{total_runs}] Grid={grid_size} Robots={num_robots} Seed={seed} ...", end="", flush=True)
                sim, err = safe_run_and_collect(grid_size, obstacles, num_robots, seed)
                if err is not None or sim is None:
                    print(" ERROR")
                    # write a placeholder row with error info
                    # use same num_replans sourcing as collector
                    num_replans_placeholder = 0
                    if sim is not None:
                        if hasattr(sim, "replans_after_overlap"):
                            try:
                                num_replans_placeholder = int(getattr(sim, "replans_after_overlap", 0))
                            except Exception:
                                num_replans_placeholder = int(getattr(sim, "backtracks", 0) if hasattr(sim, "backtracks") else 0)
                        else:
                            num_replans_placeholder = int(getattr(sim, "backtracks", 0))
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
                        "total_coverage": 0,
                        # include defaults for new metrics
                        "num_replans": num_replans_placeholder,
                        "pruned_cells_total": int(getattr(getattr(sim, "comm_system", None), "pruned_cells_count", 0)) if sim else 0,
                        "avg_backtrack_length": 0.0
                    }
                    # include no per-robot areas
                    for i in range(1, max_robots + 1):
                        row[f"area_R{i}"] = 0
                else:
                    # Collect metrics and create row
                    row = collect_run_metrics(sim, grid_size, num_robots, seed)
                    print(" OK")

                # Append single result immediately to CSV
                write_rows_to_csv(out_file, [row], max_robots, append=True)
                print(f"  -> Appended to {out_file}")

                # cleanup graphics & memory
                try:
                    plt.close("all")
                except Exception:
                    pass
                gc.collect()

        print(f"Completed config: output written (appended) to {out_file}")

    dt = datetime.now() - t0
    print("\nAll experiments finished.")
    print(f"Total runtime: {dt}")
    print("=" * 70)

if __name__ == "__main__":
    main()
