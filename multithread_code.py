# synchronous_multithreaded_ormstc.py
import random
import time
import threading
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class ThreadedRobot(threading.Thread):
    """A Robot that runs in its own thread but synchronizes with others via a barrier."""
    def __init__(self, rid, start_pos, R, C, obstacles, sim_ref):
        super().__init__(name=f"Robot-{rid}", daemon=True)
        self.id = rid
        self.start_pos = start_pos
        self.R, self.C = R, C
        self.obstacles = obstacles

        # Shared simulation resources
        self.sim = sim_ref
        
        # Robot's internal state
        self.stack = [start_pos]
        self.cur_pos = start_pos
        self.state = "exploring"
        self.visited_local = {start_pos}
        self.moves_log = []
        self.done = False
        self.stop_event = threading.Event()

    def run(self):
        """The main loop for each robot thread."""
        while not self.stop_event.is_set():
            # === PROPOSE PHASE ===
            # Decide on a move based on the simulation's public state
            proposal = self.propose_move(self.sim.global_visited)
            
            # Post the proposal for the main thread to arbitrate
            with self.sim.proposal_lock:
                self.sim.proposals[self.id] = proposal
            
            # Wait at the barrier for all other robots to finish proposing
            self.sim.barrier.wait()

            # === APPLY PHASE ===
            # Wait for the main thread to finish arbitration
            self.sim.barrier.wait()

            # Get the outcome of our proposal
            action, dest = self.sim.outcomes.get(self.id, (None, None))
            
            if action:
                self.apply_move(action, dest)

            # Wait for all robots to finish applying their moves before the next step
            self.sim.barrier.wait()

    def stop(self):
        self.stop_event.set()

    def neighbors(self, pos):
        r, c = pos
        dirs = [(-1,0), (0,1), (1,0), (0,-1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.R and 0 <= nc < self.C and not self.obstacles[nr][nc]:
                yield (nr, nc)

    def propose_move(self, global_visited):
        if self.done:
            return None
        cur = self.stack[-1]

        if self.state == "exploring":
            for nb in self.neighbors(cur):
                if nb not in self.visited_local and nb not in global_visited:
                    return ("MOVE", cur, nb)
            if len(self.stack) > 1:
                parent = self.stack[-2]
                return ("BACKTRACK", cur, parent)
            else:
                self.state = "returning_home"
                return None

        if self.state == "returning_home":
            if cur == self.start_pos:
                self.done = True
                self.state = "done"
                return ("DONE", cur, cur)
            if len(self.stack) > 1:
                parent = self.stack[-2]
                return ("RETURN", cur, parent)
        return None

    def apply_move(self, action, dest=None):
        if action == "MOVE":
            self.stack.append(dest)
            self.cur_pos = dest
            self.visited_local.add(dest)
            self.moves_log.append(("MOVE", dest))
        elif action in ("BACKTRACK", "RETURN"):
            if len(self.stack) > 1:
                self.stack.pop()
                self.cur_pos = self.stack[-1]
                self.moves_log.append(("BACKTRACK", self.cur_pos))
        elif action == "DONE":
            self.done = True
            self.state = "done"
            self.moves_log.append(("DONE", self.cur_pos))

# ---------------- Synchronous Multithreaded Simulation ----------------
class Simulation:
    def __init__(self, R=10, C=10, num_robots=4, num_obstacles=12, max_steps=500, seed=None):
        if seed is not None: random.seed(seed)
        self.R, self.C = R, C
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.obstacles, self.free_cells = self._generate_connected_obstacles(num_obstacles)
        self.total_free_cells = len(self.free_cells)

        # Synchronization primitives
        self.barrier = threading.Barrier(self.num_robots + 1) # +1 for the main thread
        self.proposal_lock = threading.Lock()
        
        # Shared data between threads and main controller
        self.proposals = {}
        self.outcomes = {}
        
        starts = random.sample(self.free_cells, self.num_robots)
        self.robots = [ThreadedRobot(i, starts[i], R, C, self.obstacles, self) for i in range(self.num_robots)]

        # Authoritative simulation state
        self.global_visited = set(starts)
        self.visited_by = [[None]*C for _ in range(R)]
        for rid, (r,c) in enumerate(starts):
            self.visited_by[r][c] = rid
        self.edges = [set() for _ in range(self.num_robots)]
        
        self.step = 0
        self.start_time = 0
        self.end_time = 0

    def _generate_connected_obstacles(self, num_obstacles):
        # ... (no changes from previous version) ...
        cells = [(r,c) for r in range(self.R) for c in range(self.C)]
        if num_obstacles >= len(cells):
            raise ValueError("num_obstacles too large")
        while True:
            random.shuffle(cells)
            obstacles = [[False]*self.C for _ in range(self.R)]
            for r,c in cells[:num_obstacles]:
                obstacles[r][c] = True
            free = [cell for cell in cells if not obstacles[cell[0]][cell[1]]]
            if not free: continue
            q = deque([free[0]])
            seen = {free[0]}
            while q:
                rr, cc = q.popleft()
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = rr+dr, cc+dc
                    if 0<=nr<self.R and 0<=nc<self.C and not obstacles[nr][nc] and (nr,nc) not in seen:
                        seen.add((nr,nc)); q.append((nr,nc))
            if len(seen) == len(free):
                return obstacles, free

    def _arbitrate_and_apply(self):
        # This logic is from the original _step_once, now run by the main thread
        cell_claims = defaultdict(list)
        for rid, proposal in self.proposals.items():
            if proposal and proposal[0] == "MOVE":
                action, _cur, dest = proposal
                cell_claims[dest].append(rid)

        winners = {}
        for dest, claimers in cell_claims.items():
            winners[dest] = max(claimers)
            
        moved_any = False
        self.outcomes.clear()
        
        for rid, proposal in self.proposals.items():
            if not proposal: continue
            
            action, cur, dest = proposal
            if action == "MOVE":
                winner = winners.get(dest)
                if winner == rid:
                    self.outcomes[rid] = ("MOVE", dest)
                    self.global_visited.add(dest)
                    self.visited_by[dest[0]][dest[1]] = rid
                    self.edges[rid].add(tuple(sorted((cur, dest))))
                    moved_any = True
                else:
                    self.outcomes[rid] = ("BACKTRACK", None)
                    moved_any = True # Backtracking is also a move
            elif action in ("BACKTRACK", "RETURN"):
                self.outcomes[rid] = (action, dest)
                moved_any = True
            elif action == "DONE":
                self.outcomes[rid] = (action, dest)
        
        self.proposals.clear()
        return moved_any

    def run(self, visualize=True, frame_interval_ms=200):
        self.start_time = time.time()
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8,8))
            plt.show()

        for robot in self.robots:
            robot.start()

        for s in range(self.max_steps):
            self.step += 1
            
            # === SYNC POINT 1: Wait for proposals ===
            # The main thread waits here until all robot threads have posted their proposals.
            self.barrier.wait()
            
            # === ARBITRATION (Main Thread Only) ===
            moved = self._arbitrate_and_apply()
            
            # === SYNC POINT 2: Wait for application ===
            # Unblock robots to let them apply their outcomes. Main thread waits for them to finish.
            self.barrier.wait()
            
            # === SYNC POINT 3: Step Complete ===
            # All robots have applied their moves and are now waiting for the next step.
            self.barrier.wait()

            if visualize:
                self._draw_frame(ax)
                plt.draw()
                plt.gcf().canvas.flush_events()
                time.sleep(frame_interval_ms/1000.0)

            if all(r.done for r in self.robots):
                break
            if not moved:
                print(f"⚠️ No robot moved at step {self.step} — terminating early.")
                break

        self.end_time = time.time()
        
        # Signal all threads to stop
        for robot in self.robots:
            robot.stop()
        
        # Release the barrier one last time to unblock any waiting threads so they can exit
        try: self.barrier.wait(timeout=1.0)
        except threading.BrokenBarrierError: pass

        if visualize:
            plt.ioff()
            self._draw_frame(ax)
            plt.show()
        
        self._report()

    def _draw_frame(self, ax):
        # ... (no changes from previous version) ...
        ax.clear()
        for r in range(self.R):
            for c in range(self.C):
                if self.obstacles[r][c]:
                    ax.scatter(c, self.R-1-r, color='black', s=18, marker='s')
                else:
                    owner = self.visited_by[r][c]
                    color = f"C{owner}" if owner is not None else "#f0f0f0"
                    ax.scatter(c, self.R-1-r, color=color, s=12)
        for rid in range(self.num_robots):
            for (u,v) in self.edges[rid]:
                ax.plot([u[1], v[1]], [self.R-1-u[0], self.R-1-v[0]], color=f"C{rid}", linewidth=1.2)
        for rid, robot in enumerate(self.robots):
            sr, sc = robot.start_pos
            ax.plot(sc, self.R-1-sr, '*', color=f"C{rid}", markersize=15, markeredgecolor='black', zorder=4)
            if not robot.done:
                rr, cc = robot.cur_pos
                ax.plot(cc, self.R-1-rr, 'o', color=f"C{rid}", markersize=10, markeredgecolor='black', zorder=5)
                ax.text(cc, self.R-1-rr, str(rid), ha='center', va='center', color='white', fontsize=8, weight='bold', zorder=6)
        elapsed = time.time() - self.start_time
        cov = len(self.global_visited)
        ax.set_title(f"Step {self.step} | Coverage: {cov}/{self.total_free_cells} | Time: {elapsed:.1f}s")
        ax.set_aspect('equal')
        ax.axis('off')

    def _report(self):
        # ... (no changes from previous version) ...
        print("\n--- Final Report ---")
        print(f"Steps: {self.step}, Time: {self.end_time - self.start_time:.3f}s")
        cov = len(self.global_visited)
        print(f"Coverage: {cov}/{self.total_free_cells} ({cov/self.total_free_cells:.1%})")
        for rid, robot in enumerate(self.robots):
            owned = sum(1 for r in range(self.R) for c in range(self.C) if self.visited_by[r][c] == rid)
            print(f"Robot {rid}: final_pos={robot.cur_pos}, done={robot.done}, actions={len(robot.moves_log)}, cells_owned={owned}")


if __name__ == "__main__":
    sim = Simulation(R=10, C=10, num_robots=6, num_obstacles=12, max_steps=500, seed=42)
    sim.run(visualize=True, frame_interval_ms=200)