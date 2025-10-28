import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Robot Class ----------------
class ORMSTCRobot:
    def __init__(self, rid, start, visited_by, obstacles):
        self.id = rid
        self.start = start
        self.cur = start
        self.stack = [(None, start, 1)]
        self.alive = True
        self.done = False
        self.tree_edges = set()
        self.connections = {}
        self.moves_log = []
        r, c = start
        visited_by[r][c] = rid
        self.moves_log.append(("START", start))

    def init_connections(self, all_ids):
        for j in all_ids:
            if j != self.id:
                self.connections[j] = [None, None]

    def neighbors(self, X, W, R, C):
        r, c = X
        dirs = [(-1,0),(0,1),(1,0),(0,-1)]  # N,E,S,W
        neigh = [(r+dr, c+dc) for dr,dc in dirs]
        # explore parent last (so backtracking happens after other neighbors)
        if W is not None and W in neigh:
            neigh.remove(W)
            neigh.append(W)
        return neigh

    def get_intention(self, visited_by, obstacles, robots):
        """Return next intended cell, or None if backtracking or idle."""
        if not self.alive or self.done:
            return None
        W, X, i = self.stack[-1]
        R, C = len(visited_by), len(visited_by[0])
        neigh = self.neighbors(X, W, R, C)
        limit = len(neigh) if W is None else min(3, len(neigh))
        while i <= limit:
            Ni = neigh[i-1] if i-1 < len(neigh) else None
            i += 1
            self.stack[-1] = (W, X, i)
            if Ni is None:
                continue
            rr, cc = Ni
            if not (0 <= rr < R and 0 <= cc < C):
                continue
            if obstacles[rr][cc]:
                continue
            if visited_by[rr][cc] == self.id:
                continue
            other = visited_by[rr][cc]
            if other is not None and other != self.id:
                if robots[other].alive:
                    if self.connections[other][0] is None:
                        self.connections[other][0] = (X, Ni)
                    self.connections[other][1] = (X, Ni)
                    continue
                else:
                    # takeover: free dead robot's cells
                    for rr2 in range(R):
                        for cc2 in range(C):
                            if visited_by[rr2][cc2] == other:
                                visited_by[rr2][cc2] = None
                    return Ni
            else:
                return Ni
        return W if W is not None else None

    def execute_move(self, target, visited_by):
        """Move to approved target (forward or backtrack)."""
        if target is None:
            return False
        W, X, i = self.stack[-1]
        if target == W and W is not None:
            # backtrack
            self.cur = W
            self.stack.pop()
            self.moves_log.append(("BACKTRACK", W))
            return True
        # forward expansion
        self.tree_edges.add((X, target))
        rr, cc = target
        visited_by[rr][cc] = self.id
        self.stack.append((X, target, 1))
        self.cur = target
        self.moves_log.append(("MOVE", target))
        if len(self.stack) == 1 and self.stack[-1][2] > 4:
            self.done = True
            self.moves_log.append(("DONE", self.cur))
        return True


# ---------------- Simulation Generator ----------------
def run_simulation_iter(R=20, C=20, num_robots=4, num_obstacles=10, fail_rate=0.0, max_steps=10000):
    obstacles = [[False]*C for _ in range(R)]
    all_cells = [(r,c) for r in range(R) for c in range(C)]
    random.shuffle(all_cells)
    for (r,c) in all_cells[:num_obstacles]:
        obstacles[r][c] = True

    visited_by = [[None]*C for _ in range(R)]

    free_cells = [(r,c) for r in range(R) for c in range(C) if not obstacles[r][c]]
    if len(free_cells) < num_robots:
        raise ValueError("Not enough free cells to place robots.")
    random.shuffle(free_cells)
    robots = {}
    for i in range(num_robots):
        cell = free_cells.pop()
        robots[i] = ORMSTCRobot(i, cell, visited_by, obstacles)
    for r in robots.values():
        r.init_connections(list(robots.keys()))

    step_count = 0
    yield robots, visited_by, obstacles, step_count

    if all((not r.alive or r.done) for r in robots.values()):
        return

    while step_count < max_steps:
        step_count += 1
        intentions = {}

        # -------- Phase 1: collect intentions --------
        for rid, r in robots.items():
            if r.alive and not r.done:
                if random.random() < fail_rate:
                    r.alive = False
                    r.moves_log.append(("FAIL", r.cur))
                else:
                    intentions[rid] = r.get_intention(visited_by, obstacles, robots)

        # -------- Phase 2: resolve conflicts --------
        cell_to_rids = {}
        for rid, cell in intentions.items():
            if cell is not None:
                cell_to_rids.setdefault(cell, []).append(rid)
        winners = {}
        for cell, contenders in cell_to_rids.items():
            winner = max(contenders)  # 🔴 highest ID wins
            for rid in contenders:
                winners[rid] = cell if rid == winner else None
        for rid in intentions.keys():
            if rid not in winners:
                winners[rid] = None

        # -------- Phase 3: apply moves together --------
        moved_any = False
        for rid, target in winners.items():
            moved = robots[rid].execute_move(target, visited_by)
            if moved:
                moved_any = True

        yield robots, visited_by, obstacles, step_count

        if all((not r.alive or r.done) for r in robots.values()):
            return
        if not moved_any:
            print("⚠️ Deadlock: no moves possible. Stopping.")
            return

    print("⚠️ Stopped due to max_steps guard (possible deadlock).")


# ---------------- Animate & Report ----------------
def animate_and_report(R=20, C=20, num_robots=4, num_obstacles=10, fail_rate=0.0, interval_ms=400):
    sim_gen = run_simulation_iter(R, C, num_robots, num_obstacles, fail_rate)
    fig, ax = plt.subplots(figsize=(6,6))
    final_state = None

    def update(frame):
        nonlocal final_state
        robots, visited_by, obstacles, step = frame
        final_state = frame
        ax.clear()
        # draw grid
        for r in range(R):
            for c in range(C):
                if obstacles[r][c]:
                    ax.scatter(c, R-1-r, color='black', s=10)
                elif visited_by[r][c] is None:
                    ax.scatter(c, R-1-r, color='white', edgecolor='gray', s=10)
                else:
                    ax.scatter(c, R-1-r, color=f"C{visited_by[r][c]}", s=10)
        # draw robot paths and current positions
        for r in robots.values():
            for (u,v) in r.tree_edges:
                (r1,c1),(r2,c2) = u,v
                ax.plot([c1,c2],[R-1-r1,R-1-r2],color=f"C{r.id}",linewidth=1)
            sr,sc = r.start
            ax.plot(sc, R-1-sr, '*', color=f"C{r.id}", markersize=12, markeredgecolor='black')
            if r.alive:
                rr,cc = r.cur
                ax.plot(cc, R-1-rr, 'o', color=f"C{r.id}", markersize=8, markeredgecolor='black')
                ax.text(cc, R-1-rr, str(r.id), color="white", ha="center", va="center", fontsize=6, fontweight="bold")
        ax.set_title(f"ORMSTC Simulation – Step {step}")
        ax.axis('off')

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=sim_gen,
        interval=interval_ms,
        repeat=False,
        cache_frame_data=False
    )

    plt.show()

    if final_state is None:
        print("No frames were generated.")
        return

    robots, visited_by, obstacles, total_steps = final_state
    print(f"\n✅ Simulation completed in {total_steps} steps.")
    total_free = sum(1 for r in range(R) for c in range(C) if not obstacles[r][c])
    covered_cells = sum(1 for r in range(R) for c in range(C) if visited_by[r][c] is not None)
    print(f"Covered cells: {covered_cells}/{total_free} free cells")

    for rid, r in robots.items():
        owned = sum(1 for rr in range(R) for cc in range(C) if visited_by[rr][cc] == rid)
        print(f"Robot {rid}: actions={len(r.moves_log):3d}, alive={r.alive}, "
              f"final_pos={r.cur}, start={r.start}, cells_owned={owned}")

    return final_state


# ---------------- Example run ----------------
if __name__ == "__main__":
    R,C = 10,10
    num_robots = 4
    num_obstacles = 12
    fail_rate = 0.0
    interval_ms = 300
    animate_and_report(R=R, C=C, num_robots=num_robots, num_obstacles=num_obstacles, fail_rate=fail_rate, interval_ms=interval_ms)
