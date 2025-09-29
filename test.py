import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------- Parameters -------------
R = 10
C = 10
num_robots = 4
num_obstacles = 12
fail_rate = 0.0         # spontaneous failure prob for non-malicious (not used much here)
interval_ms = 300
max_steps = 1000
MALICIOUS_BURST_N = 3   # every nth round malicious bursts
SEED = 12345
VERBOSE = True
# --------------------------------------

random.seed(SEED)

FREE = None

# ---------------- Robot Class ----------------
class ORMSTCRobot:
    def __init__(self, rid, start, visited_by, obstacles):
        self.id = rid
        self.start = start
        self.cur = start
        # stack entries: (parent_cell, this_cell, next_neighbor_index)
        self.stack = [(None, start, 1)]
        self.alive = True    # health flag (kept True unless explicit failure)
        self.active = True   # active==False -> robot stops moving (victim/inactive)
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

    def neighbors(self, X, W, Rn, Cn):
        r, c = X
        dirs = [(-1,0),(0,1),(1,0),(0,-1)]  # N,E,S,W
        neigh = [(r+dr, c+dc) for dr,dc in dirs]
        # explore parent last (so backtracking happens after other neighbors)
        if W is not None and W in neigh:
            neigh.remove(W)
            neigh.append(W)
        return neigh

    def get_intention(self, visited_by, obstacles, robots, treat_as_visited=None):
        """
        Decide next move (returns target cell tuple or None).
        treat_as_visited: optional set of cells to treat as already visited (used by malicious internal memory)
        """
        if not self.active or self.done:
            return None
        W, X, i = self.stack[-1]
        Rn, Cn = len(visited_by), len(visited_by[0])
        neigh = self.neighbors(X, W, Rn, Cn)
        limit = len(neigh) if W is None else min(3, len(neigh))
        while i <= limit:
            Ni = neigh[i-1] if i-1 < len(neigh) else None
            i += 1
            # save updated next-index
            self.stack[-1] = (W, X, i)
            if Ni is None:
                continue
            rr, cc = Ni
            if not (0 <= rr < Rn and 0 <= cc < Cn):
                continue
            if obstacles[rr][cc]:
                continue
            # treat internal visited (malicious) as visited so it doesn't revisit
            if treat_as_visited and (rr, cc) in treat_as_visited:
                continue
            if visited_by[rr][cc] == self.id:
                continue
            other = visited_by[rr][cc]
            if other is not None and other != self.id:
                if robots[other].active:
                    # encountered active peer -> record connection but don't enter
                    if self.connections[other][0] is None:
                        self.connections[other][0] = (X, Ni)
                    self.connections[other][1] = (X, Ni)
                    continue
                else:
                    # takeover dead/inactive peer's cells: free them (simple policy)
                    for rr2 in range(Rn):
                        for cc2 in range(Cn):
                            if visited_by[rr2][cc2] == other:
                                visited_by[rr2][cc2] = None
                    return Ni
            else:
                # free cell -> intend to move in
                return Ni
        # no forward neighbor found: backtrack to parent if exists, else None
        return W if W is not None else None

    def execute_move(self, target, visited_by, claim=True):
        """Execute move. claim=True -> attempt to mark visited_by (only if FREE)."""
        if target is None:
            return False
        W, X, i = self.stack[-1]
        # backtrack
        if target == W and W is not None:
            self.cur = W
            self.stack.pop()
            self.moves_log.append(("BACKTRACK", W))
            return True
        # forward
        self.tree_edges.add((X, target))
        rr, cc = target
        # claim only if claim=True and cell free (never overwrite)
        if claim:
            if visited_by[rr][cc] is None:
                visited_by[rr][cc] = self.id
            else:
                # cannot claim an already-claimed cell => move cannot claim; but we still push stack to keep tree consistent?
                # We'll not push stack if we fail to claim, to avoid being inconsistent.
                return False
        # push new stack entry and update current pos
        self.stack.append((X, target, 1))
        self.cur = target
        self.moves_log.append(("MOVE", target))
        # detect done at root with all neighbors considered
        if len(self.stack) == 1 and self.stack[-1][2] > 4:
            self.done = True
            self.moves_log.append(("DONE", self.cur))
        return True

# ---------------- Simulation generator (with malicious) ----------------
def run_simulation_iter(R=20, C=20, num_robots=4, num_obstacles=10, fail_rate=0.0, max_steps=10000,
                       malicious_burst_n=3, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
    # place obstacles
    obstacles = [[False]*C for _ in range(R)]
    all_cells = [(r,c) for r in range(R) for c in range(C)]
    random.shuffle(all_cells)
    for (r,c) in all_cells[:num_obstacles]:
        obstacles[r][c] = True

    visited_by = [[None]*C for _ in range(R)]

    # place robots
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

    # identify malicious: highest id
    malicious_id = max(robots.keys())
    if verbose:
        print(f"[SIM] Malicious robot id = {malicious_id}, burst interval = {malicious_burst_n}")

    # malicious buffering & internal visited set
    malicious_buffer = []          # list of target cells buffered
    malicious_internal_visited = set()  # prevents oscillation for malicious

    step_count = 0
    # yield initial state
    yield robots, visited_by, obstacles, step_count, {"burst": False, "burst_targets": [], "withheld": False, "victims": []}

    # quick termination check
    if all((not r.alive or r.done or not r.active) for r in robots.values()):
        return

    # main loop
    while step_count < max_steps:
        step_count += 1
        intentions = {}             # rid -> target (broadcasted this round)
        intent_sources = {}         # cell -> list of (rid, tag) tags: 'normal' or 'malicious_delayed'
        withheld_this_round = False
        burst_this_round = False
        burst_targets = []
        victims_this_round = set()

        # ---------- Phase 1: collect intentions ----------
        for rid, r in robots.items():
            if not r.active:
                continue
            # simulated spontaneous failure for non-malicious if desired
            if rid != malicious_id and random.random() < fail_rate:
                r.active = False
                r.moves_log.append(("FAIL", r.cur))
                if verbose:
                    print(f"[round {step_count}] Robot {rid} spont. failed (now inactive).")
                continue

            # compute intent
            if rid == malicious_id:
                tgt = r.get_intention(visited_by, obstacles, robots, treat_as_visited=malicious_internal_visited)
                # buffer any non-None intent
                if tgt is not None:
                    malicious_buffer.append(tgt)
                    withheld_this_round = True
                # malicious moves stealthily (updates internal tree & position but DOES NOT mark visited_by)
                if tgt is not None:
                    # execute stealth move (no claim)
                    moved = r.execute_move(tgt, visited_by, claim=False)
                    if moved:
                        # remember stealth-visited to avoid oscillation
                        malicious_internal_visited.add(tgt)
                        if verbose:
                            print(f"[round {step_count}] Malicious {rid} stealth-moved to {tgt} (withheld).")
                # do not add malicious to intentions (withheld)
            else:
                # normal robot broadcasts its intention immediately
                tgt = r.get_intention(visited_by, obstacles, robots)
                intentions[rid] = tgt
                if tgt is not None:
                    intent_sources.setdefault(tgt, []).append((rid, "normal"))

        # ---------- Malicious burst: every malicious_burst_n rounds announce all buffered intents ----------
        if (step_count % malicious_burst_n) == 0 and malicious_buffer:
            burst_this_round = True
            # deduplicate buffered targets preserving order
            seen = set()
            burst_targets = []
            for t in malicious_buffer:
                if t not in seen:
                    seen.add(t)
                    burst_targets.append(t)
            malicious_buffer = []
            # inject these as malicious delayed intents into intent_sources
            for t in burst_targets:
                intent_sources.setdefault(t, []).append((malicious_id, "malicious_delayed"))
            if verbose:
                print(f"[round {step_count}] MALICIOUS BURST: announcing {len(burst_targets)} buffered intents -> {burst_targets}")

        # ---------- Phase 2: resolve conflicts ----------
        # Build winners mapping: rid -> approved target (None if lost)
        winners = {}

        # handle contested cells
        for cell, contenders in intent_sources.items():
            # contenders may include malicious_delayed and normal
            cont_rids = list({rid for (rid, tag) in contenders})
            # highest id wins
            winner = max(cont_rids)
            for rid in cont_rids:
                winners[rid] = cell if rid == winner else None

        # any broadcasted intention that wasn't in intent_sources (i.e., uncontested) gets added
        for rid, tgt in intentions.items():
            if rid not in winners:
                winners[rid] = tgt

        # ensure all robots present in winners
        for rid in robots.keys():
            if rid not in winners:
                winners[rid] = None

        # ---------- Determine victims due to malicious_delayed wins ----------
        if burst_this_round and burst_targets:
            for t in burst_targets:
                contenders = intent_sources.get(t, [])
                rids_here = [rid for (rid, tag) in contenders]
                # if malicious among contenders and malicious won, legitimate contenders become victims
                if malicious_id in rids_here:
                    winner = max(set(rids_here))
                    if winner == malicious_id:
                        for (rid, tag) in contenders:
                            if rid != malicious_id and robots[rid].active:
                                victims_this_round.add(rid)
            if victims_this_round and verbose:
                print(f"[round {step_count}] Victims due to malicious burst: {sorted(list(victims_this_round))}")

        # ---------- Phase 3: apply moves together ----------
        # First, mark victims inactive (they keep their ownership and tree)
        for vid in victims_this_round:
            if robots[vid].active:
                robots[vid].active = False
                robots[vid].moves_log.append(("MALICIOUS_FAILED", robots[vid].cur))
                if verbose:
                    print(f"[round {step_count}] Robot {vid} marked INACTIVE due to malicious delayed conflict. Its tree/ownership preserved.")

        moved_any = False

        # Apply winners' moves:
        # Note: malicious already stealth-moved during Phase 1. For malicious burst winners we allow claim if FREE.
        for rid, target in winners.items():
            # skip inactive
            if not robots[rid].active:
                continue
            if target is None:
                continue
            rr, cc = target
            owner = visited_by[rr][cc]
            if owner is not None and owner != rid:
                # cannot overwrite existing owner; skip
                if verbose:
                    if rid == malicious_id:
                        # malicious attempted to claim an already-owned cell; ownership preserved
                        print(f"[round {step_count}] Malicious delayed intent {rid}->{target} found owner {owner}; not overwriting.")
                    else:
                        print(f"[round {step_count}] Robot {rid} attempted to claim already-owned {target} (owner {owner}) -> fails.")
                continue

            # Now, if rid is malicious:
            if rid == malicious_id:
                # if target free, claim it now (assign ownership)
                if owner is None:
                    # set ownership
                    visited_by[rr][cc] = rid
                    # also ensure malicious_internal_visited knows it (so it won't oscillate)
                    malicious_internal_visited.add((rr,cc))
                    moved_any = True
                    if verbose:
                        print(f"[round {step_count}] Malicious {rid} claims {target} during burst.")
                else:
                    # nothing to do if already owned (we already handled skip)
                    pass
                continue

            # For normal robots: perform normal move with claim (execute_move checks non-overwrite)
            moved = robots[rid].execute_move(target, visited_by, claim=True)
            if moved:
                moved_any = True

        # yield state and meta
        yield robots, visited_by, obstacles, step_count, {
            "burst": burst_this_round,
            "burst_targets": burst_targets,
            "withheld": withheld_this_round,
            "victims": sorted(list(victims_this_round))
        }

        # termination checks
        all_inactive_or_done = all((not r.active or r.done) for r in robots.values())
        if all_inactive_or_done:
            if verbose:
                print(f"[SIM] All robots inactive or done at round {step_count}. Terminating.")
            return

        if not moved_any and not burst_this_round:
            if verbose:
                print(f"[SIM] No robots moved in round {step_count} and no burst. Stopping to avoid deadlock.")
            return

    if verbose:
        print(f"[SIM] Reached max_steps {max_steps}. Stopping.")
    return

# ---------------- Animation & Reporting ----------------
def animate_and_report(R=R, C=C, num_robots=num_robots, num_obstacles=num_obstacles, fail_rate=fail_rate,
                       interval_ms=interval_ms, malicious_burst_n=MALICIOUS_BURST_N, max_steps=max_steps,
                       seed=SEED, verbose=VERBOSE):
    sim_gen = run_simulation_iter(R, C, num_robots, num_obstacles, fail_rate, max_steps,
                                 malicious_burst_n, seed, verbose)
    fig, ax = plt.subplots(figsize=(6,6))
    final_state = None

    def update(frame):
        nonlocal final_state
        robots, visited_by, obstacles, step, meta = frame
        final_state = frame
        burst = meta["burst"]
        burst_targets = meta.get("burst_targets", [])
        victims = meta.get("victims", [])
        withheld = meta.get("withheld", False)

        ax.clear()
        # draw grid cells
        for r in range(R):
            for c in range(C):
                if obstacles[r][c]:
                    ax.scatter(c, R-1-r, color='black', s=14)
                elif visited_by[r][c] is None:
                    ax.scatter(c, R-1-r, color='white', edgecolor='gray', s=10)
                else:
                    ax.scatter(c, R-1-r, color=f"C{visited_by[r][c]}", s=10)

        # draw trees, starts, robots
        for rid, r in robots.items():
            for (u,v) in r.tree_edges:
                (r1,c1),(r2,c2) = u,v
                ax.plot([c1,c2],[R-1-r1,R-1-r2], color=f"C{r.id}", linewidth=1)
            sr, sc = r.start
            ax.plot(sc, R-1-sr, '*', color=f"C{r.id}", markersize=12, markeredgecolor='black')
            rr, cc = r.cur
            if r.active:
                ax.plot(cc, R-1-rr, 'o', color=f"C{r.id}", markersize=8, markeredgecolor='black')
                ax.text(cc, R-1-rr, str(r.id), color="white", ha="center", va="center", fontsize=7, fontweight="bold")
            else:
                ax.plot(cc, R-1-rr, 'x', color='dimgray', markersize=8)
                ax.text(cc, R-1-rr, str(r.id), color="black", ha="center", va="center", fontsize=7)

        # withheld indicator near malicious
        malicious_id_local = max(robots.keys())
        if withheld and robots[malicious_id_local].active:
            m = robots[malicious_id_local]
            mr, mc = m.cur
            ax.text(mc + 0.25, R-1-mr, "×", color="red", fontsize=14, fontweight="bold")

        title = f"ORMSTC with Malicious (burst every {malicious_burst_n}) — Round {step}"
        if burst:
            title += "  [MALICIOUS BURST]"
        if victims:
            title += f"  (Victims: {victims})"
        ax.set_title(title)
        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=sim_gen, interval=interval_ms, repeat=False, cache_frame_data=False)
    plt.show()

    # final reporting
    if final_state is None:
        print("No frames were generated.")
        return

    robots, visited_by, obstacles, total_steps, meta = final_state
    print(f"\n=== ORMSTC Simulation Report ===")
    print(f"Grid: {R}x{C}, Robots: {num_robots}, Steps: {total_steps}")
    total_free = sum(1 for r in range(R) for c in range(C) if not obstacles[r][c])
    covered = sum(1 for r in range(R) for c in range(C) if visited_by[r][c] is not None)
    print(f"Total free: {total_free}, Covered: {covered}")
    print()
    for rid, r in robots.items():
        owned = sum(1 for rr in range(R) for cc in range(C) if visited_by[rr][cc] == rid)
        print(f" robot {rid:3d} active={r.active!s:5s} moves={len(r.moves_log):3d} owned={owned:3d} final_pos={r.cur}")

    return final_state

# ---------------- Run ----------------
if __name__ == "__main__":
    animate_and_report()
