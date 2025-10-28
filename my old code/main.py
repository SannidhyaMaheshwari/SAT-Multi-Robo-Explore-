import threading
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ---------------- Concurrency Primitives ----------------
class WriterPreferenceRWLock:
    # ... (no changes from previous version) ...
    def __init__(self):
        self._read_count = 0
        self._write_count = 0
        self._writer_waiting = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._read_cv = threading.Condition(self._read_lock)
        self._write_cv = threading.Condition(self._write_lock)

    def acquire_read(self):
        with self._read_lock:
            while self._write_count > 0 or self._writer_waiting > 0:
                self._read_cv.wait()
            self._read_count += 1

    def release_read(self):
        with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                with self._write_lock:
                    self._write_cv.notify_all()

    def acquire_write(self):
        with self._write_lock:
            self._writer_waiting += 1
            while self._read_count > 0 or self._write_count > 0:
                self._write_cv.wait()
            self._writer_waiting -= 1
            self._write_count += 1

    def release_write(self):
        with self._write_lock:
            self._write_count -= 1
            self._write_cv.notify_all()
            with self._read_lock:
                self._read_cv.notify_all()

# ---------------- Shared Communication Buffer ----------------
class MessageLog:
    # ... (no changes from previous version) ...
    def __init__(self):
        self._log = []
        self._lock = WriterPreferenceRWLock()

    def write(self, robot_id, event_type, data):
        self._lock.acquire_write()
        try:
            msg_id = len(self._log)
            message = {'id': msg_id, 'robot_id': robot_id, 'type': event_type, 'data': data, 'timestamp': time.time()}
            self._log.append(message)
            return msg_id
        finally:
            self._lock.release_write()

    def read(self, last_known_id):
        # Improvement Point 4: This incremental read is efficient.
        # Slicing the tail of a list is fast, and each robot only
        # requests messages it hasn't seen yet, preventing re-processing the whole log.
        self._lock.acquire_read()
        try:
            return self._log[last_known_id:]
        finally:
            self._lock.release_read()

# ---------------- Robot Thread Class ----------------
class ORMSTCRobot(threading.Thread):
    # ... (no changes from previous version) ...
    def __init__(self, rid, start_pos, R, C, obstacles, message_log, go_home_event, step_interval_sec=0.1):
        super().__init__(name=f"Robot-{rid}", daemon=True)
        self.id = rid
        self.start_pos = start_pos
        self.R, self.C = R, C
        self.step_interval_sec = step_interval_sec

        # Shared resources
        self.message_log = message_log
        self.obstacles = obstacles
        self.stop_event = threading.Event()
        self.go_home_event = go_home_event

        # Robot's internal belief state
        self.state = "exploring" # exploring | returning_home
        self.cur_pos = start_pos
        # Improvement Point 1: The stack consistently stores (parent, current_cell).
        # self.stack[-1][1] is always the robot's current location, and
        # self.stack[-1][0] is the cell it came from. This provides a clear path for backtracking.
        self.stack = [(None, start_pos)]
        self.visited_by = [[None] * C for _ in range(R)]
        self.last_log_id = 0
        
        r, c = start_pos
        self.visited_by[r][c] = self.id

    def run(self):
        self.message_log.write(self.id, 'START', {'pos': self.start_pos})
        while not self.stop_event.is_set():
            if self.go_home_event.is_set() and self.state == "exploring":
                self.state = "returning_home"

            self._update_local_state()
            self._decide_and_act()
            
            if self._is_done():
                self.message_log.write(self.id, 'DONE', {'pos': self.cur_pos})
                break
            time.sleep(self.step_interval_sec)

    def stop(self):
        self.stop_event.set()

    def _update_local_state(self):
        new_messages = self.message_log.read(self.last_log_id)
        for msg in new_messages:
            if msg['type'] in ['START', 'MOVE']:
                r, c = msg['data']['pos']
                if self.visited_by[r][c] is None:
                    self.visited_by[r][c] = msg['robot_id']
        if new_messages:
            self.last_log_id = new_messages[-1]['id'] + 1

    def _get_valid_neighbors(self, pos):
        r, c = pos
        parent, _ = self.stack[-1]
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, W
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (nr, nc) != parent and 0 <= nr < self.R and 0 <= nc < self.C and not self.obstacles[nr][nc]:
                yield (nr, nc)
    
    def _decide_and_act(self):
        # Improvement Point 2: Implicit Collision Resolution.
        # By checking its local `visited_by` map, a robot avoids moving to a known occupied cell.
        # If two robots decide to move to the same free cell simultaneously, the first one
        # to write its 'MOVE' message to the log wins. The losing robot will see the winner's
        # message on its next `_update_local_state` call, update its map, and choose a different path.
        if self.state == "exploring":
            for r, c in self._get_valid_neighbors(self.cur_pos):
                if self.visited_by[r][c] is None:
                    self.stack.append((self.cur_pos, (r, c)))
                    self.cur_pos = (r, c)
                    self.visited_by[r][c] = self.id
                    self.message_log.write(self.id, 'MOVE', {'pos': self.cur_pos})
                    return
            
            if len(self.stack) > 1:
                parent, _ = self.stack.pop()
                self.cur_pos = parent
                self.message_log.write(self.id, 'BACKTRACK', {'pos': self.cur_pos})

        elif self.state == "returning_home":
            if self.cur_pos != self.start_pos and len(self.stack) > 1:
                parent, _ = self.stack.pop()
                self.cur_pos = parent
                self.message_log.write(self.id, 'RETURN', {'pos': self.cur_pos})

    def _is_done(self):
        return self.state == "returning_home" and self.cur_pos == self.start_pos

# ---------------- Simulation Controller Class ----------------
class Simulation:
    def __init__(self, R, C, num_robots, num_obstacles, step_interval_sec):
        self.R, self.C = R, C
        self.num_robots = num_robots
        self.obstacles, self.free_cells = self._generate_connected_obstacles(num_obstacles)
        self.total_free_cells = len(self.free_cells)
        
        self.message_log = MessageLog()
        self.go_home_event = threading.Event()
        self.robots = self._create_robots(step_interval_sec)
        
        # Visualization state
        self._vis_state = {
            'robot_pos': {},
            'robot_status': {i: 'starting' for i in range(num_robots)},
            'tree_edges': [set() for _ in range(num_robots)],
            'global_coverage_set': set() # Improvement Point 3
        }
        self._last_vis_log_id = 0
        self.start_time = 0
        self.end_time = 0

    def _is_grid_connected(self, obstacles, free_cells):
        if not free_cells:
            return False
        q = deque([free_cells[0]])
        visited = {free_cells[0]}
        while q:
            r, c = q.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.R and 0 <= nc < self.C and not obstacles[nr][nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return len(visited) == len(free_cells)

    def _generate_connected_obstacles(self, num_obstacles):
        # Improvement Point 7: Ensure the generated map is fully connected.
        while True:
            obstacles = [[False] * self.C for _ in range(self.R)]
            all_cells = [(r, c) for r in range(self.R) for c in range(self.C)]
            random.shuffle(all_cells)
            for r, c in all_cells[:num_obstacles]:
                obstacles[r][c] = True
            
            free_cells = [cell for cell in all_cells if not obstacles[cell[0]][cell[1]]]
            if self._is_grid_connected(obstacles, free_cells):
                print("Generated a connected map.")
                return obstacles, free_cells

    def _create_robots(self, step_interval_sec):
        # Place robots on the pre-calculated free cells
        robot_starts = random.sample(self.free_cells, self.num_robots)
        return [ORMSTCRobot(i, robot_starts[i], self.R, self.C, self.obstacles, self.message_log, self.go_home_event, step_interval_sec) for i in range(self.num_robots)]

    def run(self, frame_interval_ms=100):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        plt.show()

        print("🚀 Starting simulation...")
        self.start_time = time.time()
        for r in self.robots:
            r.start()

        while not self._check_completion():
            self._draw_frame(ax)
            plt.pause(frame_interval_ms / 1000.0)
        
        self.end_time = time.time()
        print("🛑 Coverage complete. Robots returning to start...")
        
        # Final return phase
        while not all(status == 'done' for status in self._vis_state['robot_status'].values()):
            self._draw_frame(ax)
            plt.pause(frame_interval_ms / 1000.0)
        
        print("🛑 All robots have returned home. Stopping threads.")
        # Improvement Point 6: The shutdown sequence is graceful.
        # Threads exit their loops naturally upon completion. stop_event is a failsafe.
        # join() ensures the main thread waits for them to terminate completely.
        for r in self.robots:
            r.stop()
            r.join(timeout=1.0) # Add a timeout for extra safety
        
        plt.ioff()
        self._draw_frame(ax) # Draw final frame
        plt.show()
        self.generate_report()

    def _check_completion(self):
        if not self.go_home_event.is_set():
            # Improvement Point 3: Use the size of the coverage set for accuracy.
            if len(self._vis_state['global_coverage_set']) >= self.total_free_cells:
                self.go_home_event.set()
        return all(status == 'done' for status in self._vis_state['robot_status'].values())

    def _update_vis_state(self):
        messages = self.message_log.read(self._last_vis_log_id)
        if not messages: return
            
        for msg in messages:
            rid, mtype, data = msg['robot_id'], msg['type'], msg['data']
            pos = data['pos']
            
            if mtype in ['START', 'MOVE']:
                # Improvement Point 3: Update the global coverage set.
                self._vis_state['global_coverage_set'].add(pos)
            
            if mtype == 'START':
                self._vis_state['robot_pos'][rid] = pos
                self._vis_state['robot_status'][rid] = 'alive'
            elif mtype == 'MOVE':
                old_pos = self._vis_state['robot_pos'].get(rid)
                if old_pos:
                    self._vis_state['tree_edges'][rid].add(tuple(sorted((old_pos, pos))))
                self._vis_state['robot_pos'][rid] = pos
            elif mtype in ['BACKTRACK', 'RETURN']:
                self._vis_state['robot_pos'][rid] = pos
            elif mtype == 'DONE':
                self._vis_state['robot_status'][rid] = 'done'
        
        self._last_vis_log_id = messages[-1]['id'] + 1
    
    def _draw_frame(self, ax):
        self._update_vis_state()
        ax.clear()
        
        ax.scatter([c for r,c in self.free_cells], [self.R - 1 - r for r,c in self.free_cells], color='#f0f0f0', s=10)
        for r_idx in range(self.R):
            for c_idx in range(self.C):
                if self.obstacles[r_idx][c_idx]:
                    ax.scatter(c_idx, self.R - 1 - r_idx, color='black', s=15, marker='s')

        for (r_idx, c_idx) in self._vis_state['global_coverage_set']:
             owner = -1
             for rid, robot in enumerate(self.robots):
                 if (r_idx, c_idx) in robot.visited_by:
                     owner = rid
                     break
             color = f"C{owner}" if owner != -1 else '#f0f0f0'
             ax.scatter(c_idx, self.R - 1 - r_idx, color=color, s=10)

        for rid in range(self.num_robots):
            for (u, v) in self._vis_state['tree_edges'][rid]:
                ax.plot([u[1], v[1]], [self.R - 1 - u[0], self.R - 1 - v[0]], color=f"C{rid}", linewidth=1.5, zorder=5)
            sr, sc = self.robots[rid].start_pos
            ax.plot(sc, self.R - 1 - sr, '*', color=f"C{rid}", markersize=18, markeredgecolor='black', zorder=10)
            if self._vis_state['robot_status'].get(rid) == 'alive':
                 rr, cc = self._vis_state['robot_pos'][rid]
                 ax.plot(cc, self.R - 1 - rr, 'o', color=f"C{rid}", markersize=10, markeredgecolor='black', zorder=10)
                 # Improvement Point 5: Add text labels to robots.
                 ax.text(cc, self.R - 1 - rr, str(rid), ha='center', va='center', fontsize=8, color='white', weight='bold', zorder=11)
        
        elapsed_time = time.time() - self.start_time
        coverage_str = f"Coverage: {len(self._vis_state['global_coverage_set'])} / {self.total_free_cells}"
        ax.set_title(f"{coverage_str} | Time: {elapsed_time:.1f}s")
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.gcf().canvas.flush_events()

    def generate_report(self):
        # ... (no changes from previous version) ...
        print("\n--- 📊 Final Simulation Report ---")
        total_time = self.end_time - self.start_time
        print(f"Total Simulation Time: {total_time:.2f} seconds")

        all_messages = self.message_log.read(0)
        cells_covered_by_robot = defaultdict(set)
        moves_by_robot = defaultdict(int)
        
        for msg in all_messages:
            rid = msg['robot_id']
            if msg['type'] in ['START', 'MOVE']:
                cells_covered_by_robot[rid].add(msg['data']['pos'])
            if msg['type'] in ['MOVE', 'BACKTRACK', 'RETURN']:
                moves_by_robot[rid] += 1
        
        total_covered = self._vis_state['global_coverage_set']
            
        print(f"Overall Coverage: {len(total_covered)} / {self.total_free_cells} free cells ({len(total_covered)/self.total_free_cells:.1%})")
        print("\n--- Per-Robot Statistics ---")
        for rid in range(self.num_robots):
            cells_count = len(cells_covered_by_robot[rid])
            moves_count = moves_by_robot[rid]
            print(f"🤖 Robot {rid}:")
            print(f"  - Unique Cells Covered: {cells_count}")
            print(f"  - Total Actions: {moves_count}")


if __name__ == "__main__":
    sim = Simulation(
        R=10,
        C=10,
        num_robots=6,
        num_obstacles=12,
        step_interval_sec=0.5 # How fast each robot "thinks"
    )
    sim.run(frame_interval_ms=50) # How fast the animation updates