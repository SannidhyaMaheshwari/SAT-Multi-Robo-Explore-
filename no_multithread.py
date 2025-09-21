import threading
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- Concurrency Primitives ----------------
class WriterPreferenceRWLock:
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
        self._lock.acquire_read()
        try:
            return self._log[last_known_id:]
        finally:
            self._lock.release_read()

# ---------------- Robot Thread Class ----------------
class ORMSTCRobot(threading.Thread):
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
        self.obstacles = self._generate_obstacles(num_obstacles)
        self.total_free_cells = sum(1 for r in range(R) for c in range(C) if not self.obstacles[r][c])
        
        self.message_log = MessageLog()
        self.go_home_event = threading.Event()
        self.robots = self._create_robots(step_interval_sec)
        
        # Visualization state
        self._vis_state = {
            'visited_by': [[None] * C for _ in range(R)],
            'robot_pos': {},
            'robot_status': {i: 'starting' for i in range(num_robots)},
            'tree_edges': [set() for _ in range(num_robots)]
        }
        self._last_vis_log_id = 0
        self.frame_count = 0
        self.start_time = 0
        self.end_time = 0

    def _generate_obstacles(self, num_obstacles):
        obstacles = [[False] * self.C for _ in range(self.R)]
        all_cells = [(r, c) for r in range(self.R) for c in range(self.C)]
        random.shuffle(all_cells)
        for r, c in all_cells[:num_obstacles]:
            obstacles[r][c] = True
        return obstacles

    def _create_robots(self, step_interval_sec):
        free_cells = [(r,c) for r in range(self.R) for c in range(self.C) if not self.obstacles[r][c]]
        if len(free_cells) < self.num_robots:
            raise ValueError("Not enough free cells for robots.")
        random.shuffle(free_cells)
        return [ORMSTCRobot(i, free_cells[i], self.R, self.C, self.obstacles, self.message_log, self.go_home_event, step_interval_sec) for i in range(self.num_robots)]

    def run(self, frame_interval_ms=100):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        plt.show()

        print("🚀 Starting simulation...")
        self.start_time = time.time()
        for r in self.robots:
            r.start()

        while True:
            self.frame_count += 1
            self._draw_frame(ax)
            plt.pause(frame_interval_ms / 1000.0)
            if self._check_completion():
                break
        
        self.end_time = time.time()
        print("🛑 Coverage complete. Robots returning to start...")
        
        # Final return phase
        while not all(status == 'done' for status in self._vis_state['robot_status'].values()):
            self.frame_count += 1
            self._draw_frame(ax)
            plt.pause(frame_interval_ms / 1000.0)
        
        print("🛑 All robots have returned home. Stopping threads.")
        for r in self.robots:
            r.stop()
        
        plt.ioff()
        self._draw_frame(ax) # Draw final frame
        plt.show()
        self.generate_report()

    def _check_completion(self):
        # Check if coverage is complete to trigger "go home"
        if not self.go_home_event.is_set():
            covered_count = sum(1 for row in self._vis_state['visited_by'] for cell in row if cell is not None)
            if covered_count >= self.total_free_cells:
                self.go_home_event.set()
        # The simulation itself ends when all robots report 'done'
        return all(status == 'done' for status in self._vis_state['robot_status'].values())

    def _update_vis_state(self):
        messages = self.message_log.read(self._last_vis_log_id)
        if not messages: return
            
        for msg in messages:
            rid, mtype, data = msg['robot_id'], msg['type'], msg['data']
            pos = data['pos']
            
            if mtype == 'START':
                self._vis_state['robot_pos'][rid] = pos
                self._vis_state['robot_status'][rid] = 'alive'
                self._vis_state['visited_by'][pos[0]][pos[1]] = rid
            elif mtype == 'MOVE':
                old_pos = self._vis_state['robot_pos'][rid]
                self._vis_state['visited_by'][pos[0]][pos[1]] = rid
                self._vis_state['robot_pos'][rid] = pos
                self._vis_state['tree_edges'][rid].add(tuple(sorted((old_pos, pos))))
            elif mtype in ['BACKTRACK', 'RETURN']:
                self._vis_state['robot_pos'][rid] = pos
            elif mtype == 'DONE':
                self._vis_state['robot_status'][rid] = 'done'
        self._last_vis_log_id = messages[-1]['id'] + 1
    
    def _draw_frame(self, ax):
        self._update_vis_state()
        ax.clear()
        
        # Point-based background
        for r_idx in range(self.R):
            for c_idx in range(self.C):
                if self.obstacles[r_idx][c_idx]:
                    ax.scatter(c_idx, self.R - 1 - r_idx, color='black', s=15, marker='s')
                else:
                    owner = self._vis_state['visited_by'][r_idx][c_idx]
                    color = f"C{owner}" if owner is not None else '#f0f0f0'
                    ax.scatter(c_idx, self.R - 1 - r_idx, color=color, s=10)

        for rid in range(self.num_robots):
            for (u, v) in self._vis_state['tree_edges'][rid]:
                ax.plot([u[1], v[1]], [self.R - 1 - u[0], self.R - 1 - v[0]], color=f"C{rid}", linewidth=1.5)
            sr, sc = self.robots[rid].start_pos
            ax.plot(sc, self.R - 1 - sr, '*', color=f"C{rid}", markersize=18, markeredgecolor='black', zorder=10)
            if self._vis_state['robot_status'].get(rid) == 'alive':
                 rr, cc = self._vis_state['robot_pos'][rid]
                 ax.plot(cc, self.R - 1 - rr, 'o', color=f"C{rid}", markersize=10, markeredgecolor='black', zorder=10)
        
        elapsed_time = time.time() - self.start_time
        ax.set_title(f"Threaded ORMSTC Simulation | Step: {self.frame_count} | Time: {elapsed_time:.1f}s")
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.gcf().canvas.flush_events()

    def generate_report(self):
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
        
        total_covered = set()
        for cells in cells_covered_by_robot.values():
            total_covered.update(cells)
            
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
        num_robots=5,
        num_obstacles=10,
        step_interval_sec=0.5 # How fast each robot "thinks"
    )
    sim.run(frame_interval_ms=500) # How fast the animation updates