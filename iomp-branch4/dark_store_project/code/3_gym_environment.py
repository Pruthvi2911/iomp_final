"""
Dark Store Project - Gymnasium Environment
Step 3: Create the grid world for PPO training
"""

import gymnasium as gym #Creates RL environment.
from gymnasium import spaces #Action Space and Observation Space
import numpy as np #For Mathematical operations
import json #For Data Storage 
from collections import deque #For Breadth-First Search (BFS)
from pathlib import Path #For Path Manipulation

class DarkStoreEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    WALKWAY_COLS = {1, 4, 7, 10}

    def __init__(self, layout_file, max_steps=350, max_items_per_order=3):
        super(DarkStoreEnv, self).__init__()

        self.grid_rows = 13
        self.grid_cols = 13
        self.depot_position = (0, 6)
        self.max_steps = max_steps
        self.max_items_per_order = max_items_per_order

        # Build walkable cell set
        self.walkable = set()
        for c in range(self.grid_cols):
            self.walkable.add((0, c))   # top cross-aisle
            self.walkable.add((12, c))  # bottom cross-aisle
        for r in range(1, 12):
            for c in self.WALKWAY_COLS:
                self.walkable.add((r, c))

        # Load layout - Pathlib fix applied
        with open(layout_file, 'r') as f:
            self.layout = json.load(f)

        self.product_positions = {}
        for prod_id, info in self.layout.items():
            self.product_positions[int(prod_id)] = (info['row'], info['col'])

        self.all_product_ids = list(self.product_positions.keys())
        self._bfs_dist = self._precompute_bfs() #BFS Precomputation to find the shortest path between any two points

        self._shelf_pick_spots = {}
        for pid, shelf in self.product_positions.items():
            r, c = shelf
            spots = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in self.walkable:
                    spots.append(nb)
            self._shelf_pick_spots[pid] = spots

        self.action_space = spaces.Discrete(4) # 0: up, 1: down, 2: left, 3: right
        obs_size = 2 + (self.max_items_per_order * 3) + 1
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_rows, self.grid_cols, max_steps),
            shape=(obs_size,),
            dtype=np.float32 # Data Type
        )

        self.current_position = None # Current position of the robot
        self.order_items = None # Items to be picked up 
        self.items_remaining = None # Items remaining to be picked up 
        self.steps_taken = None # Steps taken by the robot 
        self.visited_positions = None # Positions visited by the robot 
        self._prev_min_dist = None

    def _precompute_bfs(self):
        dist = {}
        for start in self.walkable:
            d = {start: 0}
            q = deque([start])
            while q:
                node = q.popleft()
                nr, nc = node
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (nr + dr, nc + dc)
                    if (0 <= nb[0] < self.grid_rows and
                        0 <= nb[1] < self.grid_cols and
                        nb in self.walkable and nb not in d):
                        d[nb] = d[node] + 1
                        q.append(nb)
            dist[start] = d
        return dist
    
    #obs_size = 2 + (3*3) + 1 is the size of the observation space
    #2 is for the current position of the robot 
    #3*3 is for the 3 items to be picked up and each item has 3 values (x,y,z coordinates)
    #1 is for the steps taken by the robot

    def _nav_dist_to_item(self, from_pos, item_id):
        spots = self._shelf_pick_spots.get(item_id, [])
        if not spots: return 999
        from_pos = tuple(from_pos)
        bfs_from = self._bfs_dist.get(from_pos, {})
        return min((bfs_from.get(s, 999) for s in spots), default=999)

    def _nav_dist_to_depot(self, from_pos):
        from_pos = tuple(from_pos)
        bfs_from = self._bfs_dist.get(from_pos, {})
        return bfs_from.get(self.depot_position, 999)

    def _min_dist_to_target(self):
        pos = tuple(self.current_position)
        if self.items_remaining:
            return min(self._nav_dist_to_item(pos, i) for i in self.items_remaining)
        return self._nav_dist_to_depot(pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_position = list(self.depot_position)
        
        # Weighted sampling based on frequency
        probs = [self.layout[str(pid)].get('frequency', 1) for pid in self.all_product_ids]
        prob_sum = sum(probs)
        normalized_probs = [p / prob_sum for p in probs]

        self.order_items = list(self.np_random.choice(
            self.all_product_ids, size=self.max_items_per_order, replace=False, p=normalized_probs
        ))

        self.items_remaining = set(self.order_items)
        self.steps_taken = 0
        self.visited_positions = {tuple(self.current_position)}
        self._prev_min_dist = self._min_dist_to_target()
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = False

        r, c = self.current_position
        if action == 0: nr, nc = r - 1, c
        elif action == 1: nr, nc = r + 1, c
        elif action == 2: nr, nc = r, c - 1
        elif action == 3: nr, nc = r, c + 1
        else: nr, nc = r, c

        nr = max(0, min(self.grid_rows - 1, nr))
        nc = max(0, min(self.grid_cols - 1, nc))

        if (nr, nc) in self.walkable:
            self.current_position = [nr, nc]
            reward -= 0.1 # Penalty for moving Encourages shorter routes.
        else:
            reward -= 0.3 # Wall penalty to discourage hitting walls

        current_pos = tuple(self.current_position)
        curr_dist = self._min_dist_to_target()
        reward += 0.8 * (self._prev_min_dist - curr_dist) # Reward for getting closer to the target

        for item_id in list(self.items_remaining):
            if current_pos in self._shelf_pick_spots.get(item_id, []): #Check if current position is the pick up spot for the item
                self.items_remaining.remove(item_id) #Remove the item from the remaining items
                reward += 50 # Reward for picking up an item
                if len(self.items_remaining) == 0:
                    reward += 25 # Bonus for picking up the last item

        self._prev_min_dist = self._min_dist_to_target()
        self.visited_positions.add(current_pos)

        if not self.items_remaining and current_pos == self.depot_position:
            reward += 100 # Bonus for returning to depot
            terminated = True # Episode ends when all items are picked up and robot returns to depot

        if self.steps_taken >= self.max_steps:
            truncated = True # Maximum steps reached

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        obs[0] = self.current_position[0] / self.grid_rows
        obs[1] = self.current_position[1] / self.grid_cols

        for i, item_id in enumerate(self.order_items):
            base = 2 + i * 3
            if item_id in self.items_remaining:
                item_pos = self.product_positions[item_id]
                obs[base] = 1.0
                obs[base + 1] = item_pos[0] / self.grid_rows
                obs[base + 2] = item_pos[1] / self.grid_cols
        
        obs[-1] = self.steps_taken / self.max_steps
        return obs

    def _get_info(self):
        return {
            'items_remaining': len(self.items_remaining),
            'steps_taken': self.steps_taken,
            'success': (len(self.items_remaining) == 0 and tuple(self.current_position) == self.depot_position)
        }

    def render(self, mode='human'):
        # ... (keep your render logic, it's good for debugging)
        pass

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    layout_path = BASE_DIR / "results" / "optimized_layout.json"
    env = DarkStoreEnv(str(layout_path))
    print("Environment Loaded Successfully!")