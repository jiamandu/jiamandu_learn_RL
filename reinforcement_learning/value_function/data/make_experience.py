import gym
from gym import spaces
import numpy as np
import pandas as pd

class SimpleGridWorld(gym.Env):
    def __init__(self, Grid_Size=5, Action_Space=5):
        super(SimpleGridWorld, self).__init__()
        self.grid_size = Grid_Size
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(n=Action_Space)
        self.observation_space.n = Grid_Size * Grid_Size
        self.action_space.n = Action_Space

        self.forbidden = {(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}
        self.goal = (3, 2)
        self.truncate_step = 300
        self.current_step = 0

    def _sample_valid_position(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.forbidden and (x, y) != self.goal:
                return [x, y]

    def reset(self, *, seed=None, options=None, obs=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = [int(obs / 5), int(obs%5)]
        # self.agent_pos = self._sample_valid_position()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        x, y = self.agent_pos
        nx, ny = x, y

        if action == 0:
            nx -= 1
        elif action == 1:
            ny += 1
        elif action == 2:
            nx += 1
        elif action == 3:
            ny -= 1

        reward = 0

        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            reward = -1
        elif (nx, ny) in self.forbidden:
            self.agent_pos = [nx, ny]
            reward = -1
        else:
            self.agent_pos = [nx, ny]

        terminated = tuple(self.agent_pos) == self.goal
        if terminated:
            reward = 1

        truncated = self.current_step >= self.truncate_step

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ' . ')
        x, y = self.agent_pos
        gx, gy = self.goal

        for fx, fy in self.forbidden:
            grid[fx][fy] = ' X '

        grid[gx][gy] = ' G '
        grid[x][y] = ' A '
        print("\n".join("".join(row) for row in grid))
        print()

    def _get_obs(self):
        x, y = self.agent_pos
        return x * self.grid_size + y

if __name__ == '__main__':
    env = SimpleGridWorld()
    obs, info = env.reset(obs=0)

    done = False
    sar_log = []

    times = 1
    for i in range(times):
        done = 0
        obs, info = env.reset(obs=0)
        action = 0
        for obs in range(25):
            print('!')
            for action in range(5):
                obs, info = env.reset(obs = obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                sar_log.append({
                    'state': obs,
                    'action': action,
                    'reward': reward,
                    'state_next':next_obs
                })
                # env.render()
                done = terminated or truncated

    df = pd.DataFrame(sar_log)
    df.to_csv(f'sar_log{times}.csv', index=False)
    print("SAR log saved to sar_log.csv")
