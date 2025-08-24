import gym
from gym import spaces
import numpy as np

import torch
from torch import nn

import pandas as pd
from tqdm import tqdm


class Q_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=5),
        )

    def forward(self,x):
        x = x.type(torch.float32)
        return self.func(x)

class SimpleGridWorld(gym.Env):
    def __init__(self, grid_size=5, action_num=5):
        super(SimpleGridWorld, self).__init__()
        self.grid_size = grid_size
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(action_num)
        self.observation_space.n = self.grid_size * self.grid_size
        self.action_space.n = action_num
        self.forbidden = {(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}
        self.goal = (3, 2)
        self.trucate_step = 300
        self.current_step = 0

        self.policy_matrix = np.zeros((self.observation_space.n, self.action_space.n))
        self.value_vector = np.zeros(self.observation_space.n)
        self.q_matrix = np.zeros((self.observation_space.n, self.action_space.n))

        self.gama = 0.95

    def _sample_valid_position(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.forbidden and (x, y) != self.goal:
                return [x, y]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = self._sample_valid_position()
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
            reward = -10
        else:
            self.agent_pos = [nx, ny]

        terminated = tuple(self.agent_pos) == self.goal
        if terminated:
            reward = 1

        truncated = self.current_step >= self.trucate_step

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
    q_net = Q_Net().to('cuda')
    q_net_target = Q_Net().to('cuda')
    q_net_target.load_state_dict(q_net.state_dict())

    obs, info = env.reset()
    gama = env.gama
    action_n = env.action_space.n

    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.0005)
    loss = nn.MSELoss()

    data_package = pd.read_csv('data/sar_log1.csv')

    for epoch in tqdm(range(1000), desc="Learning", unit="times"):
        data_package = data_package.sample(frac=1).reset_index(drop=True)
        for idx, row in data_package.iterrows():
            state = torch.tensor([row['state']//5/4.0, row['state']%5/4.0], dtype=torch.float32).to('cuda')
            action = row['action']
            reward = row['reward']
            state_next = torch.tensor([row['state_next']//5/4.0, row['state_next']%5/4.0], dtype=torch.float32).to('cuda')

            max_q = torch.max(q_net_target(state_next).detach())
            target = reward + gama * max_q

            q_hat = q_net(state)
            q_val = q_hat[action]

            l = loss(q_val, target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            q_net_target.load_state_dict(q_net.state_dict())




    for s in range(25):
        input = torch.tensor([s // 5 / 4.0, s % 5 / 4.0], dtype=torch.float32).to('cuda')
        print(q_net(input))

