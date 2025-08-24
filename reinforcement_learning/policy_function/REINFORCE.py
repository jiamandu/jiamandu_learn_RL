import gym
from gym import spaces

import numpy as np

import torch
from torch import nn


class Policy_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(in_features=2, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=125),
            nn.ReLU(),
            nn.Linear(in_features=125, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=5),
            nn.Softmax(dim=-1)
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

        self.gama = 0.99
        self.epsilon = 0.5
        self.alpha = 0.01

        self.policy_matrix = np.zeros((self.observation_space.n, self.action_space.n))
        self.value_vector = np.zeros(self.observation_space.n)
        self.q_matrix = np.zeros((self.observation_space.n, self.action_space.n))

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
        # self.agent_pos = [0,0]
        return self._get_obs(), {}

    def make_policy_matrix(self, policy_net: Policy_Net):
        for obs in range(self.observation_space.n):
            input_tensor = torch.tensor([(obs//5)/4,(obs%5)/4], dtype=torch.float32).to('cuda')
            with torch.no_grad():
                output_tensor = policy_net(input_tensor)
                output_np = output_tensor.cpu().numpy()
            self.policy_matrix[obs] = output_np

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

def compute_returns(episode, gama):
    returns = []
    G = 0
    for _, _, _, reward, _ in reversed(episode):
        G = reward + gama * G
        returns.insert(0, G)
    return returns

if __name__ == '__main__':
    env = SimpleGridWorld()
    policy_net = Policy_Net().to('cuda')
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.005)

    gama = env.gama
    num_epochs = 5000
    batch_size = 10  # 多 episode 训练

    for epoch in range(num_epochs):
        all_episodes = []
        for _ in range(batch_size):
            episode = []
            obs, _ = env.reset()
            done = False
            t = 0
            while not done:
                input = torch.tensor([(obs//5)/4,(obs%5)/4], dtype=torch.float32).to('cuda')
                output = policy_net(input)
                action = np.random.choice(np.arange(5), p=output.detach().cpu().numpy())
                obs_next, reward, terminated, truncated, _ = env.step(action)
                episode.append((t, obs, action, reward, obs_next))
                obs = obs_next
                done = terminated or truncated
                t += 1
            all_episodes.append(episode)

        # 更新策略
        total_loss = 0
        for episode in all_episodes:
            returns = compute_returns(episode, gama)
            for i, (t, obs, action, reward, obs_next) in enumerate(episode):
                G = returns[i]
                input = torch.tensor([(obs//5)/4,(obs%5)/4], dtype=torch.float32).to('cuda')
                output = policy_net(input)
                log_prob = torch.log(output[action])
                total_loss += - G * log_prob

        total_loss = total_loss / batch_size
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Avg loss: {total_loss.item():.4f}, Avg len: {np.mean([len(e) for e in all_episodes]):.2f}")

    env.make_policy_matrix(policy_net)
    print(env.policy_matrix)
