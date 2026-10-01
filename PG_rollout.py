import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

class PGNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4,8)
    self.fc2 = nn.Linear(8,4)
    self.fc3 = nn.Linear(4,2)

  def forward(self,a):
    a = self.fc1(a)
    a = torch.relu(a)
    a = self.fc2(a)
    a = torch.relu(a)
    a = self.fc3(a)
    return a

class RolloutBuffer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.states = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
        self.log_probs = [[] for _ in range(num_envs)]
        self.dones = [[] for _ in range(num_envs)]

    def add(self,env_idx,state,action,reward,log_prob,done):
        self.states[env_idx].append(state)
        self.actions[env_idx].append(action)
        self.rewards[env_idx].append(reward)
        self.log_probs[env_idx].append(log_prob)
        self.dones[env_idx].append(done)

    def clear(self):
        self.states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.log_probs = [[] for _ in range(self.num_envs)]
        self.dones = [[] for _ in range(self.num_envs)]

  class VPG:
    def __init__(self, model, env, gamma,lr=0.001):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        self.buffer = RolloutBuffer(self.env.num_envs)

    def get_return(self, rewards):
        returns = [0] * len(rewards)
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        return returns


    def collect_rollout(self):
        num_envs = self.env.num_envs
        obs, _ = self.env.reset()
        finished = np.zeros(num_envs,dtype=bool)

        episode_rewards = [[] for _ in range(num_envs)]

        while not np.all(finished):
            obs_tensor = torch.tensor(obs,dtype=torch.float32,device=self.device)

            logits = self.model(obs_tensor)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            next_obs, rewards, terminated, truncated, _ = \
                self.env.step(actions.cpu().numpy())

            dones = terminated | truncated

            for i in range(num_envs):
                if finished[i]:
                    continue

                self.buffer.add(env_idx=i,state=obs_tensor[i],action=actions[i],reward=rewards[i],log_prob=log_probs[i],done=dones[i])

                episode_rewards[i].append(rewards[i])

                if dones[i]:
                    finished[i] = True
            obs = next_obs

        return [sum(x) for x in episode_rewards]

    def update(self):
        total_loss = 0

        total_transitions = 0
        for i in range(self.env.num_envs):
            rewards = self.buffer.rewards[i]

            log_probs = self.buffer.log_probs[i]

            returns = self.get_return(rewards)
            returns = torch.tensor(returns,dtype=torch.float32,device=self.device)
            log_probs = torch.stack(log_probs)

            loss = -torch.sum(log_probs * returns)
            total_loss += loss
            total_transitions += len(rewards)

        total_loss /= total_transitions

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.buffer.clear()

    def train(self, num_episode):
        rewards = []

        for i in range(num_episode):
            episode_rewards = self.collect_rollout()

            self.update()

            reward = np.mean(episode_rewards)

            rewards.append(reward)

            if i % 50 == 0:
                print(f"Update: {i}, "f"Reward: {reward:.2f}")

        rewards_numpy = np.array(rewards)
        window_size = 5
        moving_average = np.convolve(rewards_numpy,np.ones(window_size) / window_size,mode='valid')
        return moving_average
