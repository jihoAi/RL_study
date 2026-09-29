import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class PNetwork(nn.Module):
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

class PG():
  def __init__(self,model,env,gamma,lr=0.001):
    self.model = model
    self.env = env
    self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
    self.gamma = gamma
    self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    self.model.to(self.device)

  def get_return(self, rewards):
    G = [0] * len(rewards)
    return_ = 0

    for i in reversed(range(len(rewards))):
        return_ = rewards[i] + self.gamma * return_
        G[i] = return_

    return G

  def do_episode(self):
    state_list = []
    reward_list = []
    action_list = []
    log_prob_list = []
    terminated, truncated = False, False
    obs, _ = self.env.reset()
    while not (terminated or truncated):
      state_list.append(obs)
      obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
      logits = self.model(obs)
      dist = torch.distributions.Categorical(logits=logits)
      action = dist.sample()
      log_prob = dist.log_prob(action)
      obs, reward, terminated, truncated, _ = self.env.step(action.item())

      reward_list.append(reward)
      action_list.append(action)
      log_prob_list.append(log_prob)
    G = self.get_return(reward_list)

    loss = 0

    G_tensor = torch.tensor(G, dtype=torch.float32,device=self.device)
    log_prob_tensor = torch.stack(log_prob_list)

    loss = -torch.sum(log_prob_tensor*G_tensor)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return sum(reward_list)

  def train(self,num_episode):
    rewards = []
    for i in range(num_episode):
      rewards.append(self.do_episode())
    rewards_numpy = np.array(rewards)
    window_size = 5
    moving_average = np.convolve(rewards_numpy, np.ones(window_size)/window_size, mode='valid')
    return moving_average
