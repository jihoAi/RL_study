"""REINFORCE (Vanilla Policy Gradient)

한 에피소드를 끝까지 진행한 뒤, reward-to-go G_t와 log probability로 정책을 업데이트한다.

    loss = -sum_t log pi(a_t|s_t) * G_t

블로그: https://jihoai.github.io/posts/Policy-Gradient-구현-1/
실행:   python -m algorithms.reinforce --lr 0.001 --seed 0
"""

import argparse
import os

import gymnasium as gym
import torch
import torch.optim as optim

from common.networks import MLP
from common.utils import discounted_returns, save_results, set_seed


class Reinforce:
    def __init__(self, env, policy, gamma=0.9, lr=1e-3, device="cpu"):
        self.env = env
        self.policy = policy.to(device)
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def run_episode(self, seed=None):
        """에피소드 하나를 진행하고 정책을 한 번 업데이트한다. 에피소드 reward 합을 반환."""
        obs, _ = self.env.reset(seed=seed)
        rewards, log_probs = [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist = torch.distributions.Categorical(logits=self.policy(obs_t))
            action = dist.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action.item())

            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            done = terminated or truncated

        returns = torch.tensor(discounted_returns(rewards, self.gamma), dtype=torch.float32, device=self.device)
        loss = -torch.sum(torch.stack(log_probs) * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return sum(rewards)

    def train(self, num_episodes, seed=None, log_interval=100):
        history = []
        for ep in range(num_episodes):
            # 첫 에피소드에서만 환경 seed를 준다. 이후에는 환경 내부 난수가 이어서 사용된다.
            history.append(self.run_episode(seed=seed if ep == 0 else None))
            if (ep + 1) % log_interval == 0:
                recent = history[-log_interval:]
                print(f"Episode {ep + 1:5d} | Mean Reward {sum(recent) / len(recent):.2f}")
        return history


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE on CartPole")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-dir", default=None, help="기본값: results/reinforce_lr{lr}_seed{seed}")
    p.add_argument("--gif", action="store_true", help="학습 후 에이전트 GIF 저장")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Reinforce(env, MLP(obs_dim, n_actions, hidden=(8, 4)), gamma=args.gamma, lr=args.lr, device=args.device)
    history = agent.train(args.episodes, seed=args.seed)

    save_dir = args.save_dir or os.path.join("results", f"reinforce_lr{args.lr}_seed{args.seed}")
    save_results(history, save_dir, f"REINFORCE (lr={args.lr})", agent.policy, args.gif, args.env_id, args.device, args.seed)


if __name__ == "__main__":
    main()
