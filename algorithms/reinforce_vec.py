"""REINFORCE + 병렬 환경 (Rollout Buffer)

여러 환경을 동시에 실행해 환경마다 에피소드를 하나씩 끝까지 모은 뒤, 한 번에 업데이트한다.
trajectory 길이에 따라 gradient 크기가 달라지지 않도록 loss를 전체 transition 수로 나눈다.

블로그: https://jihoai.github.io/posts/Policy-Gradient-구현-1/#5-개선점
실행:   python -m algorithms.reinforce_vec --num-envs 8 --lr 0.003 --seed 0
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from common.networks import MLP
from common.utils import discounted_returns, make_vec_env, save_results, set_seed


class RolloutBuffer:
    """환경별로 reward와 log probability를 따로 저장한다."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.clear()

    def add(self, env_idx, reward, log_prob):
        self.rewards[env_idx].append(reward)
        self.log_probs[env_idx].append(log_prob)

    def clear(self):
        self.rewards = [[] for _ in range(self.num_envs)]
        self.log_probs = [[] for _ in range(self.num_envs)]


class VectorizedReinforce:
    def __init__(self, envs, policy, gamma=0.9, lr=3e-3, device="cpu"):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.policy = policy.to(device)
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer(self.num_envs)

    def collect_rollout(self, seed=None):
        """모든 환경에서 에피소드가 하나씩 끝날 때까지 진행한다. 환경별 에피소드 reward 합을 반환."""
        obs, _ = self.envs.reset(seed=seed)
        finished = np.zeros(self.num_envs, dtype=bool)

        while not finished.all():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist = torch.distributions.Categorical(logits=self.policy(obs_t))
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            obs, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
            dones = terminated | truncated

            for i in range(self.num_envs):
                if finished[i]:  # 이미 에피소드가 끝난 환경은 이번 rollout에서 제외
                    continue
                self.buffer.add(i, rewards[i], log_probs[i])
                if dones[i]:
                    finished[i] = True

        return [sum(r) for r in self.buffer.rewards]

    def update(self):
        total_loss = 0.0
        total_transitions = 0
        for i in range(self.num_envs):
            returns = torch.tensor(
                discounted_returns(self.buffer.rewards[i], self.gamma), dtype=torch.float32, device=self.device
            )
            log_probs = torch.stack(self.buffer.log_probs[i])
            total_loss = total_loss - torch.sum(log_probs * returns)
            total_transitions += len(returns)

        total_loss = total_loss / total_transitions

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def train(self, num_updates, seed=None, log_interval=50):
        history = []  # 에피소드별 reward (업데이트 1번 = 에피소드 num_envs개)
        for u in range(num_updates):
            episode_rewards = self.collect_rollout(seed=seed if u == 0 else None)
            self.update()
            history.extend(episode_rewards)
            if (u + 1) % log_interval == 0:
                print(f"Update {u + 1:4d} | Mean Reward {np.mean(episode_rewards):.2f}")
        return history


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE with vectorized envs on CartPole")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--updates", type=int, default=625, help="업데이트 횟수 (기본 625 x 8개 환경 = 5000 에피소드)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-dir", default=None, help="기본값: results/reinforce_vec_lr{lr}_seed{seed}")
    p.add_argument("--gif", action="store_true", help="학습 후 에이전트 GIF 저장")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    envs = make_vec_env(args.env_id, args.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = VectorizedReinforce(envs, MLP(obs_dim, n_actions, hidden=(8, 4)), gamma=args.gamma, lr=args.lr, device=args.device)
    history = agent.train(args.updates, seed=args.seed)

    save_dir = args.save_dir or os.path.join("results", f"reinforce_vec_lr{args.lr}_seed{args.seed}")
    save_results(history, save_dir, f"REINFORCE, {args.num_envs} envs (lr={args.lr})", agent.policy, args.gif, args.env_id, args.device, args.seed)


if __name__ == "__main__":
    main()
