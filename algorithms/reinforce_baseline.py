"""REINFORCE + Value Baseline

Value network V(s)를 함께 학습하고, A_t = G_t - V(s_t)를 advantage 추정값으로 사용해
policy gradient의 variance를 줄인다.

    policy loss = -mean( log pi(a_t|s_t) * A_t.detach() )
    value loss  =  mean( (G_t - V(s_t))^2 )

블로그: https://jihoai.github.io/posts/Policy-Gradient-구현-2/
실행:   python -m algorithms.reinforce_baseline --num-envs 8 --lr 0.08 --seed 0
        (--num-envs 1 이면 단일 환경)
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from common.networks import MLP
from common.utils import discounted_returns, make_vec_env, save_results, set_seed


class ReinforceBaseline:
    def __init__(self, envs, policy, value, gamma=0.99, lr=0.08, batch_size=5000, device="cpu"):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.policy = policy.to(device)
        self.value = value.to(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    @staticmethod
    def compute_policy_loss(log_probs, advantage):
        # advantage를 detach하지 않으면 policy loss의 gradient가 value network로도 흘러간다.
        return -(log_probs * advantage.detach()).mean()

    @staticmethod
    def compute_value_loss(advantage):
        return (advantage ** 2).mean()

    def collect_batch(self, seed=None):
        """끝난 에피소드의 transition이 batch_size를 넘을 때까지 모은다. (끝나지 않은 에피소드는 버린다)"""
        log_probs_batch, values_batch, returns_batch = [], [], []
        ep_rewards = [[] for _ in range(self.num_envs)]
        ep_log_probs = [[] for _ in range(self.num_envs)]
        ep_values = [[] for _ in range(self.num_envs)]
        finished_episode_rewards = []

        obs, _ = self.envs.reset(seed=seed)
        while len(returns_batch) <= self.batch_size:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist = torch.distributions.Categorical(logits=self.policy(obs_t))
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = self.value(obs_t).squeeze(-1)  # (num_envs, 1) -> (num_envs,)

            obs, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
            dones = terminated | truncated

            for i in range(self.num_envs):
                ep_rewards[i].append(rewards[i])
                ep_log_probs[i].append(log_probs[i])
                ep_values[i].append(values[i])

                if dones[i]:
                    returns_batch.extend(discounted_returns(ep_rewards[i], self.gamma))
                    log_probs_batch.extend(ep_log_probs[i])
                    values_batch.extend(ep_values[i])
                    finished_episode_rewards.append(sum(ep_rewards[i]))
                    ep_rewards[i], ep_log_probs[i], ep_values[i] = [], [], []

        return (
            torch.stack(log_probs_batch),
            torch.stack(values_batch),
            torch.tensor(returns_batch, dtype=torch.float32, device=self.device),
            finished_episode_rewards,
        )

    def train_one_epoch(self, seed=None):
        log_probs, values, returns, episode_rewards = self.collect_batch(seed)
        advantage = returns - values  # 모두 (N,) 모양이어야 원소별로 계산된다.

        policy_loss = self.compute_policy_loss(log_probs, advantage)
        value_loss = self.compute_value_loss(advantage)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return episode_rewards

    def train(self, epochs, seed=None):
        history = []
        for epoch in range(epochs):
            episode_rewards = self.train_one_epoch(seed=seed if epoch == 0 else None)
            history.extend(episode_rewards)
            print(
                f"Epoch {epoch + 1:3d} | Episodes {len(episode_rewards):3d} | "
                f"Mean Reward {np.mean(episode_rewards):.2f}"
            )
        return history


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE with value baseline on CartPole")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50, help="업데이트 횟수")
    p.add_argument("--batch-size", type=int, default=5000, help="업데이트 1번에 사용할 최소 transition 수")
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-dir", default=None, help="기본값: results/reinforce_baseline_lr{lr}_seed{seed}")
    p.add_argument("--gif", action="store_true", help="학습 후 에이전트 GIF 저장")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    envs = make_vec_env(args.env_id, args.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = ReinforceBaseline(
        envs,
        policy=MLP(obs_dim, n_actions, hidden=(8, 4)),
        value=MLP(obs_dim, 1, hidden=(8, 4)),
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    history = agent.train(args.epochs, seed=args.seed)

    save_dir = args.save_dir or os.path.join("results", f"reinforce_baseline_lr{args.lr}_seed{args.seed}")
    save_results(history, save_dir, f"REINFORCE + Baseline (lr={args.lr})", agent.policy, args.gif, args.env_id, args.device, args.seed)


if __name__ == "__main__":
    main()
