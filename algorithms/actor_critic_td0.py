"""TD(0) Actor-Critic

Critic V(s)의 TD error를 advantage 추정값으로 사용한다.

    delta_t     = r_t + gamma * V(s_{t+1}) - V(s_t)      (terminated면 V(s_{t+1}) = 0)
    actor loss  = -mean( log pi(a_t|s_t) * delta_t.detach() )
    critic loss =  mean( (td_target - V(s_t))^2 )

transition을 update_every개 모을 때마다 업데이트한다.
초기 에피소드가 15~20 step 정도로 짧기 때문에, 5000개처럼 크게 모으면 업데이트가 너무 드물어
학습이 느렸다. 32개로 줄였을 때 학습이 개선되었다. (블로그 참고)

블로그: https://jihoai.github.io/posts/Actor-Critic/
실행:   python -m algorithms.actor_critic_td0 --update-every 32 --seed 0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from common.networks import MLP
from common.utils import make_vec_env, save_results, set_seed


class ActorCriticTD0:
    def __init__(self, envs, policy, value, gamma=0.99, actor_lr=1e-3, critic_lr=2e-3, update_every=32, device="cpu"):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.policy = policy.to(device)
        self.value = value.to(device)
        self.gamma = gamma
        self.update_every = update_every
        self.device = device
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=critic_lr)

    def update(self, log_probs, values, td_targets):
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        td_targets = torch.stack(td_targets)
        td_errors = td_targets - values

        policy_loss = -(log_probs * td_errors.detach()).mean()
        value_loss = F.mse_loss(values, td_targets)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, total_timesteps, seed=None, log_interval=100):
        history = []
        ep_reward = np.zeros(self.num_envs)
        log_probs_buf, values_buf, targets_buf = [], [], []

        obs, _ = self.envs.reset(seed=seed)
        for _ in range(total_timesteps // self.num_envs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist = torch.distributions.Categorical(logits=self.policy(obs_t))
            actions = dist.sample()
            values = self.value(obs_t).squeeze(-1)

            next_obs, rewards, terminated, truncated, info = self.envs.step(actions.cpu().numpy())

            # SAME_STEP 모드에서는 끝난 환경의 next_obs가 이미 reset된 관측이므로,
            # TD target 계산에는 info["final_obs"]에 담긴 실제 마지막 관측을 써야 한다.
            real_next_obs = next_obs.copy()
            if "final_obs" in info:
                for i in np.where(info["_final_obs"])[0]:
                    real_next_obs[i] = info["final_obs"][i]

            with torch.no_grad():
                next_values = self.value(torch.as_tensor(real_next_obs, dtype=torch.float32, device=self.device)).squeeze(-1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                # terminated(막대가 쓰러짐)일 때만 미래 가치를 0으로 둔다.
                # truncated(500 step 시간 제한)는 진짜 종료가 아니므로 V(s_{t+1})로 bootstrap한다.
                not_terminal = torch.as_tensor(~terminated, dtype=torch.float32, device=self.device)
                td_targets = rewards_t + self.gamma * next_values * not_terminal

            log_probs_buf.extend(dist.log_prob(actions))
            values_buf.extend(values)
            targets_buf.extend(td_targets)

            if len(log_probs_buf) >= self.update_every:
                self.update(log_probs_buf, values_buf, targets_buf)
                log_probs_buf, values_buf, targets_buf = [], [], []

            ep_reward += rewards
            for i in np.where(terminated | truncated)[0]:
                history.append(ep_reward[i])
                ep_reward[i] = 0
                if len(history) % log_interval == 0:
                    print(f"Episode {len(history):5d} | Mean Reward {np.mean(history[-log_interval:]):.2f}")

            obs = next_obs
        return history


def parse_args():
    p = argparse.ArgumentParser(description="TD(0) Actor-Critic on CartPole")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--update-every", type=int, default=32, help="업데이트 1번에 사용할 transition 수")
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=2e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-dir", default=None, help="기본값: results/actor_critic_td0_n{update_every}_seed{seed}")
    p.add_argument("--gif", action="store_true", help="학습 후 에이전트 GIF 저장")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    envs = make_vec_env(args.env_id, args.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = ActorCriticTD0(
        envs,
        policy=MLP(obs_dim, n_actions, hidden=(8, 4)),
        value=MLP(obs_dim, 1, hidden=(8, 4, 2)),
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        update_every=args.update_every,
        device=args.device,
    )
    history = agent.train(args.total_timesteps, seed=args.seed)

    save_dir = args.save_dir or os.path.join("results", f"actor_critic_td0_n{args.update_every}_seed{args.seed}")
    save_results(history, save_dir, f"TD(0) Actor-Critic (update every {args.update_every})", agent.policy, args.gif, args.env_id, args.device, args.seed)


if __name__ == "__main__":
    main()
