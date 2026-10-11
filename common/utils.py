import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed):
    """python, numpy, torch의 난수를 한 번에 고정한다. (환경 seed는 reset(seed=...)로 따로 준다)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma * G_{t+1} 를 뒤에서부터 계산한다. O(n)"""
    returns = [0.0] * len(rewards)
    g = 0.0
    for i in reversed(range(len(rewards))):
        g = rewards[i] + gamma * g
        returns[i] = g
    return returns


def make_vec_env(env_id, num_envs):
    """에피소드가 끝난 step에서 바로 reset되는(SAME_STEP) 벡터 환경을 만든다.

    gymnasium 1.x의 기본값(NEXT_STEP)은 에피소드가 끝난 '다음' step에서 reset을 하면서
    action을 무시하고 reward 0을 돌려준다. 이 step을 그대로 저장하면 가짜 transition이
    학습에 섞이기 때문에 SAME_STEP 모드를 사용한다. 이때 끝난 에피소드의 마지막 관측은
    info["final_obs"]에 들어 있다.
    """
    return gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def moving_average(values, window=5):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_rewards(path, rewards):
    """에피소드별 reward를 CSV로 저장한다. 여러 seed 결과를 나중에 한 그래프로 그릴 때 사용."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, float(r)])


def plot_rewards(rewards, path, title="Reward Trend", window=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rewards, window))
    plt.title(f"{title} ({window} Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@torch.no_grad()
def record_gif(policy, path, env_id="CartPole-v1", device="cpu", seed=None, fps=30):
    """가장 확률이 높은 행동(argmax)을 고르는 greedy 정책으로 한 에피소드를 GIF로 저장한다."""
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done:
        logits = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
        obs, _, terminated, truncated, _ = env.step(torch.argmax(logits).item())
        frames.append(env.render())
        done = terminated or truncated
    env.close()
    imageio.mimsave(path, frames, fps=fps)


def save_results(rewards, save_dir, title, policy=None, gif=False, env_id="CartPole-v1", device="cpu", seed=None):
    """학습 결과(CSV, 그래프, 선택적으로 GIF)를 save_dir에 저장한다."""
    save_rewards(os.path.join(save_dir, "rewards.csv"), rewards)
    plot_rewards(rewards, os.path.join(save_dir, "reward_curve.png"), title=title)
    if gif and policy is not None:
        record_gif(policy, os.path.join(save_dir, "agent.gif"), env_id=env_id, device=device, seed=seed)
    print(f"결과 저장: {save_dir}")
