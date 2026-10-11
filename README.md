# RL Study

강화학습 알고리즘을 이론부터 직접 구현하며 공부한 기록입니다.
각 구현의 이론 설명과 실험 결과는 블로그에 정리했습니다. → **[jihoai.github.io](https://jihoai.github.io)**

<table>
  <tr>
    <td align="center"><img src="https://jihoai.github.io/assets/gif/cartpole_improved.gif" width="260"><br><sub>REINFORCE (병렬 환경)</sub></td>
    <td align="center"><img src="https://jihoai.github.io/assets/gif/cartpole_agentBaseline.gif" width="260"><br><sub>REINFORCE + Baseline</sub></td>
    <td align="center"><img src="https://jihoai.github.io/assets/gif/TD0ActorCritic.gif" width="260"><br><sub>TD(0) Actor-Critic</sub></td>
  </tr>
</table>

## 구현 목록

| 알고리즘 | 코드 | 블로그 | 핵심 |
|---|---|---|---|
| REINFORCE | [`PG.py`](PG.py), [`PG.ipynb`](PG.ipynb) | [Policy Gradient 구현 1](https://jihoai.github.io/posts/Policy-Gradient-구현-1/) | 에피소드 단위 Monte Carlo policy gradient |
| REINFORCE (병렬 환경) | [`PG_rollout.py`](PG_rollout.py) | [Policy Gradient 구현 1](https://jihoai.github.io/posts/Policy-Gradient-구현-1/#5-개선점) | 여러 환경의 trajectory를 모아 한 번에 업데이트 |
| REINFORCE + Baseline | [`PolicyGradientwithBaseline.ipynb`](PolicyGradientwithBaseline.ipynb) | [Policy Gradient 구현 2](https://jihoai.github.io/posts/Policy-Gradient-구현-2/) | Value network를 baseline으로 사용해 variance 감소 |
| TD(0) Actor-Critic | [`TD0ActorCritic.ipynb`](TD0ActorCritic.ipynb) | [Actor Critic](https://jihoai.github.io/posts/Actor-Critic/) | TD error를 advantage 추정값으로 사용 |

이론 정리: [Policy Gradient](https://jihoai.github.io/posts/Policy-Gradient-공부하기/) · [Temporal Difference](https://jihoai.github.io/posts/TD/) · [A3C 논문 리뷰](https://jihoai.github.io/posts/A3C-review/)

## 구현 상세

모든 실험은 Gymnasium의 [`CartPole-v1`](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 환경(observation 4차원, action 2개)에서 진행했고, 정책 네트워크는 `4 → 8 → 4 → 2` 구조의 MLP(ReLU)를 공통으로 사용했습니다.

### REINFORCE — `PG.py`, `PG.ipynb`

- 한 에피소드를 끝까지 진행한 뒤, reward-to-go $G_t$와 log probability로 loss를 계산해 업데이트합니다.
- Return은 $G_t = r_t + \gamma G_{t+1}$ 관계를 이용해 뒤에서부터 $O(n)$으로 계산합니다.
- 노트북에서는 학습률 0.001과 0.005를 비교했습니다. (γ = 0.9, 5000 에피소드)

### REINFORCE (병렬 환경) — `PG_rollout.py`

- Gymnasium vector env로 여러 환경을 동시에 실행하고, `RolloutBuffer`에 환경별 trajectory를 저장합니다.
- 모든 trajectory를 모은 뒤 loss를 전체 transition 수로 나눠 정규화하고 한 번에 업데이트합니다.
- 8개 환경을 병렬로 사용했을 때 5000 에피소드 학습 시간이 약 20분에서 10분으로 줄었습니다.

### REINFORCE + Baseline — `PolicyGradientwithBaseline.ipynb`

- Value network $V_\phi(s)$를 함께 학습하고, $\hat{A}_t = G_t - V_\phi(s_t)$를 advantage 추정값으로 사용합니다.
- Policy loss에서는 advantage를 `detach()`해 value network로 gradient가 흐르지 않게 했고, value network는 Return과의 MSE로 학습합니다.
- 단일 환경 버전(`PGBaseline`)과 병렬 환경 버전(`PGBaselineVecEnv`)이 있습니다.

### TD(0) Actor-Critic — `TD0ActorCritic.ipynb`

- TD error $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$를 advantage로 사용하고, critic은 TD target과의 MSE로 학습합니다.
- 처음에는 5000개 transition을 모아 업데이트했는데(`TDAgent`), 초기 에피소드가 짧아 업데이트가 너무 드물어 학습이 느렸습니다.
- 업데이트 주기를 32 transition으로 줄인 온라인 버전(`TD`)에서 학습이 개선되었습니다. (γ = 0.99, actor lr 0.001, critic lr 0.002)

## 실행 환경

```bash
pip install torch gymnasium numpy matplotlib imageio
```

- `.ipynb` 파일은 Jupyter 또는 Google Colab에서 위에서부터 순서대로 실행하면 학습과 그래프 출력, GIF 저장까지 진행됩니다.
- `.py` 파일에는 네트워크와 에이전트 클래스가 정의되어 있어 노트북 등에서 import해 사용합니다.

## 로드맵

- [x] Policy Gradient (REINFORCE, Baseline)
- [x] TD(0) Actor-Critic
- [ ] Off-policy: DQN, Double/Dueling DQN, DDPG, TD3, SAC
- [ ] On-policy: GAE, TRPO, PPO
- [ ] 프로젝트: Domain Randomization을 이용한 강건한 보행 정책 학습 (MuJoCo)
- [ ] Robotics 기초와 로봇팔 조작 프로젝트

## 참고 자료

- OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- David Silver, [Lectures on Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
