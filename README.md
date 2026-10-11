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
| REINFORCE | [`reinforce.py`](algorithms/reinforce.py) | [Policy Gradient 구현 1](https://jihoai.github.io/posts/Policy-Gradient-구현-1/) | 에피소드 단위 Monte Carlo policy gradient |
| REINFORCE (병렬 환경) | [`reinforce_vec.py`](algorithms/reinforce_vec.py) | [Policy Gradient 구현 1](https://jihoai.github.io/posts/Policy-Gradient-구현-1/#5-개선점) | 여러 환경의 trajectory를 모아 한 번에 업데이트 |
| REINFORCE + Baseline | [`reinforce_baseline.py`](algorithms/reinforce_baseline.py) | [Policy Gradient 구현 2](https://jihoai.github.io/posts/Policy-Gradient-구현-2/) | Value network를 baseline으로 사용해 variance 감소 |
| TD(0) Actor-Critic | [`actor_critic_td0.py`](algorithms/actor_critic_td0.py) | [Actor Critic](https://jihoai.github.io/posts/Actor-Critic/) | TD error를 advantage 추정값으로 사용 |

이론 정리: [Policy Gradient](https://jihoai.github.io/posts/Policy-Gradient-공부하기/) · [Temporal Difference](https://jihoai.github.io/posts/TD/) · [A3C 논문 리뷰](https://jihoai.github.io/posts/A3C-review/)

## 폴더 구조

```
RL_study/
├── algorithms/        # 알고리즘 하나 = 파일 하나. 각 파일을 바로 실행할 수 있습니다.
├── common/
│   ├── networks.py    # MLP
│   └── utils.py       # seed 고정, return 계산, 벡터 환경, 결과 저장(CSV·그래프·GIF)
├── notebooks/         # 블로그 글을 쓸 때 실행한 원본 실험 기록
└── results/           # 실행 결과가 저장되는 곳 (git에는 올리지 않음)
```

## 실행 방법

```bash
pip install -r requirements.txt

# 저장소 루트에서 모듈로 실행합니다.
python -m algorithms.reinforce --lr 0.001 --seed 0
python -m algorithms.reinforce_vec --num-envs 8 --seed 0
python -m algorithms.reinforce_baseline --seed 0
python -m algorithms.actor_critic_td0 --update-every 32 --seed 0
```

- 학습이 끝나면 `results/<알고리즘>_..._seed<seed>/`에 에피소드별 reward(`rewards.csv`)와 학습 곡선(`reward_curve.png`)이 저장됩니다.
- `--gif`를 붙이면 학습된 에이전트의 플레이 영상(`agent.gif`)도 저장합니다.
- 모든 하이퍼파라미터는 인자로 바꿀 수 있습니다. `python -m algorithms.reinforce --help`로 확인할 수 있습니다.
- 강화학습은 seed에 따라 결과 차이가 큽니다. 비교 실험을 할 때는 `--seed`를 바꿔 여러 번 실행하는 것을 권장합니다.

## 구현 상세

모든 실험은 Gymnasium의 [`CartPole-v1`](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 환경(observation 4차원, action 2개)에서 진행했고, 정책 네트워크는 `4 → 8 → 4 → 2` 구조의 MLP(ReLU)를 공통으로 사용했습니다.

### REINFORCE — `reinforce.py`

- 한 에피소드를 끝까지 진행한 뒤, reward-to-go $G_t$와 log probability로 loss를 계산해 업데이트합니다.
- Return은 $G_t = r_t + \gamma G_{t+1}$ 관계를 이용해 뒤에서부터 $O(n)$으로 계산합니다.
- 기본 설정: γ = 0.9, lr = 0.001, 5000 에피소드

### REINFORCE (병렬 환경) — `reinforce_vec.py`

- 여러 환경을 동시에 실행하고, `RolloutBuffer`에 환경별 trajectory를 저장합니다.
- 모든 trajectory를 모은 뒤 loss를 전체 transition 수로 나눠 정규화하고 한 번에 업데이트합니다.
- 8개 환경을 병렬로 사용했을 때 5000 에피소드 학습 시간이 약 20분에서 10분으로 줄었습니다.
- 기본 설정: 환경 8개, γ = 0.9, lr = 0.003, 업데이트 625번(= 5000 에피소드)

### REINFORCE + Baseline — `reinforce_baseline.py`

- Value network $V_\phi(s)$를 함께 학습하고, $\hat{A}_t = G_t - V_\phi(s_t)$를 advantage 추정값으로 사용합니다.
- Policy loss에서는 advantage를 `detach()`해 value network로 gradient가 흐르지 않게 했고, value network는 Return과의 MSE로 학습합니다.
- `--num-envs 1`로 실행하면 단일 환경 버전이 됩니다.
- 기본 설정: 환경 8개, γ = 0.99, lr = 0.08, 업데이트당 5000 transition, 50번 업데이트

### TD(0) Actor-Critic — `actor_critic_td0.py`

- TD error $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$를 advantage로 사용하고, critic은 TD target과의 MSE로 학습합니다.
- 처음에는 5000개 transition을 모아 업데이트했는데, 초기 에피소드가 짧아 업데이트가 너무 드물어 학습이 느렸습니다. 업데이트 주기를 32 transition으로 줄였을 때 학습이 개선되었습니다. (`--update-every`로 비교해 볼 수 있습니다)
- 기본 설정: 환경 1개, γ = 0.99, actor lr = 0.001, critic lr = 0.002, 50만 step

## 노트북에서 정리하며 고친 점

`notebooks/`의 원본 코드를 `algorithms/`로 옮기면서 아래 문제를 함께 고쳤습니다.

- **Baseline의 텐서 모양 문제:** value 출력이 `(N, 1)`, return이 `(N,)` 모양이라 `returns - values`가 `(N, N)`으로 broadcast되었습니다. 그 결과 상태별 $V(s_t)$가 아니라 배치 전체의 평균 value가 baseline처럼 동작했습니다. `squeeze(-1)`로 모양을 맞췄습니다.
- **벡터 환경의 자동 reset:** gymnasium 1.x 벡터 환경은 기본적으로 에피소드가 끝난 *다음* step에서 reset하면서 reward 0인 transition을 돌려줍니다. 이 transition이 학습 데이터에 섞이지 않도록 끝난 step에서 바로 reset하는 `SAME_STEP` 모드를 사용합니다.
- **시간 제한 종료(truncation) 처리:** Actor-Critic에서 500 step 제한으로 끝난 경우에도 미래 가치를 0으로 두고 있었습니다. 진짜 종료(terminated)일 때만 0으로 두고, 시간 제한일 때는 $V(s_{t+1})$로 bootstrap합니다.
- **seed 고정:** 모든 스크립트에 `--seed` 인자를 추가했습니다.

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
