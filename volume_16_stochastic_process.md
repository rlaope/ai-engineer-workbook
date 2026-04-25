# Volume 16 — 확률 과정 — Markov·SDE

> 이 권이 끝나면 *디퓨전 모델은 확률 과정이다* 라는 한 문장의 의미를 손에 잡히게 이해하게 됩니다.

## 목적

확률 과정은 *시간에 따라 변하는 확률 변수* 입니다. Markov 체인·Brownian 운동·SDE (확률 미분 방정식) 같은 도구는 디퓨전 모델 (Vol 80) 의 수학적 기반이며, 강화학습·사전학습의 일부 분석에도 등장합니다. 이 권은 그 도구들을 *디퓨전 모델 이해를 위한 최소* 수준에서 다집니다.

## 선수 지식

- Volume 11, 12 완료
- 외부 지식: 평균·분산의 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Markov 체인의 정의와 전이 행렬을 다룰 수 있습니다.
2. Brownian 운동·확률 미분 방정식의 직관을 가질 수 있습니다.
3. 디퓨전 모델의 *순방향·역방향* 과정이 모두 SDE 임을 이해합니다.
4. 점수 기반 모델 (Score-Based) 의 학습 신호가 *역방향 SDE 의 드리프트* 임을 알 수 있습니다.

---

## 1. Markov 체인

### 1.1 마르코프 성질

확률 과정 $\{X_t\}$ 가 *마르코프 성질* 을 만족하면:

$$P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$$

*미래는 현재만 주어지면 과거와 독립*. 단순한 가정이지만 매우 광범위하게 적용됩니다.

### 1.2 전이 행렬

이산 상태 마르코프 체인은 *전이 행렬* P 로 표현:

$$P_{ij} = P(X_{t+1} = j \mid X_t = i)$$

각 행의 합은 1.

```python
import numpy as np

P = np.array([
    [0.9, 0.1, 0.0],
    [0.2, 0.6, 0.2],
    [0.0, 0.3, 0.7],
])
# 초기 상태 분포
pi = np.array([1.0, 0.0, 0.0])

# 한 스텝 후
print(pi @ P)
# 100 스텝 후
print(pi @ np.linalg.matrix_power(P, 100))
# 정상 분포에 수렴
```

### 1.3 정상 분포

전이를 무한히 반복하면 *정상 분포* 에 수렴합니다 (적절한 조건 하). PageRank·MCMC 의 수학적 기반.

### 1.4 챕터 정리

Markov 체인은 *현재 상태가 미래를 결정* 하는 단순한 구조이며, 전이 행렬·정상 분포가 핵심 도구입니다.

---

## 2. Brownian 운동

### 2.1 정의

Brownian 운동 $W_t$ 는 다음을 만족합니다.

- $W_0 = 0$
- 증분 $W_t - W_s$ 는 *정규 분포* $N(0, t-s)$
- 비중첩 증분은 *독립*

물리학에서 *랜덤 입자 운동* 의 모델로 시작되었지만, 금융·ML 등 광범위하게 적용됩니다.

### 2.2 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

T = 1.0
n = 1000
dt = T / n
steps = np.random.randn(n) * np.sqrt(dt)
W = np.concatenate([[0], np.cumsum(steps)])

plt.plot(np.linspace(0, T, n+1), W)
plt.xlabel('t'); plt.ylabel('W_t')
plt.title('Brownian motion sample')
plt.show()
```

### 2.3 챕터 정리

Brownian 운동은 *정규 분포 증분의 누적* 이며, 모든 SDE 의 기반입니다.

---

## 3. 확률 미분 방정식 (SDE)

### 3.1 정의

SDE 의 일반형:

$$dX_t = f(X_t, t) \, dt + g(X_t, t) \, dW_t$$

- $f$: 드리프트 (deterministic 부분)
- $g$: 확산 (random 부분)
- $dW_t$: Brownian 증분

### 3.2 Euler-Maruyama 시뮬레이션

가장 단순한 수치 적분:

```python
import numpy as np

def simulate_sde(f, g, x0, T, n):
    dt = T / n
    x = x0
    xs = [x]
    for _ in range(n):
        dW = np.random.randn() * np.sqrt(dt)
        x = x + f(x) * dt + g(x) * dW
        xs.append(x)
    return np.array(xs)

# Ornstein-Uhlenbeck (평균 회귀)
xs = simulate_sde(lambda x: -x, lambda x: 1.0, x0=2.0, T=5.0, n=1000)
```

### 3.3 챕터 정리

SDE 는 *드리프트 + 확산* 의 형태이며, Euler-Maruyama 로 수치 시뮬레이션 가능합니다.

---

## 4. 디퓨전 모델의 SDE 시각

### 4.1 순방향 과정

디퓨전 모델의 순방향 (학습 시 데이터에 노이즈 추가) 과정은 SDE 로 표현됩니다.

$$dX_t = f(X_t, t) \, dt + g(t) \, dW_t$$

데이터 $X_0$ 에서 출발해 $T$ 시점에 *완전한 가우시안 노이즈* 가 됩니다.

### 4.2 역방향 과정

Anderson (1982) 의 결과: *순방향 SDE 가 주어지면 역방향 SDE 가 명시적으로 표현* 됩니다.

$$dX_t = [f(X_t, t) - g(t)^2 \nabla_x \log p_t(X_t)] \, dt + g(t) \, d\bar{W}_t$$

핵심 항: $\nabla_x \log p_t(X_t)$ — *score 함수*. 이것이 *디퓨전 모델이 학습하는 대상* 입니다.

### 4.3 점수 기반 모델

점수 함수를 신경망으로 추정 → 역방향 SDE 시뮬레이션 → 노이즈에서 데이터 생성. 이것이 Stable Diffusion 같은 모든 현대 디퓨전 모델의 수학적 골격입니다.

### 4.4 챕터 정리

디퓨전 모델은 *순방향 SDE + 역방향 SDE + score 추정* 의 조합이며, 이 권의 모든 도구가 디퓨전 모델 (Vol 80) 의 직접적 기반입니다.

---

## 권 정리

- Markov 체인 — 현재가 미래 결정, 전이 행렬·정상 분포
- Brownian 운동 — 정규 증분 누적, 모든 SDE 의 기반
- SDE — 드리프트 + 확산, Euler-Maruyama 시뮬레이션
- 디퓨전 모델 — 순방향·역방향 SDE 의 학습

가장 기억에 남겨야 할 한 줄은 **"디퓨전 모델은 score 함수를 학습한 SDE 시뮬레이션이며, 모든 수학이 이 한 문장에 압축된다."** 입니다.

다음 권은 [Volume 17 — 통계적 학습 이론](./volume_17_learning_theory.md) 입니다.

---

## 자가점검 키워드

`Markov`, `전이 행렬`, `정상 분포`, `Brownian`, `SDE`, `Euler-Maruyama`, `score 함수`, `역방향 SDE`

## 자가점검 질문

1. 마르코프 성질의 의미를 한 문단으로 설명하십시오.
2. Brownian 운동의 3 가지 정의 조건을 적으십시오.
3. SDE 의 일반형 $dX = f \, dt + g \, dW$ 의 두 항의 의미를 설명하십시오.
4. 디퓨전 모델의 *순방향·역방향* 과정을 SDE 로 적으십시오.

## 다음 권

[Volume 17 — 통계적 학습 이론](./volume_17_learning_theory.md)
