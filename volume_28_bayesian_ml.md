# Volume 28 — 베이지안 머신러닝

> 이 권이 끝나면 *불확실성을 모델링한다* 는 한 문장이 코드 수준에서 어떻게 가능한지 보일 수 있게 됩니다.

## 목적

표준 ML 모델은 *예측 한 값* 만 출력합니다. 베이지안 ML 은 *예측의 불확실성 분포* 를 출력합니다. 의료·금융·자율주행 같은 *결정의 신뢰도가 중요한* 영역에서 베이지안 사고가 결정적입니다. 이 권은 베이지안 회귀·MCMC·변분 추론의 직관을 다집니다.

## 선수 지식

- Volume 11, 12, 17 완료

## 학습 결과

1. 베이지안 회귀의 *예측 분포* 가 무엇인지 이해합니다.
2. MCMC 의 발상을 그릴 수 있습니다.
3. 변분 추론이 *왜 MCMC 보다 빠른가* 알 수 있습니다.
4. 베이지안 신경망의 적용 시점을 식별할 수 있습니다.

---

## 1. 베이지안 회귀

### 1.1 표준 회귀와의 차이

표준: $\hat{y} = X\hat{\beta}$ (점 예측)
베이지안: $p(y \mid X)$ (예측 분포)

가중치 $\beta$ 자체를 *분포* 로 다룸:

$$p(\beta \mid D) \propto p(D \mid \beta) p(\beta)$$

### 1.2 사용 예

```python
import pymc as pm
import numpy as np

with pm.Model() as model:
    beta = pm.Normal('beta', 0, 1, shape=3)
    sigma = pm.HalfNormal('sigma', 1)
    mu = X @ beta
    y_obs = pm.Normal('y_obs', mu, sigma, observed=y)
    trace = pm.sample(1000)

# 예측 분포에서 신뢰구간
print(pm.summary(trace, var_names=['beta']))
```

장점: *불확실성 정량화*. 단점: *계산 비용 큼*.

---

## 2. MCMC (Markov Chain Monte Carlo)

### 2.1 발상

사후 분포에서 직접 샘플링이 어렵다면, *그 분포에 수렴하는 마르코프 체인을 시뮬레이션*.

알고리즘: Metropolis-Hastings, Gibbs Sampling, Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS).

### 2.2 강약

- 강점: *임의 분포 처리*, *이론적으로 정확*
- 약점: *느림*, *수렴 진단 어려움*

대규모 신경망에는 *계산 비용 때문에* 적용 어려움.

---

## 3. 변분 추론 (Variational Inference)

### 3.1 발상

진짜 사후 분포 $p(\beta \mid D)$ 를 *간단한 분포 $q(\beta)$* 로 근사. KL 발산 최소화로 학습.

$$q^* = \arg\min_q D_{KL}(q(\beta) \| p(\beta \mid D))$$

### 3.2 장점

- *MCMC 보다 훨씬 빠름*
- *대규모 모델에 적용 가능*
- *최적화 문제* 로 환원

### 3.3 단점

- *진짜 사후 분포의 한 근사* — 정확도 손실
- *Mode-Seeking* 행동 (한 모드만 잡음)

---

## 4. 베이지안 신경망

신경망 가중치를 *분포* 로 다룸. 추론 시 *여러 가중치 샘플로 예측 → 평균* — 불확실성 정량화.

```python
# pyro 또는 BayesianTorch 사용
import torch
import pyro
import pyro.nn as pnn

class BNN(pnn.PyroModule):
    def __init__(self):
        super().__init__()
        self.fc = pnn.PyroModule[torch.nn.Linear](10, 1)
        self.fc.weight = pnn.PyroSample(
            dist.Normal(0., 1.).expand([1, 10]).to_event(2)
        )
```

응용: *의료 진단 (불확실하면 인간 의사에게)*, *자율주행 (안전 마진 계산)*.

---

## 5. 실용적 대안 — Deep Ensemble

여러 신경망을 *다른 시드로 학습* 한 뒤 *예측 분산을 불확실성으로 사용*. 베이지안보다 단순하면서 비슷한 결과.

```python
predictions = [model.predict(X) for model in ensemble]
mean_pred = np.mean(predictions, axis=0)
uncertainty = np.std(predictions, axis=0)
```

---

## 권 정리

- 베이지안 ML = 불확실성 분포 출력
- MCMC = 느리고 정확
- 변분 추론 = 빠르고 근사적
- 베이지안 신경망 = 의료·자율주행 같은 고위험 도메인
- Deep Ensemble = 실용적 대안

가장 기억할 한 줄: **"불확실성이 결정에 영향을 미치는 도메인에서는 점 예측보다 분포 예측이 가치가 있다."**

다음 권: [Volume 29 — 하이퍼파라미터 탐색](./volume_29_hpo.md)

---

## 자가점검 키워드

`사후 분포`, `MCMC`, `HMC/NUTS`, `변분 추론`, `베이지안 신경망`, `Deep Ensemble`

## 자가점검 질문

1. 표준 ML 과 베이지안 ML 의 출력 차이를 적으십시오.
2. MCMC 와 변분 추론의 트레이드오프를 비교하십시오.
3. Deep Ensemble 이 *실용적 베이지안 대안* 인 이유를 설명하십시오.

## 다음 권

[Volume 29 — 하이퍼파라미터 탐색](./volume_29_hpo.md)
