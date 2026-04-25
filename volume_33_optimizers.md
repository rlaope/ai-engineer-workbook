# Volume 33 — 옵티마이저

> 이 권이 끝나면 SGD·Adam·AdamW 중 *어떤 것을 언제 어떤 학습률로 써야 하는가* 에 답할 수 있게 됩니다.

## 목적

같은 모델·같은 데이터라도 옵티마이저와 학습률 스케줄에 따라 학습 결과가 크게 달라집니다. 옵티마이저는 *그래디언트를 어떻게 누적·정규화·적용하는가* 에 대한 서로 다른 전략이며, 그 차이의 직관을 이해하지 않으면 새 모델을 학습할 때마다 *마법의 숫자를 끼워 맞추는 일* 이 됩니다. 이 권은 SGD 부터 AdamW·Lion 까지 옵티마이저의 진화 과정을 따라갑니다.

## 선수 지식

- Volume 10, 32 완료

## 학습 결과

1. SGD·모멘텀·NAG 의 차이를 한 그림으로 설명할 수 있습니다.
2. Adam 이 *그래디언트의 1·2 차 모멘트* 를 사용함을 보일 수 있습니다.
3. AdamW 가 Adam 의 *Weight Decay 를 분리* 한 변형임을 설명할 수 있습니다.
4. 옵티마이저별 *기본 학습률 범위* 를 머릿속에 갖게 됩니다.
5. 학습률 스케줄(Warmup·Cosine 등) 의 효과를 알 수 있습니다.

---

## 1. SGD 와 변형

### 1.1 기본 SGD

매 스텝 *현재 그래디언트* 만으로 갱신:

$$W_{t+1} = W_t - \eta \nabla L(W_t)$$

문제: *좁은 골짜기에서 지그재그*, *느린 수렴*.

### 1.2 모멘텀

이전 이동을 *관성* 으로 누적:

$$v_{t+1} = \mu v_t + \nabla L, \quad W_{t+1} = W_t - \eta v_{t+1}$$

$\mu = 0.9$ 이 표준. 진동을 평균화하고 수렴을 가속.

### 1.3 NAG (Nesterov)

*예측된 다음 위치에서 그래디언트* 를 계산. 모멘텀의 개선.

```python
class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def step(self, grad):
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.momentum * self.v + grad
        return -self.lr * self.v
```

---

## 2. Adam 계열

### 2.1 Adam

*모멘텀 + 적응 학습률*. 그래디언트의 1·2 차 모멘트 사용:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$W_{t+1} = W_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

각 파라미터가 *자기에게 맞는 학습률* 을 가짐. 좁은 골짜기·다양한 스케일 그래디언트에 강함.

기본값: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, lr=$10^{-3}$.

### 2.2 AdamW

Adam 의 *L2 정칙화* 에 미묘한 결함이 있어 (가중치 감쇠가 학습률에 의존), AdamW 는 *Weight Decay 를 분리*:

```python
W_{t+1} = W_t - \eta (m_hat / (sqrt(v_hat) + eps) + wd * W_t)
```

Loshchilov & Hutter (2017). 트랜스포머 학습의 *현재 표준*.

### 2.3 Lion

*Adam 보다 단순하면서 더 빠른* 후보 (Google, 2023). 부호만 사용하는 단순한 갱신:

$$c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$W_{t+1} = W_t - \eta \cdot \text{sign}(c_t)$$

`[VERIFY: Chen et al. 2023, Symbolic Discovery of Optimization Algorithms]`

---

## 3. 학습률 선택

### 3.1 옵티마이저별 표준 학습률

```
+----------+------------------------+
| 옵티마이저 | 표준 학습률           |
+----------+------------------------+
| SGD      | 1e-2 ~ 1e-1 (모멘텀)   |
| Adam     | 1e-3 ~ 1e-4            |
| AdamW    | 1e-4 ~ 5e-4 (트랜스포머)|
| Lion     | Adam 의 1/3-1/10       |
+----------+------------------------+
```

### 3.2 LR Range Test

학습률을 *지수적으로 증가* 시키며 손실을 관찰. *손실이 가장 빠르게 떨어지는 구간* 의 학습률이 좋은 시작점.

---

## 4. 학습률 스케줄

### 4.1 표준 스케줄

- **Linear Warmup + Cosine Decay** — 트랜스포머 학습의 표준
- **OneCycle** — 빠른 수렴, fast.ai 인기
- **ReduceOnPlateau** — 검증 손실 정체 시 감소
- **Step Decay** — 단순, 고전적

### 4.2 Warmup 의 동기

큰 배치 학습 초기에 *그래디언트 분산이 크면* 모델이 망가짐. 학습률을 *0 → 목표값* 선형 증가로 보호.

```python
def warmup_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr
```

---

## 권 정리

- SGD → 모멘텀 → NAG → Adam → AdamW → Lion 진화
- AdamW 가 트랜스포머 표준
- 옵티마이저별 표준 학습률 범위
- Warmup + Cosine 이 큰 모델 학습의 표준 스케줄

가장 기억할 한 줄: **"새 모델 학습은 AdamW + Linear Warmup + Cosine Decay 부터 시작하면 거의 잘 동작한다."**

다음 권: [Volume 34 — 정규화 (Normalization)](./volume_34_normalization.md)

---

## 자가점검 키워드

`SGD`, `모멘텀`, `Adam`, `AdamW`, `Lion`, `LR Range Test`, `Warmup`, `Cosine`

## 자가점검 질문

1. SGD·모멘텀·Adam 의 차이를 식으로 비교하십시오.
2. AdamW 가 Adam 의 무엇을 분리했는지 설명하십시오.
3. 옵티마이저별 표준 학습률을 표로 정리하십시오.
4. Warmup 이 큰 모델 학습에서 필요한 이유를 설명하십시오.

## 다음 권

[Volume 34 — 정규화 (Normalization)](./volume_34_normalization.md)
