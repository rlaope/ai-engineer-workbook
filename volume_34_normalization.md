# Volume 34 — 정규화 (Normalization)

> 이 권이 끝나면 BatchNorm·LayerNorm·RMSNorm 중 어떤 것을 언제 써야 하는지 한 문장으로 답할 수 있게 됩니다.

## 목적

신경망의 *층별 활성화 분포* 가 학습 도중 변하면 *내부 공변 시프트 (Internal Covariate Shift)* 가 발생해 학습이 불안정합니다. 정규화는 *각 층의 입력을 평균 0·분산 1로 표준화* 해 이 문제를 완화하는 도구입니다. BN·LN·GN·RMSNorm 의 차이를 다집니다.

## 선수 지식

- Volume 30, 32 완료

## 학습 결과

1. BatchNorm·LayerNorm·GroupNorm·RMSNorm 의 차이를 알 수 있습니다.
2. *어떤 차원으로 통계를 계산하는가* 가 핵심임을 이해합니다.
3. CNN 은 BN, 트랜스포머는 LN/RMSNorm 인 이유를 설명합니다.
4. 학습/추론 시 BN 동작 차이를 알 수 있습니다.

---

## 1. BatchNorm

### 1.1 동작

배치 차원으로 평균·분산 계산:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

여기서 $\mu_B$, $\sigma_B$ 는 *현재 미니배치의 통계*. $\gamma, \beta$ 는 학습 가능 스케일/시프트.

### 1.2 강약

장점: *학습 안정화*, *큰 학습률 가능*, *암묵적 정칙화*.
단점:
- *작은 배치에서 통계 부정확*
- *학습/추론 차이* (추론 시 *이동 평균* 사용)
- *RNN/시퀀스 모델에 부적합*

### 1.3 Vol 32 와의 연결

CNN 학습에서 BN 이 *없으면 학습이 거의 불가능* 했던 깊은 네트워크가 BN 으로 가능해졌습니다 (ResNet 시대).

---

## 2. LayerNorm

### 2.1 동작

배치 대신 *특성 차원* 으로 통계 계산. 각 샘플의 모든 특성을 정규화.

### 2.2 트랜스포머 표준

트랜스포머는 *시퀀스 길이가 가변* 이고 *배치가 작은 경우가 많아* BN 부적합. LN 이 표준.

```python
import torch.nn as nn
ln = nn.LayerNorm(d_model)
```

---

## 3. GroupNorm

특성을 *그룹으로 나눠* 그룹별 정규화. *작은 배치* 에서 BN 의 대안.

비전 모델 (특히 detection·segmentation 의 작은 배치 환경) 에서 사용.

---

## 4. RMSNorm

LayerNorm 의 *평균 빼기 단계 생략*. 분산만으로 정규화:

$$y = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma$$

장점: *계산 간소화*, *비슷한 성능*. LLaMA·Mistral 등 *현대 LLM 표준*.

---

## 5. 비교와 선택

```
+-----------+--------+--------+--------+---------+
|           | 통계 차원 | 배치 의존 | 적용 영역 | 학습/추론 |
+-----------+--------+--------+--------+---------+
| BatchNorm | 배치    | Yes    | CNN    | 다름     |
| LayerNorm | 특성    | No     | Transformer | 같음 |
| GroupNorm | 특성 그룹 | No  | CNN (작은 배치) | 같음 |
| RMSNorm   | 특성 (분산만) | No | LLM (LLaMA+) | 같음 |
+-----------+--------+--------+--------+---------+
```

선택 가이드:
- *CNN 큰 배치* → BN
- *CNN 작은 배치* → GN
- *Transformer* → LN
- *현대 LLM* → RMSNorm

---

## 권 정리

- 정규화 = 층별 활성화 분포 안정화
- BN = CNN 표준, LN = Transformer 표준, RMSNorm = 최신 LLM 표준
- *어떤 차원으로 통계 계산하는가* 가 핵심 차이

가장 기억할 한 줄: **"CNN 은 BatchNorm, Transformer 는 LayerNorm, 최신 LLM 은 RMSNorm — 데이터 모양이 정규화 선택을 결정한다."**

다음 권: [Volume 35 — 정칙화 (Regularization)](./volume_35_regularization.md)

---

## 자가점검 키워드

`BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `내부 공변 시프트`, `이동 평균`

## 자가점검 질문

1. 4 가지 정규화의 *통계 계산 차원* 을 적으십시오.
2. BN 의 학습/추론 차이를 설명하십시오.
3. 트랜스포머가 BN 대신 LN 을 쓰는 이유를 적으십시오.
4. RMSNorm 이 LayerNorm 보다 단순화한 부분을 적으십시오.

## 다음 권

[Volume 35 — 정칙화 (Regularization)](./volume_35_regularization.md)
