# Volume 39 — 활성화 함수 깊이

> 이 권이 끝나면 ReLU·GELU·SiLU·Swish 중 *왜 현대 트랜스포머는 GELU/SiLU 를 쓰는가* 에 답할 수 있게 됩니다.

## 목적

활성화 함수 선택은 모델 성능과 학습 안정성에 영향을 주며, 시대마다 표준이 바뀌었습니다. 시그모이드 → ReLU → GELU/SiLU/SwiGLU 의 흐름에는 각각 *해결한 문제와 새로 만든 문제* 가 있습니다.

## 선수 지식

- Volume 30 완료

## 학습 결과

1. 시그모이드/tanh 가 *왜 깊은 망에서 사라졌는가* 를 설명할 수 있습니다.
2. ReLU 의 죽은 뉴런 문제와 그 변형 (LeakyReLU·PReLU·ELU) 의 동기를 알 수 있습니다.
3. GELU·SiLU·Swish·Mish 의 형태와 미분을 그릴 수 있습니다.
4. SwiGLU 가 트랜스포머 FFN 에서 채택된 이유를 설명할 수 있습니다.

---

## 1. 시그모이드와 tanh 의 한계

### 1.1 시그모이드

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

미분: $\sigma(1-\sigma)$. 최대값 0.25.

### 1.2 그래디언트 소실

깊은 망에서 시그모이드 미분이 곱해지면 (각 0.25 이하), $0.25^{10} \approx 10^{-6}$ 으로 *그래디언트 사라짐*.

### 1.3 챕터 정리

시그모이드·tanh 는 *깊은 네트워크에서 그래디언트 소실* 로 표준 위치를 잃었습니다.

---

## 2. ReLU 와 변형

### 2.1 ReLU

$$\text{ReLU}(z) = \max(0, z)$$

장점: *계산 빠름*, *그래디언트 1 (양수 영역)*, *희소 활성화*.
단점: *죽은 뉴런* — 음수 영역에서 그래디언트 0.

### 2.2 LeakyReLU·PReLU·ELU

음수 영역에 *작은 기울기* 부여:

- **LeakyReLU** $\max(\alpha z, z)$, $\alpha = 0.01$ 고정
- **PReLU** — $\alpha$ 학습 가능
- **ELU** $z$ if $z>0$, $\alpha(e^z - 1)$ if $z\leq 0$

---

## 3. GELU·SiLU·Swish·Mish

### 3.1 GELU

$$\text{GELU}(z) = z \cdot \Phi(z)$$

$\Phi$ 는 표준 정규의 CDF. *부드러운 ReLU*. BERT·GPT 의 표준.

### 3.2 SiLU/Swish

$$\text{SiLU}(z) = z \cdot \sigma(z)$$

GELU 와 거의 같은 모양. 더 단순한 계산. LLaMA·Mistral 등 표준.

### 3.3 Mish

$$\text{Mish}(z) = z \cdot \tanh(\ln(1+e^z))$$

비전 분야 일부.

### 3.4 챕터 정리

GELU·SiLU 는 *부드러운 ReLU* 형태이며, 미세한 정확도 개선 + 학습 안정성을 가집니다.

---

## 4. SwiGLU

### 4.1 정의

GLU (Gated Linear Unit) 의 SiLU 버전:

$$\text{SwiGLU}(x, W, V) = (Wx \cdot \text{SiLU}(Vx))$$

게이트 (SiLU) 와 값 (Linear) 을 곱하는 *2 경로 구조*.

### 4.2 트랜스포머 FFN 에 채택

표준 FFN 은 *Linear → ReLU → Linear*. SwiGLU 변형은 *2 경로* — 약간 더 많은 파라미터지만 정확도 개선이 검증됨. LLaMA 시리즈가 채택.

```python
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w3(self.w1(x) * torch.silu(self.w2(x)))
```

---

## 권 정리

- 시그모이드/tanh = 깊은 망에서 그래디언트 소실로 사라짐
- ReLU = 표준이지만 죽은 뉴런 문제
- LeakyReLU/PReLU/ELU = 음수 영역 보완
- GELU/SiLU = 부드러운 ReLU, 트랜스포머 표준
- SwiGLU = 게이트 구조, 최신 LLM 표준

가장 기억할 한 줄: **"활성화 함수의 진화는 그래디언트 소실에서 죽은 뉴런까지의 문제 해결사이며, 현대 LLM 은 SwiGLU 가 표준이다."**

다음 권: [Volume 40 — 학습률 스케줄 깊이](./volume_40_lr_schedules.md)

---

## 자가점검 키워드

`Sigmoid`, `tanh`, `ReLU`, `Leaky/P/ELU`, `GELU`, `SiLU/Swish`, `SwiGLU`, `죽은 뉴런`

## 자가점검 질문

1. 시그모이드의 그래디언트 소실 원인을 식으로 설명하십시오.
2. ReLU 의 죽은 뉴런과 해결책을 적으십시오.
3. GELU 와 SiLU 의 차이를 적으십시오.
4. SwiGLU 가 표준 ReLU FFN 보다 가지는 이점을 설명하십시오.

## 다음 권

[Volume 40 — 학습률 스케줄 깊이](./volume_40_lr_schedules.md)
