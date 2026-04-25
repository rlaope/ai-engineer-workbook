# Volume 32 — 역전파와 자동미분

> 이 권이 끝나면 PyTorch 의 `loss.backward()` 한 줄이 내부에서 무엇을 하는지 그릴 수 있게 됩니다.

## 목적

역전파 (Backpropagation) 는 *체인룰을 효율적으로 적용해 신경망의 모든 파라미터에 대한 그래디언트를 한 번의 순회로 계산* 하는 알고리즘입니다. 1986 년 Rumelhart·Hinton·Williams 의 논문으로 정립되었으며, 모든 딥러닝 학습의 수학적 기반입니다.

## 선수 지식

- Volume 10, 30 완료

## 학습 결과

1. 역전파 알고리즘의 4 단계를 적을 수 있습니다.
2. 체인룰을 다층 신경망에 적용할 수 있습니다.
3. PyTorch autograd 의 *동적 계산 그래프* 를 이해합니다.
4. NumPy 로 작은 신경망의 역전파를 직접 구현할 수 있습니다.

---

## 1. 역전파 알고리즘

### 1.1 4 단계

1. **순전파** — 입력 → 출력 + 손실 계산
2. **역전파 시작** — 손실의 출력에 대한 그래디언트 계산
3. **체인룰 적용** — 출력 → 입력 방향으로 그래디언트 전파
4. **파라미터 갱신** — 그래디언트로 가중치 갱신

### 1.2 체인룰의 핵심

다층 함수 $L = f_n \circ f_{n-1} \circ \cdots \circ f_1$ 의 미분:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_{i+1}}{\partial f_i} \cdot \frac{\partial f_i}{\partial w_i}$$

각 항이 *지역 미분 (Local Derivative)*. 한 번의 역방향 순회로 모든 파라미터의 그래디언트가 계산됨.

---

## 2. NumPy 로 직접 구현

### 2.1 2 층 신경망

```python
import numpy as np

# 데이터
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, (100,))

# 가중치 초기화
W1 = np.random.randn(3, 5) * 0.1
b1 = np.zeros(5)
W2 = np.random.randn(5, 1) * 0.1
b2 = np.zeros(1)

lr = 0.01
for epoch in range(100):
    # 순전파
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)            # ReLU
    z2 = h @ W2 + b2
    p = 1 / (1 + np.exp(-z2.squeeze()))   # Sigmoid
    
    # 손실 (BCE)
    loss = -(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12)).mean()
    
    # 역전파
    dz2 = (p - y).reshape(-1, 1) / len(X)
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)
    
    dh = dz2 @ W2.T
    dz1 = dh * (z1 > 0)              # ReLU 미분
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    
    # 갱신
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    
    if epoch % 20 == 0:
        print(f"epoch {epoch}: loss={loss:.4f}")
```

이 코드가 *PyTorch autograd 가 내부에서 자동화하는* 작업입니다.

---

## 3. PyTorch Autograd

### 3.1 동적 계산 그래프

PyTorch 는 *순전파 도중에 계산 그래프를 구성* 하고, `backward()` 호출 시 *역방향 순회* 를 자동 수행합니다.

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)   # 8.0 = 2x + 2
```

### 3.2 자동미분의 두 종류

- *Forward-mode* — 입력 → 출력 방향. 입력 차원이 작을 때 효율적.
- *Reverse-mode (= 역전파)* — 출력 → 입력 방향. 출력 차원이 작을 때 효율적 (신경망의 경우).

신경망은 *입력이 크고 출력 (스칼라 손실) 이 작아* 역방향이 표준.

---

## 4. 그래디언트의 함정

### 4.1 그래디언트 소실

깊은 네트워크에서 *작은 미분이 곱해져 0 으로 수렴*. 시그모이드·tanh 가 특히 심함.

방어: ReLU·잔차 연결.

### 4.2 그래디언트 폭발

큰 미분이 곱해져 무한대. RNN 에서 흔함.

방어: *그래디언트 클리핑* (`clip_grad_norm_`).

### 4.3 죽은 ReLU

음수 입력 영역에서 그래디언트가 0 → 뉴런이 *영원히 비활성*.

방어: Leaky ReLU·Parametric ReLU.

---

## 권 정리

- 역전파 = 체인룰의 효율적 적용
- 4 단계: 순전파·손실·역전파·갱신
- PyTorch autograd = 동적 그래프 + reverse-mode 자동미분
- 그래디언트 함정: 소실·폭발·죽은 ReLU

가장 기억할 한 줄: **"역전파는 체인룰을 한 번의 역방향 순회로 자동화한 알고리즘이며, 모든 딥러닝 학습의 수학적 기반이다."**

다음 권: [Volume 33 — 옵티마이저](./volume_33_optimizers.md)

---

## 자가점검 키워드

`역전파`, `체인룰`, `Forward/Reverse-mode`, `Autograd`, `그래디언트 소실/폭발`

## 자가점검 질문

1. 역전파 4 단계를 적으십시오.
2. NumPy 로 2 층 신경망의 역전파를 5 줄 안에 구현하십시오.
3. PyTorch 의 `loss.backward()` 가 내부에서 무엇을 하는지 설명하십시오.
4. 그래디언트 소실·폭발·죽은 ReLU 의 방어책을 각각 적으십시오.

## 다음 권

[Volume 33 — 옵티마이저](./volume_33_optimizers.md)
