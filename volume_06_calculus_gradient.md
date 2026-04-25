# Volume 6 — 미적분과 그래디언트

> 이 권이 끝나면 *학습이란 곧 손실의 그래디언트를 따라 내려가는 일* 이라는 한 문장을 완전히 이해하게 됩니다.

## 목적

딥러닝의 학습 알고리즘은 *손실 함수의 기울기를 계산해 파라미터를 그 반대 방향으로 조금씩 옮기는 일* 의 반복입니다. 이 사실을 이해하지 못하면 옵티마이저·학습률·역전파·그래디언트 클리핑 같은 모든 학습 관련 개념이 *마법의 주문* 이 됩니다. 이 권은 미적분의 가장 핵심적인 부분만 골라, 그래디언트라는 개념을 직관적으로 손에 잡히게 만듭니다.

## 선수 지식

- Volume 4 완료 (벡터·행렬)
- 외부 지식: 함수와 그래프, 변화율의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 미분이 *순간 변화율* 임을 그림으로 설명할 수 있습니다.
2. 편미분과 그래디언트의 차이를 구분할 수 있습니다.
3. 체인룰을 다층 함수에 적용할 수 있습니다.
4. 야코비안과 헤시안의 의미를 한 줄로 설명할 수 있습니다.
5. 경사하강법의 한 스텝을 손으로 계산하고 코드로 옮길 수 있습니다.

---

## 이 권을 읽기 전에

미적분은 학교에서 *복잡한 적분 공식을 외우는 과목* 으로 기억되는 경우가 많습니다. 그러나 ML 에서 필요한 미적분의 90% 는 *함수의 기울기* 라는 단 하나의 개념입니다.

이 권은 적분을 거의 다루지 않습니다. *미분 (그래디언트)* 만 깊이 다룹니다. 신경망 학습은 본질적으로 *손실 함수의 기울기를 따라 파라미터를 조금씩 이동* 하는 일이며, 이 한 가지를 정확히 이해하면 모든 학습 알고리즘이 변형으로 보입니다.

NumPy 로 모든 개념을 직접 계산하고 그림으로 그려 봅니다. 수식을 외우지 마시고 *왜 이렇게 되는가* 를 코드로 확인하시기 바랍니다.

---

## 1. 미분의 정의와 직관

### 1.1 변화율로서의 미분

함수 $f(x)$ 의 *x 에 대한 미분* $f'(x)$ 또는 $\frac{df}{dx}$ 는 *x 가 아주 조금 변할 때 f 가 얼마나 변하는가* 의 비율입니다.

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

기하학적으로는 *그래프 위의 한 점에서의 접선의 기울기* 입니다.

### 1.2 NumPy 로 수치 미분

해석적 미분 공식을 모르더라도 *수치적으로* 미분을 계산할 수 있습니다.

```python
import numpy as np

def f(x):
    return x**2

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# f(x) = x^2 의 미분은 2x
print(numerical_derivative(f, 3))   # 6.0 에 가까움
print(numerical_derivative(f, 5))   # 10.0 에 가까움
```

이 *중심 차분* 방식은 단순하지만 *이론적 미분* 과 매우 가깝습니다.

### 1.3 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = x**2
dy = 2 * x  # 해석적 미분

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x, y); axes[0].set_title('f(x) = x^2'); axes[0].grid(True)
axes[1].plot(x, dy); axes[1].set_title("f'(x) = 2x"); axes[1].grid(True)
plt.show()
```

미분 곡선은 *원래 함수의 기울기 변화* 를 보여 줍니다. $x = 0$ 에서 미분값이 0 이며, 이는 *원래 함수가 극값* 을 가지는 점입니다.

### 1.4 ML 에서의 의미

신경망 학습에서 *미분 = 그래디언트* 는 *손실 함수가 파라미터에 대해 얼마나 민감한가* 를 측정합니다. 그래디언트가 큰 방향은 *조금만 움직여도 손실이 크게 변하는 방향* 이며, 학습은 *그 방향의 반대로 움직여 손실을 줄이는* 일입니다.

### 1.5 챕터 정리

미분은 *함수의 순간 변화율* 이며, 그래프의 *접선 기울기* 입니다. NumPy 로 수치 미분을 손쉽게 계산할 수 있고, ML 에서는 *손실의 변화율* 로 사용됩니다. 다음 챕터에서는 *입력이 여러 개* 일 때의 미분을 봅니다.

---

## 2. 편미분

### 2.1 다변수 함수의 미분

함수가 여러 변수를 받을 때 (예: $f(x, y) = x^2 + y^2$), *각 변수에 대해 따로* 미분할 수 있습니다.

**x 에 대한 편미분** $\frac{\partial f}{\partial x}$ — y 를 *상수처럼* 고정하고 x 만 변화시켜 미분.

**y 에 대한 편미분** $\frac{\partial f}{\partial y}$ — x 를 고정하고 y 만 변화.

$f(x, y) = x^2 + y^2$ 의 경우:
- $\frac{\partial f}{\partial x} = 2x$
- $\frac{\partial f}{\partial y} = 2y$

### 2.2 NumPy 로 편미분

```python
import numpy as np

def f(x, y):
    return x**2 + y**2

def partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

print(partial_x(f, 3, 4))   # 6 에 가까움 (= 2*3)
print(partial_y(f, 3, 4))   # 8 에 가까움 (= 2*4)
```

### 2.3 신경망에서의 편미분

신경망의 손실 함수는 *수백만-수십억 개의 파라미터* 에 대한 함수입니다. 각 파라미터에 대한 *편미분* 들의 묶음이 *그래디언트* 가 됩니다.

```
파라미터 W = (w_1, w_2, ..., w_n)
손실 L(W)

각 w_i 에 대한 편미분:
  dL/dw_1, dL/dw_2, ..., dL/dw_n
```

### 2.4 챕터 정리

편미분은 *다변수 함수에서 한 변수만 변화시킬 때의 변화율* 입니다. 다른 변수들은 *상수처럼* 고정합니다. 신경망의 그래디언트는 *모든 파라미터에 대한 편미분의 묶음* 입니다. 다음 챕터에서 그 묶음을 *벡터* 로 다룹니다.

---

## 3. 그래디언트

### 3.1 그래디언트의 정의

함수 $f(x_1, x_2, \ldots, x_n)$ 의 *그래디언트* $\nabla f$ 는 *모든 편미분을 모아 둔 벡터* 입니다.

$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### 3.2 그래디언트의 기하학적 의미

그래디언트 벡터는 *함수 f 가 가장 빠르게 증가하는 방향* 을 가리킵니다. 그 방향의 *반대* 가 *가장 빠르게 감소하는 방향* 이며, 학습은 이 반대 방향으로 *조금씩 이동* 합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# f(x, y) = x^2 + y^2 의 등고선 + 그래디언트 벡터
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# 그래디언트 = (2x, 2y)
gx = 2 * X
gy = 2 * Y

fig, ax = plt.subplots(figsize=(8, 8))
ax.contour(X, Y, Z, levels=10)
# 일부 점에서만 화살표 표시
step = 5
ax.quiver(X[::step, ::step], Y[::step, ::step],
          gx[::step, ::step], gy[::step, ::step],
          color='red')
ax.set_title('등고선과 그래디언트 (가장 빠르게 증가하는 방향)')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal'); ax.grid(True)
plt.show()
```

화살표는 항상 *높은 곳을 향하며*, 등고선과 *수직* 입니다.

### 3.3 NumPy 로 그래디언트

```python
import numpy as np

def f(point):
    x, y = point
    return x**2 + y**2

def gradient(f, point, h=1e-5):
    grad = np.zeros_like(point, dtype=float)
    for i in range(len(point)):
        delta = np.zeros_like(point, dtype=float)
        delta[i] = h
        grad[i] = (f(point + delta) - f(point - delta)) / (2 * h)
    return grad

p = np.array([3, 4], dtype=float)
print(gradient(f, p))   # [6. 8.]  (= 2x, 2y)
```

### 3.4 손실 함수의 그래디언트

신경망 학습에서 *손실 L 의 가중치 W 에 대한 그래디언트* $\nabla_W L$ 가 학습의 핵심 신호입니다.

```python
# 단순화된 예시
def loss(w):
    return (w - 5)**2  # 정답 w=5 에서 가까울수록 좋음

def loss_gradient(w):
    return 2 * (w - 5)

# 학습 시뮬레이션
w = 0.0
lr = 0.1
for step in range(50):
    g = loss_gradient(w)
    w = w - lr * g       # 그래디언트 반대 방향으로 이동
    if step % 5 == 0:
        print(f"step {step}: w = {w:.4f}, loss = {loss(w):.4f}")
```

w 가 점차 5 로 수렴합니다. 이것이 *경사하강법* 의 핵심 (6장).

### 3.5 챕터 정리

그래디언트는 *모든 편미분을 모아 둔 벡터* 이며, *함수가 가장 빠르게 증가하는 방향* 을 가리킵니다. 학습은 *그 반대 방향으로 이동* 하는 일입니다. 다음 챕터에서는 *합성 함수의 미분 규칙* 인 체인룰을 봅니다.

---

## 4. 체인룰

### 4.1 합성 함수

함수가 *함수의 함수* 일 때 (예: $h(x) = f(g(x))$), 미분은 *체인룰* 로 계산합니다.

$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

다층 합성:

$$\frac{d}{dx} f(g(h(x))) = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

### 4.2 신경망 = 깊은 합성

신경망의 한 층은 함수입니다. 깊은 신경망은 *함수의 함수의 함수의 ...* 의 합성입니다.

```
y = f_L( f_{L-1}( ... f_1(x) ... ))
```

학습을 위해 *입력에 가까운 층의 가중치에 대한 그래디언트* 를 계산하려면 체인룰을 *층의 깊이만큼 반복* 적용해야 합니다. 이것이 *역전파 (Backpropagation)* 입니다.

### 4.3 손으로 체인룰

$f(x) = (x^2 + 1)^3$ 의 미분:

- $g(x) = x^2 + 1$, $h(g) = g^3$
- $\frac{dg}{dx} = 2x$, $\frac{dh}{dg} = 3g^2$
- $\frac{df}{dx} = 3g^2 \cdot 2x = 6x(x^2+1)^2$

NumPy 로 검증:

```python
def f(x):
    return (x**2 + 1)**3

def f_grad_analytic(x):
    return 6 * x * (x**2 + 1)**2

def f_grad_numerical(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

x = 2.0
print("해석적:", f_grad_analytic(x))           # 6*2*5^2 = 300
print("수치적:", f_grad_numerical(f, x))        # ~300
```

### 4.4 신경망에서의 체인룰

```
입력 x → [선형층] → z₁ → [활성화] → a₁ → [선형층] → z₂ → 손실 L

dL/dW_1 = dL/da₁ * da₁/dz₁ * dz₁/dW_1   (체인룰)
```

이 식의 각 항은 *해당 층의 지역 미분* 이며, 자동미분 시스템 (PyTorch autograd) 이 자동으로 계산합니다. 사용자가 직접 손으로 계산할 일은 거의 없지만, *내부에서 무엇이 일어나는지* 의 그림은 가지고 있어야 합니다.

### 4.5 챕터 정리

체인룰은 *합성 함수의 미분 규칙* 이며, *각 층의 지역 미분을 곱해* 전체 미분을 구합니다. 신경망의 역전파는 체인룰을 *층의 깊이만큼* 반복 적용한 것입니다. 다음 챕터에서는 그래디언트의 *고차원 일반화* 인 야코비안과 헤시안을 봅니다.

---

## 5. 야코비안과 헤시안

### 5.1 야코비안 (Jacobian)

함수가 *벡터를 받아 벡터를 출력* 할 때 (예: $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$), 야코비안은 *모든 출력의 모든 입력에 대한 편미분 행렬* 입니다.

$$J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$

크기: $m \times n$.

신경망의 한 층 $\mathbf{y} = \mathbf{f}(\mathbf{x})$ 에 대해, *입력 x 에 대한 출력 y 의 야코비안* 이 역전파의 한 단계 입니다.

### 5.2 헤시안 (Hessian)

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$ 의 *2차 미분의 행렬* 이 헤시안입니다.

$$H = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

헤시안은 *손실 함수의 곡률 (curvature)* 을 나타냅니다.

- 헤시안의 고유값이 *모두 양수* → 함수가 *위로 굽은 그릇 모양* (지역 최소)
- 모두 음수 → *아래로 굽은 그릇* (지역 최대)
- 양수와 음수 섞임 → *안장점*

### 5.3 ML 에서의 사용

- *야코비안* — 역전파의 모든 단계에서 자동 계산
- *헤시안* — Newton 법·자연 그래디언트·신경 탄젠트 커널·Lottery Ticket 분석 같은 곳

대부분의 일반적 학습은 *1차 정보 (그래디언트) 만* 사용하므로 헤시안을 직접 다룰 일은 적습니다. 그러나 *학습이 왜 그런 식으로 진행되는가* 의 깊은 이해에는 헤시안 사고가 필요합니다.

### 5.4 챕터 정리

야코비안은 *벡터-벡터 함수의 1차 미분 행렬*, 헤시안은 *스칼라 함수의 2차 미분 행렬* 입니다. 야코비안은 신경망 역전파 자체이고, 헤시안은 손실 풍경의 *곡률* 정보를 담습니다. 다음 챕터에서는 그래디언트를 사용해 *함수의 최솟값* 을 찾는 알고리즘을 봅니다.

---

## 6. 경사하강법

### 6.1 경사하강법의 정의

*경사하강법 (Gradient Descent)* 은 *그래디언트의 반대 방향으로 조금씩 이동* 해 함수의 최솟값을 찾는 알고리즘입니다.

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

여기서:
- $\mathbf{w}_t$: 현재 파라미터
- $\eta$ (eta): *학습률 (learning rate)*
- $\nabla L$: 손실의 그래디언트

### 6.2 NumPy 로 경사하강법

```python
import numpy as np
import matplotlib.pyplot as plt

# 손실 함수: L(w) = (w - 5)^2
# 정답: w = 5
def loss(w):
    return (w - 5)**2

def grad(w):
    return 2 * (w - 5)

w = 0.0
lr = 0.1
trajectory = [w]
losses = [loss(w)]

for _ in range(50):
    w = w - lr * grad(w)
    trajectory.append(w)
    losses.append(loss(w))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선
axes[0].plot(losses)
axes[0].set_xlabel('step'); axes[0].set_ylabel('loss')
axes[0].set_title('손실 감소'); axes[0].grid(True)

# 파라미터 궤적
ws = np.linspace(-1, 11, 100)
axes[1].plot(ws, [loss(w) for w in ws], 'b-', label='loss')
axes[1].plot(trajectory, [loss(w) for w in trajectory], 'r.-', markersize=8, label='trajectory')
axes[1].set_xlabel('w'); axes[1].set_ylabel('loss')
axes[1].legend(); axes[1].set_title('파라미터 궤적'); axes[1].grid(True)

plt.show()
```

w 가 5 에 빠르게 수렴하는 모습을 그래프로 확인할 수 있습니다.

### 6.3 다차원 경사하강법

여러 파라미터에 대해서도 같은 식이 적용됩니다.

```python
def loss(W):
    # W = (w1, w2)
    return (W[0] - 3)**2 + (W[1] - 4)**2

def grad(W):
    return np.array([2*(W[0]-3), 2*(W[1]-4)])

W = np.array([0.0, 0.0])
lr = 0.1

for step in range(50):
    W = W - lr * grad(W)
    if step % 10 == 0:
        print(f"step {step}: W = {W}, loss = {loss(W):.4f}")
```

### 6.4 변형들

기본 경사하강법은 다음 변형들을 가집니다 (Vol 20 에서 자세히):

- *SGD* (Stochastic GD) — 미니배치마다 그래디언트 추정
- *Momentum* — 이전 스텝의 그래디언트를 누적해 관성 부여
- *Adam* — 모멘텀 + 적응 학습률
- *AdamW* — Adam 의 weight decay 분리
- *RMSProp, AdaGrad, NAG, Lion* 등

모두 *그래디언트 반대 방향으로 이동* 한다는 핵심은 같습니다.

### 6.5 챕터 정리

경사하강법은 *그래디언트의 반대 방향으로 학습률만큼 이동* 하는 단순한 알고리즘이며, 모든 학습 알고리즘의 골격입니다. 변형들 (SGD, Adam, AdamW 등) 은 모두 이 골격의 변주입니다. 다음 챕터에서는 *학습률* 의 의미를 깊이 봅니다.

---

## 7. 학습률의 의미와 직관

### 7.1 학습률이 만드는 차이

같은 손실 함수·같은 그래디언트라도, *학습률* 에 따라 학습 결과가 극적으로 달라집니다.

- *너무 작은 학습률* — 수렴은 하지만 *매우 느림*
- *적정 학습률* — *빠르게 수렴*
- *너무 큰 학습률* — *발산* 또는 *진동*

### 7.2 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def loss(w):
    return (w - 5)**2

def grad(w):
    return 2 * (w - 5)

learning_rates = [0.01, 0.1, 0.5, 1.05]   # 마지막은 발산
results = {}

for lr in learning_rates:
    w = 0.0
    history = [w]
    for _ in range(30):
        w = w - lr * grad(w)
        history.append(w)
    results[lr] = history

fig, ax = plt.subplots(figsize=(10, 6))
for lr, history in results.items():
    ax.plot(history, marker='o', label=f'lr={lr}')
ax.axhline(y=5, color='gray', linestyle='--', label='target=5')
ax.set_xlabel('step'); ax.set_ylabel('w')
ax.set_title('학습률에 따른 수렴 양상')
ax.legend(); ax.grid(True)
plt.show()
```

`lr=0.01` 은 천천히 수렴, `lr=0.1` 은 빠르게 수렴, `lr=0.5` 는 진동하면서 수렴, `lr=1.05` 는 *발산* 합니다.

### 7.3 학습률 선택

실무에서 *학습률은 가장 중요한 하이퍼파라미터* 입니다. 너무 크면 학습이 깨지고, 너무 작으면 시간이 너무 듭니다.

표준 시작점:
- *Adam, AdamW* — `1e-3` 또는 `1e-4`
- *SGD* — `1e-2` 또는 `1e-1`
- *대형 트랜스포머 사전학습* — `1e-4` 부터 시작 + warmup + cosine decay
- *LoRA 미세조정* — `1e-4` 부터 `5e-4`

자세한 학습률 스케줄은 Vol 61 (학습률 스케줄 깊이) 에서 다룹니다.

### 7.4 학습률 탐색 (LR Range Test)

좋은 학습률을 찾는 빠른 방법: *학습률을 지수적으로 증가시키며* 손실을 관찰. 손실이 *가장 빠르게 떨어지는 구간* 의 학습률이 좋은 시작점입니다.

```python
import numpy as np

# 가짜 손실 함수와 그래디언트 (실제로는 모델로부터)
def step(lr, w_init=0.0, steps=10):
    w = w_init
    for _ in range(steps):
        g = 2 * (w - 5)
        w = w - lr * g
    return (w - 5)**2

lrs = np.logspace(-4, 0, 30)
losses = [step(lr) for lr in lrs]

import matplotlib.pyplot as plt
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('learning rate'); plt.ylabel('loss')
plt.title('LR Range Test')
plt.grid(True)
plt.show()
```

### 7.5 챕터 정리

학습률은 *학습 결과를 좌우하는 가장 중요한 하이퍼파라미터* 이며, 너무 작거나 크면 모두 학습이 실패합니다. 옵티마이저별 표준 시작점이 있고, LR Range Test 같은 빠른 탐색 도구로 적정 값을 찾을 수 있습니다. 다음 챕터에서는 모든 도구를 NumPy 로 종합합니다.

---

## 8. NumPy 로 직접 그래디언트 계산하기

### 8.1 종합 실습

이 챕터는 지금까지의 모든 개념을 NumPy 로 직접 검증하고 구현합니다.

### 8.2 수치 미분 vs 해석적 미분

```python
import numpy as np

def f(x):
    return x**3 + 2*x**2 - 5*x + 1

def f_analytic_grad(x):
    return 3*x**2 + 4*x - 5

def f_numerical_grad(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# 비교
xs = np.linspace(-3, 3, 7)
for x in xs:
    a = f_analytic_grad(x)
    n = f_numerical_grad(f, x)
    print(f"x={x:+.1f}: 해석={a:+.4f}, 수치={n:+.4f}, 오차={abs(a-n):.2e}")
```

수치 미분은 해석적 미분과 *매우 가까운* 값을 줍니다.

### 8.3 다변수 그래디언트 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + 2*y**2

x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 해석적 그래디언트
gx = 2 * X
gy = 4 * Y

fig, ax = plt.subplots(figsize=(10, 8))
cs = ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.clabel(cs, inline=True, fontsize=8)

step = 3
ax.quiver(X[::step, ::step], Y[::step, ::step],
          -gx[::step, ::step], -gy[::step, ::step],
          color='red', label='-gradient (감소 방향)')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('등고선 + 음의 그래디언트 (감소 방향)')
ax.set_aspect('equal'); ax.legend()
plt.show()
```

화살표는 항상 *최솟값 (원점) 을 향합니다*. 이것이 경사하강법이 작동하는 이유입니다.

### 8.4 2D 경사하강법 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

def f(W):
    return W[0]**2 + 2*W[1]**2

def grad(W):
    return np.array([2*W[0], 4*W[1]])

W = np.array([2.5, 2.0])
lr = 0.1
trajectory = [W.copy()]

for _ in range(30):
    W = W - lr * grad(W)
    trajectory.append(W.copy())

trajectory = np.array(trajectory)

# 시각화
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2*Y**2

fig, ax = plt.subplots(figsize=(10, 8))
cs = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=8)
ax.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', s=100, label='시작')
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, label='끝')
ax.set_xlabel('w1'); ax.set_ylabel('w2')
ax.set_title('2D 경사하강법 궤적')
ax.legend(); ax.grid(True); ax.set_aspect('equal')
plt.show()
```

궤적이 *최솟값 (원점) 을 향해 곡선을 그리며* 수렴하는 것을 확인할 수 있습니다.

### 8.5 학습률 비교 실험

```python
import numpy as np
import matplotlib.pyplot as plt

def loss(w):
    return (w - 5)**2

def grad(w):
    return 2 * (w - 5)

learning_rates = [0.001, 0.01, 0.1, 0.5, 0.99, 1.01]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, lr in enumerate(learning_rates):
    ax = axes[idx // 3, idx % 3]
    w = 0.0
    history = [w]
    for _ in range(50):
        w = w - lr * grad(w)
        history.append(w)
    
    ax.plot(history, 'o-')
    ax.axhline(y=5, color='gray', linestyle='--')
    ax.set_title(f'lr={lr}')
    ax.set_xlabel('step'); ax.set_ylabel('w')
    ax.grid(True)

plt.tight_layout()
plt.show()
```

`lr` 가 너무 작으면 *느리고*, 너무 크면 (1 이상) *발산* 함을 한눈에 볼 수 있습니다.

### 8.6 챕터 정리

NumPy 만으로도 *수치 미분·그래디언트·경사하강법·학습률 분석* 을 모두 구현하고 시각화할 수 있습니다. 이 코드들을 직접 돌려 본 사람은 PyTorch 의 *autograd*·*optimizer* 가 무엇을 자동화해 주는지 정확히 이해하게 됩니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **미분** 은 *함수의 순간 변화율*, 그래프의 접선 기울기입니다.
- **편미분** 은 *다변수 함수에서 한 변수만 변화* 시킬 때의 변화율입니다.
- **그래디언트** 는 *모든 편미분의 묶음 벡터* 이며, 함수가 *가장 빠르게 증가하는 방향* 을 가리킵니다.
- **체인룰** 은 합성 함수의 미분 규칙이며, 신경망 *역전파* 의 본질입니다.
- **야코비안** 은 벡터-벡터 함수의 1차 미분 행렬, **헤시안** 은 스칼라 함수의 2차 미분 행렬입니다.
- **경사하강법** 은 *그래디언트의 반대 방향으로 학습률만큼 이동* 하는 단순한 알고리즘이며, 모든 학습 알고리즘의 골격입니다.
- **학습률** 은 학습 결과를 좌우하는 가장 중요한 하이퍼파라미터이며, 너무 크거나 작으면 모두 실패합니다.
- **NumPy** 로 모든 개념을 직접 시각화·구현할 수 있고, 이 경험이 PyTorch autograd 이해의 기반이 됩니다.

가장 기억에 남겨야 할 한 줄은 **"학습은 손실 함수의 그래디언트 반대 방향으로 한 걸음씩 내려가는 일이며, 모든 옵티마이저는 이 한 걸음의 변형이다."** 입니다.

다음 권은 [Volume 7 — 확률과 통계](./volume_07_probability_statistics.md) 입니다. 거기서는 *모든 ML 손실 함수가 확률 분포에서 유도된다* 는 사실을 이해합니다.

---

## 자가점검 키워드

`미분`, `편미분`, `그래디언트`, `체인룰`, `야코비안`, `헤시안`, `경사하강법`, `학습률`

## 자가점검 질문

다음 질문에 막힘없이 답할 수 있을 때 다음 권으로 넘어가십시오.

1. *수치 미분 (중심 차분)* 의 공식을 적고, NumPy 로 한 줄로 구현하십시오.
2. 편미분과 그래디언트의 차이를 한 문단으로 설명하십시오.
3. 그래디언트 벡터가 *함수가 가장 빠르게 증가하는 방향* 임을 등고선 그림으로 설명하십시오.
4. 체인룰을 사용해 $f(x) = (x^3 + 1)^4$ 의 미분을 계산하고, NumPy 로 검증하십시오.
5. 야코비안과 헤시안의 차이를 *입력 차원·출력 차원·미분 차수* 로 분류해 적으십시오.
6. 학습률이 너무 클 때와 너무 작을 때 학습이 어떻게 실패하는지 시각화하십시오.
7. 신경망 학습이 *그래디언트의 반대 방향으로 이동* 하는 일이라는 명제를 한 문단으로 자기 말로 설명하십시오.

## 다음 권

[Volume 7 — 확률과 통계](./volume_07_probability_statistics.md)
