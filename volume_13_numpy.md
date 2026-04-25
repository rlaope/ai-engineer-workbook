# Volume 13 — NumPy 로 다시 푸는 수학

> 이 권이 끝나면 머릿속의 수학 개념을 즉시 NumPy 코드로 옮기고 결과를 확인하는 습관이 몸에 배게 됩니다.

## 목적

수학을 *읽기만* 한 학습은 시간이 지나면 잊힙니다. 손으로 코드로 다시 쳐 보고 변수의 모양과 값을 눈으로 확인할 때 비로소 직관이 생깁니다. NumPy 는 PyTorch 의 텐서와 거의 같은 인터페이스를 가지므로, NumPy 로 익힌 사고법은 그대로 딥러닝으로 옮겨갑니다. 이 권은 Volume 8–8 의 모든 핵심 개념을 *작은 코드 조각* 으로 다시 점검합니다.

## 선수 지식

- Volume 8–8 완료
- 외부 지식: Python 기본 문법, 가상환경

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. NumPy 의 브로드캐스팅 규칙을 사고의 도구로 사용할 수 있습니다.
2. 행렬 연산의 모양 변화를 쓰지 않고 머릿속에서 계산할 수 있습니다.
3. PCA·KL 발산·경사하강법을 NumPy 만으로 구현할 수 있습니다.
4. NumPy 와 PyTorch 텐서의 공통점·차이점을 구분할 수 있습니다.
5. 벡터화 사고법으로 for 루프를 행렬 연산으로 바꿀 수 있습니다.

---

## 이 권을 읽기 전에

이 권은 *새로운 개념* 을 배우는 권이 아니라 *지금까지 배운 것을 NumPy 코드로 통합* 하는 권입니다. 책상 옆에 노트북을 두고, 모든 코드를 직접 실행하시기 바랍니다. 한 번에 다 읽기보다 *한 챕터씩 손으로 따라치며* 진행하면 학습 효과가 가장 큽니다.

NumPy 의 핵심 사고는 *벡터화 (vectorization)* 와 *브로드캐스팅 (broadcasting)* 입니다. 이 두 사고를 손에 익히면 *for 루프 없는* 효율적인 코드를 자연스럽게 쓰게 되며, PyTorch 로 넘어갈 때도 같은 사고가 그대로 적용됩니다.

---

## 1. NumPy 배열의 기본

### 1.1 ndarray 생성

```python
import numpy as np

# 리스트로부터
a = np.array([1, 2, 3])

# 0 으로 채우기
zeros = np.zeros((3, 4))

# 1 로 채우기
ones = np.ones((2, 3, 4))

# 범위
r = np.arange(0, 10, 2)            # [0 2 4 6 8]

# 등간격 분할
ls = np.linspace(0, 1, 5)          # [0. 0.25 0.5 0.75 1.]

# 랜덤
rnd = np.random.randn(3, 4)        # 정규 분포
uni = np.random.uniform(0, 1, 5)   # 균등 분포
```

### 1.2 모양·자료형·디바이스

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)      # (2, 3)
print(a.dtype)      # int64
print(a.ndim)       # 2
print(a.size)       # 6
```

자료형 명시:

```python
a = np.array([1, 2, 3], dtype=np.float32)
b = a.astype(np.float64)
```

### 1.3 챕터 정리

NumPy 배열은 *모양·자료형·차원* 의 세 메타데이터로 정의됩니다. `np.zeros`·`np.ones`·`np.arange`·`np.linspace`·`np.random` 이 가장 자주 쓰는 생성 함수입니다.

---

## 2. 브로드캐스팅

### 2.1 규칙

브로드캐스팅은 *모양이 다른 배열을 자동으로 호환되게* 만드는 규칙입니다. 맨 뒤 차원부터 비교해, *같거나 1 인 경우* 호환됩니다.

```python
import numpy as np

# (3,) + (3, 1) → (3, 3)
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
print(a + b)
# [[11 12 13]
#  [21 22 23]
#  [31 32 33]]
```

### 2.2 흔한 패턴

```python
# 행렬의 각 행에서 평균 빼기
X = np.random.randn(100, 5)
mean = X.mean(axis=0)             # (5,)
X_centered = X - mean             # 자동 브로드캐스트

# 행마다 정규화
X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
```

### 2.3 챕터 정리

브로드캐스팅을 활용하면 *for 루프 없이* 행 단위·열 단위 연산을 쓸 수 있습니다. *axis* 와 *keepdims* 를 의식적으로 사용해 모양을 통제하시기 바랍니다.

---

## 3. 인덱싱과 슬라이싱

### 3.1 기본

```python
a = np.arange(20).reshape(4, 5)
print(a[0])           # 첫 행
print(a[:, 0])        # 첫 열
print(a[1:3, 2:4])    # 부분 행렬
```

### 3.2 부울 마스크

```python
x = np.array([1, -2, 3, -4, 5])
mask = x > 0
print(x[mask])        # [1 3 5]

# 조건부 갱신
x[x < 0] = 0
print(x)              # [1 0 3 0 5]
```

### 3.3 팬시 인덱싱

```python
a = np.arange(10) * 10
indices = [3, 1, 4, 1, 5]
print(a[indices])     # [30 10 40 10 50]

# 행 선택
A = np.arange(12).reshape(4, 3)
print(A[[0, 2]])      # 0 행과 2 행
```

### 3.4 챕터 정리

인덱싱은 *기본 슬라이스·부울 마스크·팬시 인덱싱* 의 세 형태가 있습니다. 부울 마스크는 *조건 기반 선택·갱신* 에 매우 강력합니다.

---

## 4. 벡터화

### 4.1 for 루프를 행렬 연산으로

같은 결과를 *명시적 루프* 와 *벡터화* 로 비교:

```python
import numpy as np
import time

n = 1000000
a = np.random.randn(n)
b = np.random.randn(n)

# for 루프 (느림)
start = time.time()
result = np.zeros(n)
for i in range(n):
    result[i] = a[i] * b[i]
print(f"for: {time.time() - start:.4f}s")

# 벡터화 (빠름)
start = time.time()
result = a * b
print(f"vectorized: {time.time() - start:.4f}s")
```

벡터화 버전이 *수십~수백 배* 빠릅니다.

### 4.2 보다 복잡한 예시

두 점 집합 사이의 모든 거리 계산:

```python
# A: (m, d), B: (n, d) → 거리 행렬 (m, n)
A = np.random.randn(100, 5)
B = np.random.randn(50, 5)

# 벡터화: A[:, None, :] - B[None, :, :] → (100, 50, 5)
diff = A[:, None, :] - B[None, :, :]
dist = np.linalg.norm(diff, axis=-1)
print(dist.shape)     # (100, 50)
```

### 4.3 챕터 정리

벡터화는 *NumPy 의 가장 큰 가치* 이며, for 루프를 행렬 연산으로 바꾸는 사고입니다. 코드가 간결해지고 *수십~수백 배 빨라* 집니다.

---

## 5. 선형대수 모듈

### 5.1 자주 쓰는 함수

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

print(np.linalg.det(A))       # 행렬식
print(np.linalg.inv(A))       # 역행렬
print(np.linalg.matrix_rank(A)) # 랭크
print(np.linalg.norm(A))      # 프로베니우스 노름

# 고유값·고유벡터
w, v = np.linalg.eig(A)
print("고유값:", w)

# SVD
U, S, Vt = np.linalg.svd(A)
print("특이값:", S)

# 선형 시스템 해 (Ax = b)
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("x:", x)
```

### 5.2 챕터 정리

`np.linalg` 모듈이 *대부분의 선형대수 연산* 을 한 줄로 제공합니다. 직접 구현하기 전에 항상 이 모듈을 먼저 확인하시기 바랍니다.

---

## 6. 무작위성

### 6.1 시드와 재현성

```python
np.random.seed(42)
print(np.random.randn(3))    # 동일한 결과
np.random.seed(42)
print(np.random.randn(3))    # 같은 값
```

ML 실험에서 *시드 고정* 은 재현성의 기본입니다.

### 6.2 분포

```python
np.random.randn(100)             # 표준 정규
np.random.normal(mu, sigma, 100) # 일반 정규
np.random.uniform(0, 1, 100)     # 균등
np.random.binomial(10, 0.5, 100) # 이항
np.random.choice([1,2,3], 100, p=[0.5, 0.3, 0.2])  # 가중 선택
```

### 6.3 새 API (Generator)

```python
rng = np.random.default_rng(seed=42)
print(rng.normal(0, 1, 5))
```

새 API 는 *글로벌 상태를 공유하지 않아* 더 안전합니다. 큰 프로젝트에서는 이쪽을 권장.

### 6.4 챕터 정리

NumPy 의 무작위 함수는 *시드 고정* 으로 재현 가능합니다. 새 API (`default_rng`) 가 글로벌 상태 문제를 피해 더 안전합니다.

---

## 7. PCA 직접 구현

### 7.1 코드 한 화면

```python
import numpy as np

def pca(X, n_components):
    # 중심화
    X_centered = X - X.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # 상위 k 개 주성분
    return X_centered @ Vt[:n_components].T

# 테스트
np.random.seed(0)
X = np.random.randn(200, 10)
X_pca = pca(X, 2)
print(X_pca.shape)            # (200, 2)
```

### 7.2 챕터 정리

PCA 는 *3 줄의 NumPy 코드* 로 구현됩니다. SVD 와 행렬 곱만 알면 됩니다.

---

## 8. KL·경사하강법 구현

### 8.1 KL 발산

```python
def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(p * np.log((p + eps) / (q + eps)))
```

### 8.2 경사하강법

```python
def gradient_descent(loss_fn, grad_fn, w_init, lr=0.1, steps=100):
    w = w_init.copy()
    history = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad_fn(w)
        history.append(w.copy())
    return w, history

# 사용 예
def loss(w):
    return ((w - 5)**2).sum()

def grad(w):
    return 2 * (w - 5)

w_final, hist = gradient_descent(loss, grad, np.array([0.0, 0.0]))
print(w_final)               # [5. 5.]
```

### 8.3 챕터 정리

KL 과 경사하강법은 각각 *수 줄 안* 에서 구현됩니다. 핵심 알고리즘들이 *수학을 코드로 옮기는 일* 임을 확인합니다.

---

## 9. NumPy ↔ PyTorch 텐서

### 9.1 변환

```python
import numpy as np
import torch

# NumPy → PyTorch
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)

# PyTorch → NumPy
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()
```

### 9.2 공통점과 차이

공통점:
- shape·dtype·인덱싱·브로드캐스팅 같은 인터페이스
- 행렬 연산의 함수명 (`@`, `matmul`, `sum`, `mean` 등)

차이:
- PyTorch 텐서는 *GPU 로 이동 가능* (`t.to('cuda')`)
- PyTorch 는 *autograd* 를 가짐 (`requires_grad=True`)
- NumPy 는 *항상 CPU·numpy 배열* 이고 *그래디언트 추적 없음*

### 9.3 챕터 정리

NumPy 사고는 PyTorch 로 *거의 1:1 이전* 됩니다. PyTorch 가 추가하는 것은 *GPU + autograd* 입니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **ndarray** — 모양·자료형·차원의 메타데이터
- **브로드캐스팅** — 모양이 다른 배열의 자동 정렬
- **인덱싱** — 슬라이스·부울 마스크·팬시 인덱싱
- **벡터화** — for 루프를 행렬 연산으로
- **선형대수 모듈** — `np.linalg` 의 표준 함수들
- **무작위성** — 시드와 재현성
- **PCA·KL·경사하강법** — 모두 수 줄로 구현 가능
- **NumPy ↔ PyTorch** — 거의 같은 인터페이스 + GPU/autograd

가장 기억에 남겨야 할 한 줄은 **"수학을 코드로 옮기는 사고가 NumPy 에서 시작되며, 그대로 PyTorch 로 이전된다."** 입니다.

다음 권은 [Volume 20 — 머신러닝의 본질과 학습 패러다임](./volume_20_machine_learning_essence.md) 입니다.

---

## 자가점검 키워드

`ndarray`, `브로드캐스팅`, `벡터화`, `np.linalg`, `random/seed`, `PCA`, `KL`, `경사하강법`

## 자가점검 질문

1. NumPy 의 *브로드캐스팅 규칙* 을 한 문단으로 설명하고 예시 두 개를 적으십시오.
2. for 루프와 벡터화의 속도 차이를 직접 측정해 보십시오.
3. PCA·KL·경사하강법 각각을 NumPy 5 줄 이내로 구현하십시오.
4. NumPy 와 PyTorch 텐서의 공통점·차이점을 표로 정리하십시오.
5. *시드 고정* 이 ML 실험에서 왜 중요한지 한 문단으로 설명하십시오.
6. 두 점 집합 사이의 *모든 쌍 거리* 를 벡터화로 한 줄에 구현하십시오.
7. `np.linalg` 모듈의 함수 5 개를 나열하고 각각의 용도를 적으십시오.

## 다음 권

[Volume 20 — 머신러닝의 본질과 학습 패러다임](./volume_20_machine_learning_essence.md)
