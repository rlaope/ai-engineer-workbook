# Volume 8 — 선형대수 1 — 벡터·행렬·내적·노름

> 이 권이 끝나면 모든 데이터를 *벡터·행렬·텐서* 로 추상화해 사고할 수 있게 됩니다.

## 목적

딥러닝의 모든 입력·출력·파라미터는 결국 *숫자의 배열* 입니다. 이 배열을 다루는 언어가 선형대수입니다. 벡터의 의미·행렬 곱의 기하학적 의미·내적이 *유사도* 가 되는 이유·노름이 *크기* 를 재는 방식을 직관적으로 잡지 못하면 이후의 모든 모델 설명이 *수식만 따라 읽는 일* 이 됩니다. 이 권은 코드와 그림으로 선형대수의 가장 핵심적인 부분을 다시 익힙니다.

## 선수 지식

- Volume 1–3 완료
- 외부 지식: 고등학교 수학 수준 (좌표 평면·벡터 합)

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 벡터와 행렬을 NumPy 로 만들고 기본 연산을 수행할 수 있습니다.
2. 행렬 곱의 의미를 *한 공간에서 다른 공간으로의 사상* 으로 설명할 수 있습니다.
3. 두 벡터의 내적이 *유사도* 인 이유를 코사인 관점에서 설명할 수 있습니다.
4. L1·L2·L∞ 노름의 차이를 그림으로 그릴 수 있습니다.
5. 임베딩 검색의 코어가 *내적/코사인 유사도 정렬* 임을 알게 됩니다.

---

## 이 권을 읽기 전에

선형대수는 많은 사람에게 *학교에서 배웠지만 잊은 과목* 입니다. 이 권의 목적은 *수학과 학생을 만드는 것* 이 아니라, *AI 모델 안에서 일어나는 일* 을 머릿속에 그릴 수 있게 만드는 것입니다.

추상적인 정의보다 *구체적인 NumPy 코드* 와 *기하학적 그림* 으로 접근합니다. 모든 개념은 *왜 이게 ML 에 필요한가* 라는 질문에 답하는 형태로 도입됩니다. 예를 들어, *코사인 유사도* 는 단순한 수학적 정의가 아니라 *RAG 가 작동하는 근본 원리* 로 도입됩니다.

이 권을 마칠 때 자기 노트북의 NumPy 가 *손에 익숙* 해져 있어야 합니다. 모든 코드 블록을 직접 실행하시기 바랍니다.

---

## 1. 벡터 — 방향과 크기를 가진 양

### 1.1 벡터의 두 가지 정의

벡터는 두 가지 방식으로 정의할 수 있습니다.

**기하학적 정의** — *방향과 크기를 가진 화살표.* 2D 평면이나 3D 공간에서 화살표 하나가 곧 벡터입니다.

**대수학적 정의** — *순서가 있는 숫자의 묶음.* `(3, 4)` 같은 두 숫자, `(1, 2, 3)` 같은 세 숫자, 또는 `(0.1, -0.5, 0.7, 0.2, ...)` 같은 768 개의 숫자도 벡터입니다.

이 두 정의는 *완전히 동등* 합니다. 2D 의 화살표 `→ (3, 4)` 는 *원점에서 (3, 4) 좌표까지의 화살표* 와 같습니다. 차원이 4 이상이 되면 머릿속에 그림으로 그릴 수 없지만, *대수학적 정의는 여전히 명확* 합니다.

ML 에서 다루는 벡터는 거의 항상 *고차원* 입니다. 768 차원 (BERT 임베딩), 1536 차원 (OpenAI Embedding), 4096 차원 (LLaMA 히든 스테이트) 같은 벡터들이 일상적으로 등장합니다. 사람이 시각화할 수는 없지만, *대수학적으로* 다루는 일은 NumPy 한 줄로 가능합니다.

### 1.2 NumPy 로 벡터 만들기

```python
import numpy as np

# 2 차원 벡터
v = np.array([3, 4])
print(v)            # [3 4]
print(v.shape)      # (2,)
print(v.dtype)      # int64

# 4 차원 벡터 (실수)
u = np.array([0.1, -0.5, 0.7, 0.2])
print(u.shape)      # (4,)

# 768 차원 벡터 (랜덤)
emb = np.random.randn(768)
print(emb.shape)    # (768,)
```

`shape` 가 `(N,)` 인 형태가 1차원 벡터입니다. 이는 *N 개의 원소를 가진 1차원 배열* 을 의미합니다.

### 1.3 벡터 연산

벡터의 기본 연산:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 덧셈 — 같은 위치의 원소끼리
print(a + b)        # [5 7 9]

# 뺄셈
print(a - b)        # [-3 -3 -3]

# 스칼라 곱
print(2 * a)        # [2 4 6]

# 원소별 곱 (Hadamard product)
print(a * b)        # [ 4 10 18]

# 원소별 나눗셈
print(b / a)        # [4.  2.5 2. ]
```

기하학적으로:

- *덧셈* 은 두 화살표를 *이어 붙이는 것* 입니다.
- *스칼라 곱* 은 화살표를 *늘리거나 줄이는 것* 입니다 (음수면 방향 반전).
- *뺄셈* 은 *벡터 b 의 끝에서 a 로 가는 화살표* 입니다.

### 1.4 ML 에서 벡터의 역할

ML 에서 벡터가 등장하는 가장 흔한 형태:

**1. 입력 특성 벡터.** 한 샘플이 가진 모든 특성을 묶은 벡터. 예: 사람 한 명의 [나이, 키, 몸무게, 학력, ...] 같은 N 개 특성.

**2. 임베딩.** 단어·문장·이미지·사용자 같은 *추상적 개체* 를 *고차원 벡터* 로 표현한 것. 예: `cat` 단어의 768 차원 임베딩.

**3. 모델 출력.** 분류 모델이 예측한 *클래스별 확률* 의 벡터. 예: `[0.1, 0.7, 0.2]` (3 개 클래스의 확률).

**4. 그래디언트.** 손실 함수가 각 파라미터에 대해 가지는 편미분 값을 묶은 벡터.

**5. 모델 가중치.** 파라미터 자체. 1B 모델은 *10 억 차원의 벡터* 입니다.

### 1.5 챕터 정리

벡터는 *방향과 크기를 가진 양* 이며 동시에 *순서가 있는 숫자의 묶음* 입니다. NumPy `np.array` 로 다루며, 덧셈·스칼라 곱·원소별 곱이 기본 연산입니다. ML 에서는 입력 특성·임베딩·모델 출력·그래디언트·가중치 모두가 벡터입니다. 다음 챕터에서는 벡터를 *변환하는* 도구인 행렬을 봅니다.

---

## 2. 행렬 — 벡터를 변환하는 함수

### 2.1 행렬의 정의

행렬은 *숫자를 사각형 격자로 배열한 것* 입니다. m 행 × n 열의 형태로 적습니다.

```
| 1  2  3 |
| 4  5  6 |   ← 2x3 행렬
```

NumPy 에서:

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)     # (2, 3)
print(A.dtype)     # int64
print(A)
# [[1 2 3]
#  [4 5 6]]
```

shape `(2, 3)` 은 *2 행 3 열* 을 의미합니다.

### 2.2 행렬 곱의 정의

두 행렬을 곱할 수 있는 조건은 *왼쪽의 열 수 = 오른쪽의 행 수* 입니다.

A 가 (m, n), B 가 (n, p) 일 때, A @ B 의 결과는 (m, p) 입니다.

각 원소는 *왼쪽 행과 오른쪽 열의 내적* 입니다.

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

C = A @ B
# C[0,0] = 1*5 + 2*7 = 19
# C[0,1] = 1*6 + 2*8 = 22
# C[1,0] = 3*5 + 4*7 = 43
# C[1,1] = 3*6 + 4*8 = 50

print(C)
# [[19 22]
#  [43 50]]
```

NumPy 에서 `@` 가 행렬 곱 연산자이며, `np.matmul(A, B)` 와 동일합니다. *주의: `A * B` 는 원소별 곱이지 행렬 곱이 아닙니다.*

### 2.3 행렬 = 함수

행렬의 가장 중요한 해석은 *함수* 입니다. m × n 행렬 A 는 *n 차원 벡터를 m 차원 벡터로 보내는 함수* 입니다.

```python
A = np.array([[1, 0, 1],
              [0, 1, 1]])  # 2 × 3 행렬
v = np.array([1, 2, 3])     # 3 차원 벡터

result = A @ v
print(result)               # [4 5]  ← 2 차원 벡터
```

이 행렬 A 는 *3 차원 벡터를 2 차원 벡터로 변환* 하는 함수입니다.

신경망의 한 층은 *행렬 곱 + 비선형성* 입니다.

```python
# nn.Linear(in_features=3, out_features=2) 의 본질:
W = np.random.randn(2, 3)   # 가중치 행렬
b = np.random.randn(2)      # 바이어스
x = np.array([0.1, 0.5, -0.3])

# 선형 변환
y = W @ x + b
print(y.shape)              # (2,)

# 비선형성 (ReLU)
y_activated = np.maximum(0, y)
```

이 한 줄이 *완전 연결층* 의 본질이며, 모든 신경망의 빌딩블록입니다.

### 2.4 특별한 행렬들

다음 행렬들은 자주 등장합니다.

**단위 행렬 (Identity matrix).** 대각선이 1 이고 나머지가 0. 어떤 벡터에 곱해도 *그대로* 돌려줍니다.

```python
I = np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

v = np.array([1, 2, 3])
print(I @ v)  # [1. 2. 3.]
```

**대각 행렬 (Diagonal matrix).** 대각선만 0 이 아니고 나머지는 0.

```python
D = np.diag([2, 3, 5])
# [[2 0 0]
#  [0 3 0]
#  [0 0 5]]
```

**전치 행렬 (Transpose).** 행과 열을 바꾼 것. `A.T` 로 표기.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(A.T.shape)   # (3, 2)
```

**역행렬 (Inverse).** `A @ A_inv = I` 인 행렬. 정사각 행렬에만 정의됩니다.

```python
A = np.array([[1, 2],
              [3, 4]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)
# [[1.0000000e+00 0.0000000e+00]
#  [8.8817842e-16 1.0000000e+00]]   ← 부동소수점 오차
```

### 2.5 챕터 정리

행렬은 *숫자의 사각형 격자* 이며, 가장 중요한 해석은 *벡터를 다른 차원의 벡터로 보내는 함수* 입니다. 행렬 곱은 NumPy 의 `@` 연산자로 수행하며, 신경망의 한 층은 본질적으로 *행렬 곱 + 비선형성* 입니다. 단위 행렬·대각 행렬·전치·역행렬이 자주 등장하는 특별한 형태들입니다. 다음 챕터에서는 *행렬을 더 일반화한* 텐서를 봅니다.

---

## 3. 텐서 — n 차원 배열의 일반화

### 3.1 차원의 일반화

텐서는 *n 차원 배열의 일반화된 이름* 입니다.

- 0 차원 텐서: 스칼라 (단일 숫자)
- 1 차원 텐서: 벡터
- 2 차원 텐서: 행렬
- 3 차원 이상: 일반 텐서

PyTorch·TensorFlow 같은 프레임워크의 이름이 *Tensor* 로 시작하는 이유가 여기에 있습니다. 딥러닝의 모든 데이터는 텐서입니다.

```python
import numpy as np

# 0 차원 (스칼라)
s = np.array(5)
print(s.shape)              # ()

# 1 차원 (벡터)
v = np.array([1, 2, 3])
print(v.shape)              # (3,)

# 2 차원 (행렬)
m = np.array([[1, 2], [3, 4]])
print(m.shape)              # (2, 2)

# 3 차원
t = np.zeros((4, 5, 6))
print(t.shape)              # (4, 5, 6)

# 4 차원 (이미지 배치)
img_batch = np.zeros((32, 3, 224, 224))
print(img_batch.shape)      # (32, 3, 224, 224)
```

마지막 4 차원 예시는 *32 장의 RGB (3 채널) 224×224 이미지* 를 표현합니다. 딥러닝에서 자주 보는 모양입니다.

### 3.2 ML 에서의 표준 모양 규약

다양한 데이터 종류별 표준 텐서 모양:

**이미지 (단일).** `(C, H, W)` — 채널, 높이, 너비. 예: `(3, 224, 224)`.

**이미지 배치.** `(B, C, H, W)` — 배치 크기, 채널, 높이, 너비. 예: `(32, 3, 224, 224)`.

**텍스트 시퀀스 (단일).** `(L,)` — 시퀀스 길이 (토큰 ID). 예: `(128,)`.

**텍스트 시퀀스 임베딩.** `(L, D)` — 길이, 임베딩 차원. 예: `(128, 768)`.

**텍스트 배치.** `(B, L, D)` — 배치, 길이, 차원. 예: `(8, 128, 768)`.

**비디오.** `(B, T, C, H, W)` — 배치, 시간 프레임, 채널, 높이, 너비.

이 규약은 *프레임워크에 따라* 또는 *라이브러리에 따라* 약간씩 다를 수 있습니다. PyTorch 는 `(B, C, H, W)` 가 표준이고, TensorFlow 는 `(B, H, W, C)` 가 기본입니다 (NHWC 라고 부름).

새 코드를 만나면 *모양을 항상 의식* 하시기 바랍니다. 모양이 의도한 것과 다르면 즉시 버그가 생깁니다.

### 3.3 텐서 모양 조작

자주 쓰는 모양 조작:

```python
import numpy as np

x = np.zeros((2, 3, 4))
print(x.shape)              # (2, 3, 4)

# Reshape — 원소 수가 같으면 모양 변경 가능
y = x.reshape(6, 4)
print(y.shape)              # (6, 4)

# 차원 추가 — np.newaxis 또는 None
z = x[np.newaxis, :, :, :]
print(z.shape)              # (1, 2, 3, 4)

# Transpose — 차원 순서 변경
w = x.transpose(1, 2, 0)    # (3, 4, 2)
print(w.shape)

# Squeeze — 크기 1 인 차원 제거
a = np.zeros((1, 5, 1, 3))
print(a.squeeze().shape)    # (5, 3)
```

### 3.4 브로드캐스팅

NumPy 의 *브로드캐스팅* 은 *모양이 다른 두 텐서를 자동으로 호환되게 만드는* 규칙입니다.

```python
# 행렬 + 벡터
M = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)
v = np.array([10, 20, 30])  # (3,)

result = M + v
print(result)
# [[11 22 33]
#  [14 25 36]]
```

여기서 v 는 자동으로 *모든 행에 같은 값으로 더해집니다*. 이 규칙은 *맨 뒤 차원부터 비교* 하며, *같거나 1 인 경우* 호환됩니다.

브로드캐스팅 규칙:

```
a.shape:  (2, 3, 4)
b.shape:     (3, 4)    ← 자동으로 (1, 3, 4) 로 확장
        ────────────
result:   (2, 3, 4)


a.shape:  (5, 1, 3)
b.shape:  (1, 4, 3)    ← 자동으로 (5, 4, 3) 로 확장
        ────────────
result:   (5, 4, 3)
```

브로드캐스팅을 활용하면 *명시적 반복 없이* 효율적인 코드를 쓸 수 있습니다.

### 3.5 챕터 정리

텐서는 *n 차원 배열의 일반화* 이며, 딥러닝의 모든 데이터는 텐서로 표현됩니다. 데이터 종류별 표준 모양 규약 (이미지 BCHW, 텍스트 BLD 등) 을 익혀 두면 코드의 모양 흐름을 즉시 따라갈 수 있습니다. 브로드캐스팅 규칙을 활용하면 *반복 없이 효율적인 텐서 연산* 이 가능합니다. 다음 챕터에서는 두 벡터의 *유사도* 를 재는 핵심 도구인 내적과 외적을 봅니다.

---

## 4. 내적과 외적

### 4.1 내적의 정의

두 벡터 $\mathbf{a}$ 와 $\mathbf{b}$ 의 내적 (dot product) 은 *원소별 곱의 합* 입니다.

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

NumPy 에서:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 세 가지 동등한 방법
print(np.dot(a, b))         # 32
print(a @ b)                 # 32
print((a * b).sum())         # 32

# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

### 4.2 내적의 기하학적 의미

내적은 단순한 합이 아니라, 다음 식으로도 표현됩니다.

$$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| \, |\mathbf{b}| \, \cos\theta$$

여기서 $|\mathbf{a}|$ 는 벡터 $\mathbf{a}$ 의 크기 (노름), $\theta$ 는 두 벡터 사이의 각도입니다.

이 식이 의미하는 것은:

- 두 벡터가 *같은 방향* 이면 ($\theta = 0$, $\cos\theta = 1$) 내적이 *최대*
- 두 벡터가 *수직* 이면 ($\theta = 90°$, $\cos\theta = 0$) 내적이 *0*
- 두 벡터가 *반대 방향* 이면 ($\theta = 180°$, $\cos\theta = -1$) 내적이 *최소 (음수)*

따라서 *내적의 부호와 크기* 가 *두 벡터가 얼마나 같은 방향을 향하는가* 를 알려 줍니다.

### 4.3 내적 = 유사도

이 사실은 ML 에서 결정적입니다. 두 임베딩 벡터의 내적은 그 두 임베딩이 *얼마나 비슷한 의미를 가지는가* 를 측정합니다.

```python
# 'cat' 과 'kitten' 의 임베딩이 비슷하다면 내적이 큼
cat = np.array([0.8, 0.1, -0.3, 0.5])
kitten = np.array([0.7, 0.15, -0.25, 0.45])
dog = np.array([0.6, 0.3, -0.1, 0.2])
car = np.array([-0.4, 0.7, 0.8, -0.1])

print(np.dot(cat, kitten))   # 큼 (둘 다 작은 고양이류)
print(np.dot(cat, dog))      # 중간
print(np.dot(cat, car))      # 작거나 음수 (전혀 다름)
```

이것이 *RAG·시멘틱 검색·추천 시스템* 의 코어입니다. 모든 *벡터 유사도 검색* 은 본질적으로 *내적 (또는 그 정규화 변형) 의 정렬* 입니다.

### 4.4 행렬 곱은 내적의 묶음

행렬 곱 `A @ B` 의 각 원소는 *A 의 행* 과 *B 의 열* 의 내적입니다.

```python
A = np.array([[1, 2],
              [3, 4]])     # (2, 2)
B = np.array([[5, 6],
              [7, 8]])     # (2, 2)

C = A @ B
# C[0,0] = A[0,:] · B[:,0] = [1,2]·[5,7] = 5 + 14 = 19
# C[0,1] = A[0,:] · B[:,1] = [1,2]·[6,8] = 6 + 16 = 22
# C[1,0] = A[1,:] · B[:,0] = [3,4]·[5,7] = 15 + 28 = 43
# C[1,1] = A[1,:] · B[:,1] = [3,4]·[6,8] = 18 + 32 = 50

print(C)
# [[19 22]
#  [43 50]]
```

따라서 *큰 행렬 곱* 은 *수많은 내적을 한 번에* 계산하는 일이며, GPU 가 그것을 매우 잘하도록 설계되어 있습니다. Tensor Core·CUTLASS·cuBLAS 같은 가속 라이브러리가 모두 *행렬 곱 = 내적의 대량 병렬 처리* 에 최적화되어 있습니다.

### 4.5 외적

외적 (outer product) 은 두 벡터로 *행렬* 을 만듭니다.

$$\mathbf{a} \otimes \mathbf{b} = \mathbf{a} \mathbf{b}^T$$

```python
a = np.array([1, 2, 3])     # (3,)
b = np.array([4, 5])        # (2,)

# np.outer 또는 명시적 reshape
result = np.outer(a, b)
print(result.shape)          # (3, 2)
print(result)
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
```

외적은 ML 에서는 내적만큼 자주 쓰이진 않지만, *low-rank 분해·LoRA 의 ΔW = AB 표현* 같은 곳에 등장합니다.

### 4.6 챕터 정리

내적은 두 벡터의 *원소별 곱의 합* 이며, 기하학적으로는 *두 벡터의 방향 일치도* 를 측정합니다. 따라서 *유사도 = 내적* 이라는 등식이 성립하며, 이것이 RAG·검색·추천의 코어입니다. 행렬 곱은 *수많은 내적의 묶음* 이고, GPU 가속의 가장 큰 표적입니다. 다음 챕터에서는 벡터의 *크기* 자체를 다루는 노름을 봅니다.

---

## 5. 노름 — 크기를 재는 방법

### 5.1 노름의 정의

노름 (norm) 은 *벡터의 크기* 를 재는 방법이며, 한 가지 방법이 아니라 여러 방법이 있습니다. 표기는 $\|\mathbf{v}\|$ 또는 $\|\mathbf{v}\|_p$ 입니다.

가장 흔한 세 노름:

**L2 노름 (Euclidean norm).** 가장 익숙한 *피타고라스 거리*.

$$\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$$

**L1 노름 (Manhattan norm).** 절댓값의 합.

$$\|\mathbf{v}\|_1 = \sum_i |v_i|$$

**L∞ 노름 (max norm).** 절댓값의 최댓값.

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

NumPy 로 모두 계산:

```python
import numpy as np

v = np.array([3, 4])

print(np.linalg.norm(v))            # 5.0   (기본값 = L2)
print(np.linalg.norm(v, ord=2))     # 5.0
print(np.linalg.norm(v, ord=1))     # 7.0   (3 + 4)
print(np.linalg.norm(v, ord=np.inf))# 4.0   (max(3, 4))
```

### 5.2 노름의 기하학적 의미

각 노름의 *단위 원 (norm = 1 인 점들의 집합)* 을 그려 보면 차이가 보입니다.

```
L2 노름의 단위 원:        진짜 원
        ___
       /   \
      |  o  |
       \___/

L1 노름의 단위 원:        마름모
         /\
        /  \
       <    >
        \  /
         \/

L∞ 노름의 단위 원:        정사각형
       _____
      |     |
      |  o  |
      |_____|
```

L2 는 *대각선 방향에 페널티가 적고*, L1 은 *축 방향에 더 가까운 점을 선호* 하며, L∞ 는 *가장 큰 좌표만 봅니다*.

### 5.3 정규화 (Normalization)

벡터를 *L2 노름 1* 로 만드는 일을 *L2 정규화* 라 부릅니다.

```python
v = np.array([3, 4])
v_normalized = v / np.linalg.norm(v)
print(v_normalized)              # [0.6 0.8]
print(np.linalg.norm(v_normalized))  # 1.0
```

정규화된 두 벡터의 *내적* 은 *코사인 유사도* 와 동등합니다 (다음 챕터에서).

### 5.4 ML 에서 노름의 역할

**1. 가중치 정칙화.** 모델의 가중치 노름이 너무 커지지 않도록 손실 함수에 *L1 또는 L2 노름 페널티* 를 더합니다 (Lasso, Ridge).

```python
loss = mse_loss + lambda_l2 * np.linalg.norm(weights)**2
```

**2. 그래디언트 클리핑.** 학습 중 그래디언트의 노름이 임계값을 넘으면 스케일을 줄여 *발산을 방지* 합니다.

```python
grad_norm = np.linalg.norm(gradients)
if grad_norm > max_norm:
    gradients = gradients * (max_norm / grad_norm)
```

**3. 임베딩 정규화.** 검색·유사도 계산 전에 임베딩을 *L2 정규화* 합니다.

**4. 거리 측정.** 두 점 사이의 거리는 *그 차이의 노름* 입니다.

```python
distance = np.linalg.norm(point_a - point_b)
```

### 5.5 챕터 정리

노름은 *벡터의 크기를 재는 방법* 이며, L1·L2·L∞ 가 가장 흔한 세 가지입니다. 각각 다른 단위 원 모양을 가지며, ML 에서는 *정칙화·그래디언트 클리핑·임베딩 정규화·거리 측정* 에 사용됩니다. 다음 챕터에서는 정규화 + 내적의 결합인 *코사인 유사도* 가 어떻게 RAG 의 기반이 되는지를 봅니다.

---

## 6. 코사인 유사도와 임베딩 검색

### 6.1 코사인 유사도

코사인 유사도는 두 벡터 사이의 *각도의 코사인* 입니다.

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \, |\mathbf{b}|}$$

값의 범위:
- $1$ : 같은 방향 (가장 유사)
- $0$ : 수직 (관계 없음)
- $-1$ : 반대 방향 (가장 다름)

NumPy 로:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([0.8, 0.1, -0.3, 0.5])
b = np.array([0.7, 0.15, -0.25, 0.45])
print(cosine_similarity(a, b))   # 0.998... (매우 유사)

c = np.array([-0.4, 0.7, 0.8, -0.1])
print(cosine_similarity(a, c))   # 음수 (전혀 다름)
```

### 6.2 정규화 후 내적 = 코사인 유사도

만약 두 벡터를 *미리 L2 정규화* 해 두면, 코사인 유사도는 *단순 내적* 과 같습니다.

```python
def normalize(v):
    return v / np.linalg.norm(v)

a_norm = normalize(a)
b_norm = normalize(b)

# 정규화된 벡터의 내적 = 코사인 유사도
print(np.dot(a_norm, b_norm))   # 0.998...
print(cosine_similarity(a, b))  # 0.998...  ← 동일
```

이 사실은 *대규모 검색 시스템* 의 효율을 크게 올립니다. 임베딩을 *저장 시점에 한 번만 정규화* 해 두면, 검색 시에는 *내적만* 계산하면 됩니다. 내적은 행렬 곱으로 *수백만 벡터* 를 한 번에 처리할 수 있어 GPU 에서 매우 빠릅니다.

### 6.3 임베딩 검색의 본질

RAG·시멘틱 검색·추천 시스템의 코어는 다음 한 줄입니다.

```python
# query_emb: (D,)
# corpus_embs: (N, D)  ← 미리 정규화됨

scores = corpus_embs @ query_emb       # (N,)
top_k = np.argsort(scores)[::-1][:k]   # 상위 k 개
```

이 한 줄이 *수백만 문서 중 가장 비슷한 K 개를 찾는* 일을 합니다.

실제 시스템은 *인덱스 (FAISS, HNSW)* 로 더 빠르게 만들지만, *수학적 본질* 은 위 두 줄입니다. Vol 57 (벡터 검색과 ANN) 에서 자세히 다룹니다.

### 6.4 작은 RAG 만들기

100 줄 안쪽으로 *완전한 RAG 검색* 을 직접 만들 수 있습니다 (모델 부분은 건너뜀).

```python
import numpy as np

# 가상의 문서 임베딩 (실제로는 Sentence-BERT 등으로 생성)
docs = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "GPUs accelerate matrix multiplication.",
    "Apples are a fruit.",
]
# 가상의 768 차원 임베딩 (실제로는 모델로 생성)
embeddings = np.random.randn(len(docs), 768)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 쿼리도 같은 방식으로 임베딩 (예시는 랜덤)
query = "Tell me about deep learning."
query_emb = np.random.randn(768)
query_emb = query_emb / np.linalg.norm(query_emb)

# 검색 — 정규화된 벡터의 내적이 코사인 유사도
scores = embeddings @ query_emb
top_k = np.argsort(scores)[::-1][:3]

print("Top 3 results:")
for idx in top_k:
    print(f"  score={scores[idx]:.3f}: {docs[idx]}")
```

이 패턴은 *프로덕션 RAG 시스템도 같은 골격* 을 가집니다. 다른 점은 *임베딩 모델이 더 좋고, 인덱스가 더 빠르고, 데이터가 더 많을* 뿐입니다.

### 6.5 챕터 정리

코사인 유사도는 *두 벡터 사이 각도의 코사인* 이며, *L2 정규화 후 내적* 과 같습니다. 임베딩 검색의 본질은 *정규화된 벡터의 내적 정렬* 이며, 이 한 줄이 RAG·시멘틱 검색·추천의 코어입니다. 다음 챕터에서는 행렬을 *기하학적 변환* 으로 보는 시각을 다집니다.

---

## 7. 선형 변환의 기하학

### 7.1 행렬은 변환

2장에서 *행렬은 함수* 라고 했습니다. 이번에는 그 함수가 *기하학적으로 무엇을 하는가* 를 봅니다.

2D 평면의 점에 행렬을 곱하면, 그 점이 *어딘가로 이동* 합니다. 평면 전체에 같은 행렬을 곱하면, 평면 전체가 *변형* 됩니다. 이를 *선형 변환 (Linear Transformation)* 이라 부릅니다.

### 7.2 회전

2D 회전 행렬:

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

```python
import numpy as np

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

R_90 = rotation_matrix(np.pi / 2)
v = np.array([1, 0])

result = R_90 @ v
print(result)   # [0. 1.]   ← (1,0) 이 90 도 회전해 (0,1)
```

### 7.3 스케일

대각 행렬은 *스케일링* 입니다.

```python
S = np.diag([2, 0.5])    # x 방향 2 배, y 방향 0.5 배
v = np.array([1, 1])
print(S @ v)              # [2.  0.5]
```

### 7.4 반사

특정 축 기준 반사:

```python
# x 축 기준 반사 (y 좌표 부호 반전)
F = np.array([[1, 0],
              [0, -1]])
v = np.array([3, 4])
print(F @ v)   # [3 -4]
```

### 7.5 전단 (Shear)

```python
# x 방향 전단
Sh = np.array([[1, 1],
               [0, 1]])
v = np.array([0, 1])
print(Sh @ v)   # [1 1]   ← 위쪽 점이 오른쪽으로 밀림
```

### 7.6 합성 변환

여러 변환을 *행렬 곱* 으로 합성할 수 있습니다.

```python
R = rotation_matrix(np.pi / 4)
S = np.diag([2, 1])

# 먼저 스케일, 그 다음 회전 (오른쪽부터 적용)
combined = R @ S
v = np.array([1, 0])
print(combined @ v)
```

*행렬 곱의 순서가 중요* 합니다. `R @ S` 와 `S @ R` 은 일반적으로 다른 결과를 줍니다. *행렬 곱은 교환법칙이 성립하지 않습니다*.

### 7.7 신경망의 한 층 = 선형 변환 + 비선형성

이 시각은 신경망의 한 층을 다시 보게 합니다.

```python
y = W @ x + b   # 선형 변환 (회전 + 스케일 + 이동)
y = relu(y)     # 비선형성 (음수를 잘라냄)
```

신경망이 깊어진다는 것은 *수많은 선형 변환을 비선형성으로 연결* 한다는 의미입니다. 각 층은 *입력 공간을 변형* 하며, 마지막 층은 *분류 가능한 형태로 공간을 펼친* 결과를 만듭니다.

### 7.8 챕터 정리

행렬은 *기하학적으로 회전·스케일·반사·전단의 변환* 이며, 여러 변환은 *행렬 곱으로 합성* 됩니다. 신경망의 한 층은 *선형 변환 + 비선형성* 이며, 깊은 신경망은 *수많은 변환의 연쇄* 입니다. 다음 챕터에서는 지금까지 다룬 모든 개념을 NumPy 코드로 한 번 더 정리합니다.

---

## 8. NumPy 로 직접 해 보기

### 8.1 종합 실습

이 챕터는 *지금까지 배운 모든 개념을 NumPy 로 직접 확인* 하는 실습입니다. 모든 코드를 자기 환경에서 실행해 보시기 바랍니다.

### 8.2 벡터 연산 종합

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print("덧셈:", a + b)
print("스칼라 곱:", 3 * a)
print("원소별 곱:", a * b)
print("내적:", a @ b)
print("L1 노름:", np.linalg.norm(a, ord=1))
print("L2 노름:", np.linalg.norm(a, ord=2))
print("L∞ 노름:", np.linalg.norm(a, ord=np.inf))

# 코사인 유사도
cos_sim = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("코사인 유사도:", cos_sim)

# 정규화
a_normalized = a / np.linalg.norm(a)
b_normalized = b / np.linalg.norm(b)
print("정규화 후 내적:", a_normalized @ b_normalized)  # 코사인 유사도와 동일
```

### 8.3 행렬 연산 종합

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("행렬 곱:", A @ B)
print("원소별 곱:", A * B)
print("전치:", A.T)
print("역행렬:", np.linalg.inv(A))
print("단위 행렬 검증:", A @ np.linalg.inv(A))   # I 에 가까움
print("행렬식:", np.linalg.det(A))
```

### 8.4 미니 RAG 시스템

```python
import numpy as np

# 5 개 문서, 각각 16 차원 임베딩
np.random.seed(42)
docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
embeddings = np.random.randn(5, 16)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 쿼리 임베딩
query_emb = np.random.randn(16)
query_emb = query_emb / np.linalg.norm(query_emb)

# 코사인 유사도 = 내적 (정규화되어 있으므로)
scores = embeddings @ query_emb
top_indices = np.argsort(scores)[::-1]

print("순위별 결과:")
for rank, idx in enumerate(top_indices):
    print(f"  {rank+1}. {docs[idx]} (score={scores[idx]:.3f})")
```

### 8.5 신경망 한 층 만들기

```python
import numpy as np

def linear_layer(x, W, b):
    return W @ x + b

def relu(x):
    return np.maximum(0, x)

# 입력: 3 차원 → 은닉: 5 차원 → 출력: 2 차원
np.random.seed(0)
W1 = np.random.randn(5, 3) * 0.1
b1 = np.zeros(5)
W2 = np.random.randn(2, 5) * 0.1
b2 = np.zeros(2)

x = np.array([0.5, -0.3, 0.8])

# 1 층
h = relu(linear_layer(x, W1, b1))
print("은닉 층:", h)

# 2 층
y = linear_layer(h, W2, b2)
print("출력:", y)
```

이 5 줄 안에 *모든 신경망의 빌딩블록* 이 들어 있습니다. 더 깊은 신경망은 같은 패턴을 반복할 뿐입니다.

### 8.6 회전 변환 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# 단위 정사각형의 네 꼭짓점
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# 45 도 회전 행렬
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

rotated = R @ square

plt.figure(figsize=(6, 6))
plt.plot(square[0], square[1], 'b-', label='원본')
plt.plot(rotated[0], rotated[1], 'r-', label='45도 회전')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title('선형 변환 시각화')
plt.show()
```

이 코드를 실행하면 정사각형이 *45 도 회전한 마름모 모양* 이 되는 것을 그래프로 확인할 수 있습니다.

### 8.7 챕터 정리

이 챕터의 모든 코드는 *지금까지 배운 개념을 직접 검증* 하는 실습이었습니다. 같은 코드를 자기 노트북에서 직접 실행하고 *값을 눈으로 확인* 하면 책으로 읽은 것보다 훨씬 오래 기억에 남습니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **벡터** 는 방향과 크기를 가진 양이자 순서 있는 숫자의 묶음. ML 에서는 임베딩·특성·그래디언트·가중치 모두가 벡터입니다.
- **행렬** 은 벡터를 다른 차원으로 보내는 함수입니다. 신경망의 한 층 = 행렬 곱 + 비선형성.
- **텐서** 는 n 차원 배열의 일반화이며, 데이터 종류별 표준 모양 규약 (BCHW, BLD 등) 을 의식해야 합니다.
- **내적** 은 *두 벡터의 방향 일치도* 를 측정하며, 행렬 곱 = 내적의 묶음입니다.
- **노름** 은 벡터 크기 측정 방법이며 L1·L2·L∞ 가 표준. 정칙화·그래디언트 클리핑·정규화에 사용.
- **코사인 유사도** = 정규화된 벡터의 내적. RAG·검색·추천의 코어.
- **선형 변환** 은 회전·스케일·반사·전단을 행렬 곱으로 표현. 신경망 한 층의 본질.
- **NumPy** 로 모든 개념을 직접 코드로 검증. 이 권의 코드는 모두 실행 가능.

가장 기억에 남겨야 할 한 줄은 **"두 임베딩의 내적 한 줄이 RAG 의 본질이며, 신경망의 한 층은 결국 행렬 곱이다."** 입니다.

다음 권은 [Volume 5 — 선형대수 2 — 행렬 분해·고유값·SVD](./volume_09_linear_algebra_2.md) 입니다. 거기서는 행렬을 *세 개의 단순한 행렬의 곱으로 분해* 함으로써 데이터의 본질적 구조를 추출하는 도구를 다룹니다.

---

## 자가점검 키워드

`벡터`, `행렬`, `텐서`, `행렬 곱`, `내적`, `코사인 유사도`, `노름`, `선형 변환`

## 자가점검 질문

다음 질문에 막힘없이 답할 수 있을 때 다음 권으로 넘어가십시오.

1. 벡터의 두 정의 (기하학적·대수학적) 를 적고, 둘이 동등한 이유를 설명하십시오.
2. 행렬 곱 `A @ B` 가 가능한 모양 조건과, 결과 모양은 어떻게 결정됩니까?
3. 신경망의 한 완전 연결 층을 NumPy 로 5 줄 이내로 구현하십시오.
4. 두 벡터의 내적이 *유사도* 가 되는 이유를 *코사인* 관점에서 설명하십시오.
5. L1·L2·L∞ 노름의 단위 원 모양을 그리고, 각 노름이 ML 에서 사용되는 사례를 적으십시오.
6. *임베딩 검색* 의 본질을 NumPy 두 줄로 구현하십시오.
7. 신경망이 깊어지면서 *공간을 어떻게 변형* 한다고 말할 수 있습니까?

## 다음 권

[Volume 5 — 선형대수 2 — 행렬 분해·고유값·SVD](./volume_09_linear_algebra_2.md)
