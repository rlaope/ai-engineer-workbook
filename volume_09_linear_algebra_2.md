# Volume 9 — 선형대수 2 — 행렬 분해·고유값·SVD

> 이 권이 끝나면 *데이터의 본질적 차원* 을 행렬 분해로 추출할 수 있게 됩니다.

## 목적

행렬을 *세 개의 단순한 행렬의 곱* 으로 분해하면, 데이터가 가진 본질적 방향과 그 중요도가 드러납니다. 이것이 PCA·임베딩·압축·추천 시스템의 기반입니다. 또한 LoRA·DoRA 같은 최신 모델 적응 기법도 *낮은 랭크 분해* 에 뿌리를 둡니다. 이 권은 고유값 분해와 SVD 의 직관·기하·코드 구현을 함께 다룹니다.

## 선수 지식

- Volume 8 완료
- 외부 지식: 고등학교 함수와 그래프

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 고유값과 고유벡터의 의미를 *변환에서 방향이 바뀌지 않는 축* 으로 설명할 수 있습니다.
2. 대각화 가능 행렬과 그렇지 않은 행렬의 차이를 구분할 수 있습니다.
3. SVD 가 *어떤 행렬에도 적용 가능한 일반화된 분해* 임을 이해합니다.
4. PCA 가 SVD 의 직접적 응용임을 보일 수 있습니다.
5. LoRA 가 *낮은 랭크 분해를 학습 가능 파라미터로 두는* 기법임을 설명할 수 있습니다.

---

## 이 권을 읽기 전에

행렬 분해는 *복잡해 보이는 데이터를 단순한 부품들의 조합으로 다시 적는 일* 입니다. 한 행렬을 분해하면, *어떤 방향이 중요하고 어떤 방향이 무시할 만한가* 가 자동으로 드러납니다.

이 권의 도구들은 ML 의 곳곳에 등장합니다. PCA 는 차원 축소·시각화에, SVD 는 추천 시스템·LSA 에, 낮은 랭크 분해는 LoRA·MoE·임베딩 압축에 쓰입니다. 한 번 깊이 익혀 두면 이후의 많은 권에서 *이미 아는 도구* 로 다시 만나게 됩니다.

수식이 늘어나지만, 모든 수식은 NumPy 코드로 검증합니다. 그림과 코드로 직관을 잡아 두면 수식이 *외울 대상* 이 아니라 *논리적 결과* 가 됩니다.

---

## 1. 고유값과 고유벡터의 정의

### 1.1 변환의 *주축*

행렬 A 가 어떤 벡터 $\mathbf{v}$ 에 작용했을 때, 결과가 *원래 벡터의 스칼라 배* 가 되는 경우가 있습니다.

$$A \mathbf{v} = \lambda \mathbf{v}$$

이때 $\mathbf{v}$ 를 *고유벡터 (eigenvector)*, $\lambda$ 를 *고유값 (eigenvalue)* 이라 부릅니다.

직관적으로, 고유벡터는 *변환 A 가 적용되어도 방향이 바뀌지 않는 특별한 방향* 입니다. 다른 모든 방향은 변환 후 회전·스케일이 함께 일어나지만, 고유벡터는 *오직 스케일만* 일어납니다.

### 1.2 NumPy 로 고유값 분해

```python
import numpy as np

A = np.array([[4, -2],
              [1,  1]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("고유값:", eigenvalues)
# 고유값: [3. 2.]

print("고유벡터 (열 단위):")
print(eigenvectors)
# [[0.89442719 0.70710678]
#  [0.4472136  0.70710678]]

# 검증: A @ v == λ * v
v0 = eigenvectors[:, 0]
print("A @ v0:", A @ v0)
print("λ0 * v0:", eigenvalues[0] * v0)
# 두 결과가 같음
```

`np.linalg.eig` 는 *각 고유벡터를 행렬의 열* 로 반환합니다.

### 1.3 의미

고유값·고유벡터의 의미를 한 문단으로 정리합니다.

- *고유값* 은 그 방향에서의 *스케일링 비율* 입니다. 양수면 같은 방향, 음수면 반대 방향, 절댓값이 1 이면 크기 보존.
- *고유벡터* 는 변환 A 의 *고유한 축* 입니다. 행렬 A 의 본질을 설명하는 *내부 좌표계* 라고 볼 수 있습니다.
- *큰 고유값에 대응하는 고유벡터* 가 *변환 A 의 가장 영향력 있는 방향* 입니다.

### 1.4 ML 에서의 등장

고유값·고유벡터가 등장하는 ML 영역:

- *PCA* — 데이터 공분산 행렬의 고유 분해
- *그래프 라플라시안* — 그래프 구조의 스펙트럴 분석 (GNN)
- *PageRank* — 전이 행렬의 가장 큰 고유벡터
- *Hessian 분석* — 학습 손실 풍경의 곡률
- *모드 붕괴 분석* — GAN 에서

### 1.5 챕터 정리

고유값·고유벡터는 *행렬이 변환할 때 방향이 바뀌지 않는 특별한 축과 그 스케일* 입니다. 고유 분해는 행렬의 *내부 좌표계* 를 드러내는 도구이며, ML 의 곳곳에 등장합니다. 다음 챕터에서는 이 분해를 *대각화* 라는 형식으로 정리합니다.

---

## 2. 대각화

### 2.1 대각화의 정의

n × n 행렬 A 가 *n 개의 선형 독립 고유벡터* 를 가질 때, A 는 *대각화 가능 (diagonalizable)* 합니다.

대각화 형태:

$$A = P D P^{-1}$$

여기서:
- $P$: 고유벡터들을 열로 가지는 행렬
- $D$: 고유값들을 대각선에 가지는 대각 행렬
- $P^{-1}$: $P$ 의 역행렬

### 2.2 NumPy 로 대각화

```python
import numpy as np

A = np.array([[4, -2],
              [1,  1]])

eigenvalues, eigenvectors = np.linalg.eig(A)

P = eigenvectors          # 고유벡터를 열로
D = np.diag(eigenvalues)  # 대각 행렬
P_inv = np.linalg.inv(P)

# A = P D P^-1 검증
A_reconstructed = P @ D @ P_inv
print(A_reconstructed)
# [[ 4. -2.]
#  [ 1.  1.]]
```

원래 A 가 복원됩니다.

### 2.3 대각화의 의미

대각화는 *복잡한 변환을 세 단계로 분해* 합니다.

1. $P^{-1}$: *고유 좌표계로 회전*
2. $D$: 각 축으로 *독립적으로 스케일*
3. $P$: *원래 좌표계로 다시 회전*

복잡한 행렬도 *고유 좌표계에서는 단순한 스케일링* 일 뿐이라는 의미입니다. 이는 *복잡함이 좌표계 선택의 문제일 뿐* 이라는 깊은 통찰을 줍니다.

### 2.4 모든 행렬이 대각화 가능한가?

답은 *아니오* 입니다. 다음 조건이 모두 만족돼야 합니다.

- 정사각 행렬일 것
- n 개의 *선형 독립* 고유벡터를 가질 것

대칭 행렬 ($A = A^T$) 은 *항상 대각화 가능* 합니다. 그 외 행렬은 경우에 따라 다릅니다.

대각화가 안 되는 행렬도 처리하기 위해 *조르당 표준형 (Jordan Normal Form)* 같은 더 일반적인 분해가 있지만, 실무에서는 거의 사용되지 않습니다. 대신 *모든 행렬에 적용 가능* 한 SVD 가 사용됩니다.

### 2.5 챕터 정리

대각화는 행렬을 *고유벡터 + 고유값 + 역행렬* 의 곱으로 표현하는 일이며, 복잡한 변환을 *단순한 스케일링* 으로 보는 시각을 줍니다. 그러나 *대각화가 항상 가능한 것은 아닙니다*. 다음 챕터에서는 *모든 행렬에 적용 가능* 한 SVD 를 봅니다.

---

## 3. SVD — 모든 행렬을 위한 분해

### 3.1 SVD 의 정의

**특이값 분해 (Singular Value Decomposition, SVD)** 는 *어떤 모양의 행렬* 도 다음 형태로 분해합니다.

$$A = U \Sigma V^T$$

m × n 행렬 A 에 대해:
- $U$: m × m 직교 행렬 (왼쪽 특이 벡터들의 모임)
- $\Sigma$: m × n 대각 행렬 (특이값들이 대각선)
- $V^T$: n × n 직교 행렬의 전치 (오른쪽 특이 벡터들)

특이값 (singular value) 은 *항상 0 이상* 이며, *내림차순* 으로 정렬됩니다.

### 3.2 NumPy 로 SVD

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])  # 4 x 3

U, S, Vt = np.linalg.svd(A)
print("U shape:", U.shape)    # (4, 4)
print("S shape:", S.shape)    # (3,)  ← 대각 원소만
print("Vt shape:", Vt.shape)  # (3, 3)
print("특이값:", S)
# 특이값: [25.46  1.29  0.  ]   ← 마지막이 거의 0
```

`np.linalg.svd` 는 효율을 위해 $\Sigma$ 의 *대각 원소만* 1 차원 배열로 반환합니다.

복원:

```python
m, n = A.shape
Sigma = np.zeros((m, n))
Sigma[:len(S), :len(S)] = np.diag(S)

A_reconstructed = U @ Sigma @ Vt
print(A_reconstructed)
# 원래 A 와 동일
```

### 3.3 SVD 의 기하학적 의미

SVD 는 *어떤 행렬도 회전 → 스케일 → 회전* 의 세 단계 변환으로 분해할 수 있다는 것을 보여 줍니다.

1. $V^T$: *입력 공간에서의 회전*
2. $\Sigma$: 각 축으로의 *독립적 스케일*
3. $U$: *출력 공간에서의 회전*

대각화와의 차이는, SVD 는 *입력 좌표계와 출력 좌표계가 다를 수 있다* 는 점입니다. 대각화는 *같은 좌표계* 를 가정하므로 정사각 행렬에만 적용되지만, SVD 는 *직사각 행렬* 에도 적용됩니다.

### 3.4 특이값의 해석

특이값 $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$ 은 *행렬이 각 방향으로 얼마나 늘리는가* 를 나타냅니다.

- 큰 특이값 → 그 방향이 *변환에서 중요한 방향*
- 작은 특이값 → 그 방향이 *거의 영향이 없는 방향*
- 특이값이 0 → 그 방향이 *변환에서 사라짐*

이 사실이 다음 챕터의 *저차원 근사* 의 기반입니다.

### 3.5 챕터 정리

SVD 는 *어떤 모양의 행렬에도 적용되는 일반화된 분해* 입니다. *회전 → 스케일 → 회전* 의 세 단계로 변환을 분해하며, 특이값의 크기가 각 방향의 중요도를 나타냅니다. 다음 챕터에서는 이 사실을 이용해 *행렬을 압축* 하는 저차원 근사를 봅니다.

---

## 4. 저차원 근사

### 4.1 큰 특이값만 남기기

SVD 의 가장 강력한 응용은 *큰 특이값만 남겨 행렬을 근사* 하는 것입니다.

```
A = σ₁ u₁ v₁ᵀ + σ₂ u₂ v₂ᵀ + ... + σ_r u_r v_rᵀ
```

각 항은 *랭크 1 행렬* 이며, 큰 $\sigma$ 가 큰 기여를 합니다. *상위 k 개의 항만 남기면* 원래 행렬에 가장 가까운 *랭크 k 근사* 를 얻습니다.

### 4.2 NumPy 로 랭크-k 근사

```python
import numpy as np

# 100 × 100 랜덤 행렬
np.random.seed(0)
A = np.random.randn(100, 100)

U, S, Vt = np.linalg.svd(A)

# 상위 5 개 특이값만으로 재구성
k = 5
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 원본과 근사의 차이
error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
print(f"k=5 근사 오차: {error:.4f}")
```

원래 100 × 100 = 10000 개의 숫자였던 행렬이 (100 × 5) + 5 + (5 × 100) = 1005 개의 숫자로 줄었습니다. 약 *10 배 압축* 됐습니다.

### 4.3 Eckart-Young 정리

랭크-k SVD 근사는 *모든 랭크 k 근사 중 최선* 임이 수학적으로 증명됩니다 (Eckart-Young 정리).

> 어떤 랭크 k 행렬 B 보다도, $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ 가 $\|A - B\|_F$ (프로베니우스 노름) 를 최소화한다.

이 사실 때문에 SVD 가 *데이터 압축의 황금 표준* 이 됩니다.

### 4.4 챕터 정리

SVD 의 *상위 k 개 특이값만 남기는 근사* 는 데이터 압축의 핵심입니다. 이 근사는 *Eckart-Young 정리* 에 의해 *최적* 임이 보장됩니다. 다음 챕터에서는 이 사실의 가장 유명한 응용인 PCA 를 봅니다.

---

## 5. PCA — SVD 로 보는 차원 축소

### 5.1 PCA 의 동기

데이터가 1000 차원이라도, 그 중 *진짜로 변하는 방향* 은 훨씬 적을 수 있습니다. 사람의 키·몸무게·신발 사이즈는 강하게 상관되어 있어, 셋이 *거의 같은 방향* 으로 변동합니다. 즉 *3 차원이지만 본질적으로 1 차원에 가까움*.

PCA (주성분 분석) 는 *데이터의 가장 큰 분산을 가진 방향들을 찾아* 그 방향만 남기는 차원 축소 기법입니다.

### 5.2 PCA 알고리즘

데이터 X (n 샘플 × d 특성) 에 PCA 를 적용:

1. 중심화: $X_c = X - \bar{X}$
2. 공분산 행렬: $C = \frac{1}{n} X_c^T X_c$
3. C 의 고유 분해: $C = V \Lambda V^T$
4. 큰 고유값에 대응하는 고유벡터 k 개를 선택: $V_k$
5. 변환: $X' = X_c V_k$

또는 SVD 로 직접:

1. 중심화: $X_c = X - \bar{X}$
2. SVD: $X_c = U \Sigma V^T$
3. 상위 k 개 V 의 열을 사용: $X' = X_c V_k$

두 방법은 수학적으로 동등합니다.

### 5.3 NumPy 로 PCA

```python
import numpy as np
import matplotlib.pyplot as plt

# 100 개 샘플, 3 차원 데이터 (실제로는 거의 평면 위에 놓임)
np.random.seed(0)
X_2d = np.random.randn(100, 2) * np.array([3, 1])
# 3 번째 차원에 작은 노이즈만
X = np.hstack([X_2d, np.random.randn(100, 1) * 0.1])

# 중심화
X_centered = X - X.mean(axis=0)

# SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# 상위 2 개 주성분으로 변환
X_pca = X_centered @ Vt[:2].T  # (100, 2)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 결과')
plt.axis('equal')
plt.grid(True)
plt.show()

# 분산 설명력
explained_variance_ratio = S**2 / (S**2).sum()
print("각 주성분의 분산 설명력:", explained_variance_ratio)
```

`explained_variance_ratio` 가 `[0.85, 0.13, 0.02]` 처럼 나오면, *상위 2 개 주성분이 분산의 98% 를 설명* 한다는 의미입니다. 따라서 3 차원을 2 차원으로 줄여도 정보 손실이 적습니다.

### 5.4 scikit-learn 의 PCA

scikit-learn 은 PCA 를 한 줄로 제공합니다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("분산 설명력:", pca.explained_variance_ratio_)
print("주성분:", pca.components_)
```

내부 구현은 SVD 입니다.

### 5.5 PCA 의 ML 응용

- *데이터 시각화* — 1000 차원 임베딩을 2D 로 그림
- *전처리* — 차원 축소로 학습 속도·메모리 절감
- *노이즈 제거* — 작은 특이값에 대응하는 방향이 노이즈
- *압축* — 이미지·신호 압축
- *이상 탐지* — 주성분 공간에서 멀리 떨어진 점이 이상치

### 5.6 챕터 정리

PCA 는 *데이터의 분산을 최대화하는 방향들을 찾는* 차원 축소 기법이며, SVD 의 직접적 응용입니다. 시각화·전처리·노이즈 제거·이상 탐지에 광범위하게 사용됩니다. 다음 챕터에서는 SVD 의 또 다른 응용인 *이미지 압축* 을 봅니다.

---

## 6. 응용 1 — 이미지 압축

### 6.1 흑백 이미지를 행렬로 보기

흑백 이미지는 *2 차원 행렬* 입니다. 각 원소가 0-255 의 픽셀 값.

512 × 512 이미지는 262144 개의 숫자를 가집니다. 이 행렬을 SVD 로 저차원 근사하면 *이미지를 압축* 할 수 있습니다.

### 6.2 NumPy 로 이미지 압축

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 흑백 이미지 로드 (자기 이미지로 교체 가능)
# img = Image.open('photo.jpg').convert('L')
# img = np.array(img)

# 데모용 가상 이미지
img = np.random.randn(512, 512)
img[100:300, 100:300] = 1.0  # 흰 사각형
img[200:400, 200:400] = 0.5

# SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 다양한 k 로 근사
ks = [10, 30, 100, 300]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('원본 (512)')

for i, k in enumerate(ks):
    img_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    axes[i+1].imshow(img_approx, cmap='gray')
    axes[i+1].set_title(f'k={k}')

plt.show()

# 압축률 계산
original_size = 512 * 512
for k in ks:
    compressed_size = 512 * k + k + k * 512
    print(f"k={k}: {compressed_size}/{original_size} = {compressed_size/original_size:.2%}")
```

k = 30 정도면 사람 눈에 거의 차이가 없으면서 *원본의 12%* 크기로 압축됩니다.

### 6.3 JPEG 와의 관계

JPEG 압축 자체는 SVD 가 아니라 *DCT (Discrete Cosine Transform)* 을 사용하지만, 발상은 비슷합니다. *주파수 영역에서 작은 계수를 잘라내* 압축합니다.

SVD 는 *데이터에 적응적으로* 분해하므로 JPEG 보다 *이론적으로 더 좋은 압축* 이 가능하지만, *계산 비용이 크고 표준화가 안 되어* 실제 이미지 포맷으로는 사용되지 않습니다.

### 6.4 챕터 정리

SVD 로 이미지를 *데이터에 적응적으로 압축* 할 수 있으며, 작은 k 에서도 시각적으로 받아들일 만한 품질을 얻습니다. 실용적인 이미지 압축 (JPEG) 은 다른 변환을 쓰지만 발상은 같습니다. 다음 챕터에서는 또 다른 SVD 응용인 추천 시스템을 봅니다.

---

## 7. 응용 2 — 추천 시스템

### 7.1 사용자-아이템 행렬

추천 시스템의 가장 단순한 형태는 *사용자 × 아이템 평점 행렬* 입니다.

```
        영화1  영화2  영화3  영화4
사용자1  5     ?     3     ?
사용자2  ?     4     ?     5
사용자3  4     ?     ?     2
사용자4  ?     5     4     ?
```

대부분의 칸이 비어 있습니다 (희소). *비어 있는 칸을 예측* 하는 것이 추천의 본질입니다.

### 7.2 행렬 분해로 추천

핵심 가정: 평점 행렬은 *사용자 잠재 요인 + 아이템 잠재 요인* 의 곱으로 표현됩니다.

$$R \approx U V^T$$

- $U$: (사용자 수 × k) — 각 사용자의 잠재 요인
- $V$: (아이템 수 × k) — 각 아이템의 잠재 요인

R 의 칸이 비어 있어도 U, V 를 학습하면 *모든 칸을 예측* 할 수 있습니다.

### 7.3 SVD 기반 추천 (단순화 버전)

```python
import numpy as np

# 평점 행렬 (0 = 미평가)
R = np.array([
    [5, 0, 3, 0],
    [0, 4, 0, 5],
    [4, 0, 0, 2],
    [0, 5, 4, 0],
], dtype=float)

# 결측치를 평균으로 채움 (간단한 방법)
mask = R > 0
R_filled = R.copy()
R_filled[~mask] = R[mask].mean()

# SVD
U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)

# 랭크 2 근사
k = 2
R_pred = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
print("예측 행렬:")
print(np.round(R_pred, 2))
```

이 단순화 버전은 *프로덕션 수준이 아니지만* 핵심 발상을 보여 줍니다. 실제 추천 시스템은 *Funk SVD·SVD++·ALS* 같은 변형을 사용해 *결측치를 처리* 하면서 학습합니다.

### 7.4 잠재 요인의 해석

학습된 U, V 의 각 차원은 *학습 데이터에서 발견된 추상적 특성* 입니다. 예: 영화 추천에서 한 차원이 *액션-드라마 축*, 다른 차원이 *유머-진지함 축* 같은 의미를 가질 수 있습니다.

이 *잠재 요인* 은 *임베딩* 의 한 형태이며, 이후 등장하는 모든 임베딩 기반 시스템 (Word2Vec, Sentence-BERT, RAG) 의 직접적 조상입니다.

### 7.5 챕터 정리

추천 시스템의 핵심은 *사용자 × 아이템 행렬을 잠재 요인 행렬의 곱으로 분해* 하는 일이며, SVD 가 그 첫 도구입니다. 잠재 요인은 *임베딩의 원형* 이며, 현대 모든 임베딩 시스템의 발상의 뿌리입니다. 다음 챕터에서는 같은 *낮은 랭크 분해* 발상이 *현대 LLM 의 미세조정* 에 어떻게 쓰이는지를 봅니다.

---

## 8. 응용 3 — LoRA

### 8.1 LoRA 의 동기

7B LLM 을 미세조정하려면 7B 파라미터를 모두 학습해야 합니다. 메모리·계산이 막대하며, 미세조정한 결과를 *저장하기도* 7B × 2 bytes = 14 GB 가 필요합니다.

**LoRA (Low-Rank Adaptation)** 의 핵심 발상: *미세조정 시 가중치 변화량 ΔW 가 낮은 랭크* 라고 가정하고, ΔW 를 *두 작은 행렬의 곱* 으로 표현합니다.

$$\Delta W = A B$$

- $A$: (d × r), $B$: (r × d), 여기서 r 은 매우 작음 (예: 8, 16)

전체 W 가 (d × d) 라면, ΔW 도 (d × d) 지만, A·B 만 학습하면 됩니다. 학습할 파라미터 수는 (d × r) + (r × d) = 2dr 로 줄어듭니다. d = 4096, r = 8 이면 *원래의 0.4%* 만 학습합니다.

### 8.2 LoRA 의 수식

미세조정 후 가중치는:

$$W' = W + \Delta W = W + AB$$

추론 시:

$$y = W'x = (W + AB)x = Wx + ABx$$

원래 가중치 $W$ 는 *동결* 되고, A·B 만 학습됩니다.

### 8.3 NumPy 로 LoRA 의 핵심

```python
import numpy as np

# 사전학습된 가중치 (동결)
np.random.seed(0)
d = 4096
W = np.random.randn(d, d) * 0.01

# LoRA 어댑터 (학습 대상)
r = 8
A = np.random.randn(d, r) * 0.01
B = np.zeros((r, d))   # 보통 B 는 0 으로 초기화

# 입력
x = np.random.randn(d)

# 추론
delta_W_x = A @ (B @ x)   # 효율: (rxd)(dxd 배치) 가 아닌 (dxr)(rxd)(d) 순서
y = W @ x + delta_W_x
print(y.shape)             # (4096,)

# 학습 파라미터 수 비교
full_params = d * d
lora_params = d * r + r * d
print(f"Full FT: {full_params:,} ({full_params*2/1024**3:.2f} GB)")
print(f"LoRA:    {lora_params:,} ({lora_params*2/1024**3:.4f} GB)")
print(f"비율: {lora_params/full_params:.4%}")
```

출력:
```
Full FT: 16,777,216 (0.03 GB)
LoRA:    65,536 (0.0001 GB)
비율: 0.3906%
```

원래의 *0.4%* 만 학습합니다.

### 8.4 LoRA 의 실무 함의

LoRA 가 만든 변화:

- *24 GB 단일 GPU* 로 7B 모델 미세조정 가능
- *어댑터 파일* 만 저장 (수 MB) → 같은 베이스 모델에 *수백 개 어댑터* 보관
- *어댑터 합치기·교체* — 운영 시 사용자별 어댑터를 동적으로 적용
- *추론 시 ΔW = AB 를 W 에 합쳐 (W' = W + AB)* 두면 추가 비용 없음

PEFT (Vol 98) 의 핵심이며, 산업 현장의 거의 모든 LLM 미세조정에 사용됩니다.

### 8.5 챕터 정리

LoRA 는 *미세조정 시 가중치 변화가 낮은 랭크* 라는 가정으로 학습 파라미터를 *원래의 1% 미만* 으로 줄이는 기법입니다. 발상의 뿌리는 *낮은 랭크 분해* 이며, 이 권에서 다룬 SVD·PCA 와 같은 가족입니다. 다음 챕터에서는 모든 도구를 NumPy 로 종합합니다.

---

## 9. NumPy 로 직접 해 보기

### 9.1 종합 실습

이 권의 모든 개념을 코드로 검증합니다.

### 9.2 고유값 분해 검증

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("고유값:", eigenvalues)

# 검증
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    print(f"A @ v_{i} = {A @ v}")
    print(f"λ_{i} * v_{i} = {eigenvalues[i] * v}")
    print()
```

### 9.3 SVD 와 저차원 근사

```python
np.random.seed(0)
A = np.random.randn(100, 100)

U, S, Vt = np.linalg.svd(A)

# 다양한 k 로 근사 오차 측정
for k in [5, 20, 50, 100]:
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    error = np.linalg.norm(A - A_k) / np.linalg.norm(A)
    print(f"k={k}: 상대 오차 {error:.4f}")
```

### 9.4 PCA 직접 구현

```python
def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T

np.random.seed(0)
X = np.random.randn(200, 10) @ np.random.randn(10, 5)  # 본질적으로 5 차원

X_2d = pca(X, 2)
print("축소 후 모양:", X_2d.shape)
```

### 9.5 미니 LoRA 학습 시뮬레이션

```python
import numpy as np

np.random.seed(42)
d = 64    # 작은 차원으로
r = 4     # LoRA 랭크

W_pretrained = np.random.randn(d, d) * 0.01

# 가짜 학습 데이터: y = W_target @ x
W_target_delta = np.random.randn(d, r) @ np.random.randn(r, d)  # 랭크 r 의 ΔW
W_target = W_pretrained + W_target_delta

X = np.random.randn(100, d)
Y = X @ W_target.T

# LoRA 학습
A = np.random.randn(d, r) * 0.01
B = np.zeros((r, d))
lr = 0.01

for step in range(500):
    Y_pred = X @ (W_pretrained + A @ B).T
    diff = Y_pred - Y
    loss = (diff ** 2).mean()
    
    # 그래디언트 계산 (매뉴얼)
    # dL/dB = (1/n) * (A.T @ diff.T @ X)
    # dL/dA = (1/n) * (diff.T @ X @ B.T)
    n = X.shape[0]
    dB = (A.T @ diff.T @ X) / n
    dA = (diff.T @ X @ B.T) / n
    
    A -= lr * dA
    B -= lr * dB
    
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss:.6f}")
```

이 시뮬레이션은 *PyTorch 없이도 LoRA 의 핵심을 손으로 구현할 수 있다* 는 것을 보여 줍니다.

### 9.6 챕터 정리

이 권의 모든 도구 — 고유 분해·대각화·SVD·PCA·LoRA — 를 NumPy 로 직접 검증해 봤습니다. *손으로 구현* 한 경험이 *라이브러리 호출* 만 한 경험보다 훨씬 오래 갑니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **고유값·고유벡터** 는 변환에서 *방향이 바뀌지 않는 특별한 축과 그 스케일* 입니다.
- **대각화** 는 행렬을 *단순한 좌표계 + 스케일링* 으로 보는 시각이지만, 모든 행렬에 가능하지 않습니다.
- **SVD** 는 *어떤 모양의 행렬에도 적용되는 일반화된 분해* 이며, 회전 → 스케일 → 회전의 세 단계로 분해합니다.
- **저차원 근사** 는 SVD 의 상위 k 개 특이값만 남기는 것이며, *Eckart-Young 정리* 에 의해 최적입니다.
- **PCA** 는 *데이터의 분산이 큰 방향들을 찾는* 차원 축소 기법이며, SVD 의 직접 응용입니다.
- **이미지 압축** 은 SVD 로 데이터에 적응적인 압축이 가능함을 보여 줍니다.
- **추천 시스템** 의 *행렬 분해* 가 임베딩의 원형이 됩니다.
- **LoRA** 는 *낮은 랭크 분해* 를 학습 가능 파라미터로 두는 미세조정 기법으로, 산업 표준이 되었습니다.

가장 기억에 남겨야 할 한 줄은 **"행렬 분해는 데이터의 본질적 구조를 드러내는 도구이며, PCA·추천·LoRA·임베딩 모두 같은 가족이다."** 입니다.

다음 권은 [Volume 10 — 미적분과 그래디언트](./volume_10_calculus_gradient.md) 입니다. 거기서는 *학습이 본질적으로 그래디언트를 따라 내려가는 일* 임을 배웁니다.

---

## 자가점검 키워드

`고유값`, `고유벡터`, `대각화`, `SVD`, `PCA`, `저차원 근사`, `LoRA`, `행렬 인수분해`

## 자가점검 질문

다음 질문에 막힘없이 답할 수 있을 때 다음 권으로 넘어가십시오.

1. *고유벡터* 가 *변환에서 방향이 바뀌지 않는 축* 이라는 말의 의미를 NumPy 코드로 검증하십시오.
2. 모든 행렬이 대각화 가능하지 않은 이유와, SVD 가 그 한계를 어떻게 우회하는지 설명하십시오.
3. SVD 의 *기하학적 3 단계 (회전 → 스케일 → 회전)* 를 그림으로 그리십시오.
4. PCA 와 SVD 의 관계를 한 문단으로 설명하십시오.
5. 100 × 100 행렬을 SVD 로 *랭크 5 근사* 했을 때 저장 공간이 얼마나 줄어드는지 계산하십시오.
6. LoRA 가 *학습 파라미터 수를 1% 미만으로 줄이는* 이유를 d=4096, r=8 예시로 설명하십시오.
7. PCA·추천 시스템·LoRA 가 같은 가족이라고 말하는 이유는 무엇입니까?

## 다음 권

[Volume 10 — 미적분과 그래디언트](./volume_10_calculus_gradient.md)
