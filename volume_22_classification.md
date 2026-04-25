# Volume 22 — 분류

> 이 권이 끝나면 *왜 분류가 회귀와 다른 손실 함수를 쓰는가* 를 정보 이론적으로 설명할 수 있게 됩니다.

## 목적

분류는 회귀와 함께 ML 의 두 기본 문제이며, *이산 클래스를 출력* 한다는 차이가 *손실 함수·평가 지표·확률 해석* 모두를 다르게 만듭니다. 이 권은 로지스틱 회귀부터 의사결정나무·k-NN 까지 분류의 표준 알고리즘을 다룹니다.

## 선수 지식

- Volume 12, 21 완료
- 외부 지식: 확률·로그의 기본

## 학습 결과

1. 로지스틱 회귀의 시그모이드 + BCE 손실을 유도할 수 있습니다.
2. 다중 분류의 소프트맥스 + CCE 를 적용할 수 있습니다.
3. 의사결정나무의 분할 기준 (정보 이득·지니) 을 설명할 수 있습니다.
4. k-NN 의 강약과 적용 시점을 알 수 있습니다.
5. 분류 평가 지표 (정확도·F1·AUC) 의 차이를 알 수 있습니다.

---

## 1. 로지스틱 회귀

### 1.1 시그모이드

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

z 를 0-1 사이의 확률로 변환.

### 1.2 BCE 손실

$$L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

베르누이 분포의 음의 로그가능도. (Vol 11 의 MLE 유도)

### 1.3 NumPy 구현

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def bce(y, y_hat, eps=1e-12): return -(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps)).mean()

# 학습
np.random.seed(0)
X = np.random.randn(200, 3)
true_w = np.array([1, -1, 0.5])
y = (X @ true_w + np.random.randn(200)*0.3 > 0).astype(float)

w = np.zeros(3)
lr = 0.1
for _ in range(500):
    z = X @ w
    p = sigmoid(z)
    grad = X.T @ (p - y) / len(X)
    w -= lr * grad

print(w)   # 약 [1, -1, 0.5] 방향
```

### 1.4 챕터 정리

로지스틱 회귀는 *시그모이드 + BCE* 이며, 가장 단순한 분류 모델입니다.

---

## 2. 다중 분류 — 소프트맥스 + CCE

```python
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# K 클래스 분류
n, d, K = 200, 5, 3
X = np.random.randn(n, d)
y = np.random.randint(0, K, n)
W = np.random.randn(d, K) * 0.01

lr = 0.1
for _ in range(500):
    z = X @ W
    p = softmax(z)
    p[np.arange(n), y] -= 1
    grad = X.T @ p / n
    W -= lr * grad
```

소프트맥스 + CCE 는 *분류 모델의 표준 출력* 입니다.

---

## 3. 의사결정나무

### 3.1 분할 기준

각 분기에서 *정보 이득* 또는 *지니 불순도 감소* 가 최대인 특성·임계값 선택.

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X, y)
```

### 3.2 강점·약점

- 강점: *해석 가능*, *수치·범주 모두 처리*
- 약점: *과적합 경향*, *작은 데이터 변화에 민감*

→ 앙상블 (랜덤포레스트·XGBoost) 로 약점 보완 (Vol 23).

---

## 4. k-NN

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
```

*학습 없음* — 추론 시 *모든 학습 데이터와 거리 계산*. 큰 데이터에서는 느림. 작은 데이터·저차원에서 유용.

---

## 5. 평가 지표

```
+----------+------------------+------------------+
| 지표     | 정의             | 적용 시점        |
+----------+------------------+------------------+
| 정확도   | 정답 / 전체      | 균형 데이터      |
| 정밀도   | TP / (TP+FP)     | False Positive 비용 큼 |
| 재현율   | TP / (TP+FN)     | False Negative 비용 큼 |
| F1       | P·R 조화 평균    | 균형             |
| ROC-AUC  | ROC 곡선 아래 면적 | 임계 무관       |
| PR-AUC   | PR 곡선 면적      | 클래스 불균형    |
+----------+------------------+------------------+
```

---

## 권 정리

- 로지스틱 회귀 = 시그모이드 + BCE
- 다중 분류 = 소프트맥스 + CCE
- 의사결정나무 = 해석 가능, 과적합 경향
- k-NN = 학습 없음, 작은 데이터에 유용
- 평가 지표 = 데이터 균형·비용 구조에 맞춰 선택

가장 기억해야 할 한 줄: **"분류 손실 (BCE/CCE) 은 모두 베르누이/범주 분포의 MLE 에서 유도된다."**

다음 권: [Volume 23 — 앙상블](./volume_23_ensemble.md)

---

## 자가점검 키워드

`로지스틱 회귀`, `시그모이드`, `BCE`, `소프트맥스`, `CCE`, `의사결정나무`, `k-NN`, `F1/AUC`

## 자가점검 질문

1. 시그모이드 함수의 미분이 σ(1-σ) 임을 보이십시오.
2. BCE 가 *베르누이의 음의 로그가능도* 임을 유도하십시오.
3. 의사결정나무의 분할 기준 두 가지를 설명하십시오.
4. ROC-AUC 와 PR-AUC 의 적용 시점 차이를 적으십시오.

## 다음 권

[Volume 23 — 앙상블](./volume_23_ensemble.md)
