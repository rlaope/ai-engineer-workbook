# Volume 25 — 비지도학습

> 이 권이 끝나면 *정답이 없는 데이터에서 구조를 발견하는 일* 이 어떻게 가능한지를 설명할 수 있게 됩니다.

## 목적

현실 데이터의 대부분은 *정답이 붙어 있지 않습니다*. 이런 데이터에서 *패턴·구조·이상치* 를 발견하는 능력은 ML 엔지니어의 핵심 역량입니다. 군집화·밀도 추정·이상 탐지가 비지도학습의 세 갈래이며, 각 갈래마다 표준 알고리즘이 있습니다.

## 선수 지식

- Volume 9, 20 완료
- 외부 지식: 거리의 직관

## 학습 결과

1. K-Means 알고리즘을 손으로 한 스텝씩 굴릴 수 있습니다.
2. K 값을 선택하는 방법(엘보우·실루엣)을 적용할 수 있습니다.
3. K-Means 의 한계를 보일 수 있습니다.
4. GMM·DBSCAN·계층적 군집의 차이를 알 수 있습니다.
5. 이상 탐지의 기본 도구를 적용할 수 있습니다.

---

## 1. K-Means

### 1.1 알고리즘

1. K 개의 중심점 무작위 초기화
2. 각 점을 가장 가까운 중심에 할당
3. 각 군집의 평균으로 중심 갱신
4. 변화 없을 때까지 2-3 반복

### 1.2 NumPy 구현

```python
import numpy as np

def kmeans(X, K, max_iter=100):
    centers = X[np.random.choice(len(X), K, replace=False)]
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers, axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return labels, centers

X = np.random.randn(200, 2)
labels, centers = kmeans(X, K=3)
```

### 1.3 K 선택

- *엘보우* — 군집 수에 따른 *총 inertia* 곡선의 굽은 점
- *실루엣* — 군집 응집도와 분리도의 균형 측정

### 1.4 한계

- *구형 군집* 가정 — 길쭉한 군집 못 잡음
- *동일 크기* 가정
- *K 사전 지정* 필요
- *초기화 민감* — k-means++ 로 완화

---

## 2. GMM (Gaussian Mixture Model)

K-Means 의 *확률적 일반화*. 각 점이 K 개 가우시안 중 하나에서 *확률적으로* 생성. EM 알고리즘으로 학습.

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3).fit(X)
probs = gmm.predict_proba(X)   # 각 군집 소속 확률
```

장점: *구형이 아닌 군집* 처리, *확률적 소속*.

---

## 3. DBSCAN

*밀도 기반 군집*. 밀도가 높은 영역을 군집으로, 낮은 영역의 점은 *이상치* 로 분류.

장점: *임의 모양의 군집* 처리, *K 사전 지정 불필요*, *이상치 자동 탐지*.

```python
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
# clusters == -1 인 점이 이상치
```

---

## 4. 계층적 군집

*트리 형태로 군집을 점진적으로 합치거나 분리*. 결과가 *덴드로그램 (dendrogram)* 으로 시각화.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
```

장점: *계층 구조* 가 의미 있는 도메인 (생물 분류 등) 에 유용.

---

## 5. 이상 탐지

- **Isolation Forest** — 무작위 분할로 *고립* 되기 쉬운 점이 이상치
- **One-Class SVM** — 정상 영역을 SVM 으로 포위
- **Local Outlier Factor** — 이웃 밀도 비교

```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05).fit(X)
preds = clf.predict(X)   # -1 이면 이상치
```

---

## 권 정리

- K-Means = 단순·빠름·구형 가정
- GMM = 확률적·임의 모양
- DBSCAN = 밀도 기반·이상치 자동
- 계층적 = 덴드로그램·해석 가능
- 이상 탐지 = Isolation Forest·One-Class SVM·LOF

가장 기억할 한 줄: **"군집화 알고리즘 선택은 데이터의 모양 (구형·길쭉함·임의 모양) 과 K 사전 지정 가능성으로 결정된다."**

다음 권: [Volume 26 — 차원 축소와 시각화](./volume_26_dimensionality_reduction.md)

---

## 자가점검 키워드

`K-Means`, `엘보우`, `실루엣`, `GMM`, `DBSCAN`, `계층적 군집`, `Isolation Forest`

## 자가점검 질문

1. K-Means 알고리즘의 4 단계를 적으십시오.
2. K-Means 의 4 가지 한계를 나열하십시오.
3. DBSCAN 이 K-Means 보다 유리한 경우를 적으십시오.
4. 이상 탐지 알고리즘 3 가지를 비교하십시오.

## 다음 권

[Volume 26 — 차원 축소와 시각화](./volume_26_dimensionality_reduction.md)
