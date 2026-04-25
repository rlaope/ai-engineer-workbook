# Volume 21 — 회귀

> 이 권이 끝나면 회귀가 *입력에서 실수 출력을 예측하는 가장 기본적인 학습 문제* 임을 코드로 증명할 수 있게 됩니다.

## 목적

회귀는 ML 의 가장 기초적인 문제이며, 선형 회귀의 닫힌 해부터 정칙화·다항 회귀까지의 흐름이 *모델 복잡도와 일반화* 의 핵심 사고를 만듭니다. 이 권은 회귀의 표준 알고리즘과 함정을 다집니다.

## 선수 지식

- Volume 8, 11, 20 완료
- 외부 지식: NumPy 기본

## 학습 결과

1. 선형 회귀의 닫힌 해 (정규 방정식) 를 NumPy 로 구현할 수 있습니다.
2. 릿지·라쏘·엘라스틱넷의 차이를 설명할 수 있습니다.
3. 다항 회귀의 과적합 경향을 시각화할 수 있습니다.
4. 회귀 평가 지표 (MSE·MAE·R²) 의 차이를 알 수 있습니다.

---

## 1. 선형 회귀

### 1.1 정의

$y = X \beta + \epsilon$ 의 형태에서 $\beta$ 를 학습.

### 1.2 닫힌 해

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

```python
import numpy as np

# 데이터 생성
np.random.seed(0)
X = np.random.randn(100, 3)
true_beta = np.array([2, -1, 0.5])
y = X @ true_beta + np.random.randn(100) * 0.1

# 선형 회귀 (닫힌 해)
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_hat)   # 약 [2, -1, 0.5]
```

### 1.3 챕터 정리

선형 회귀는 *닫힌 해가 존재* 하는 드문 ML 문제이며, NumPy 한 줄로 풀립니다.

---

## 2. 정칙화 — 릿지·라쏘·엘라스틱넷

### 2.1 릿지 (L2)

$$\min \|y - X\beta\|^2 + \lambda \|\beta\|_2^2$$

가중치를 *작게 만들어* 과적합 방지. 닫힌 해:

$$\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$$

### 2.2 라쏘 (L1)

$$\min \|y - X\beta\|^2 + \lambda \|\beta\|_1$$

*일부 가중치를 정확히 0* 으로 만듦 → *특성 선택* 효과. 닫힌 해 없음, 좌표 강하법 등 사용.

### 2.3 엘라스틱넷

L1 + L2 결합. 라쏘의 *불안정성* 과 릿지의 *희소성 부족* 을 보완.

### 2.4 scikit-learn 사용

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=0.1).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
```

### 2.5 챕터 정리

정칙화는 *L1·L2·결합* 의 세 종류이며, 각자 *과적합 방지·특성 선택·균형* 의 다른 효과를 가집니다.

---

## 3. 다항 회귀와 과적합

### 3.1 다항 특성

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
```

차수가 높아질수록 *훈련 정확도는 향상* 하지만 *일반화는 악화* 됩니다.

### 3.2 검증 곡선

```python
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

train_scores, val_scores = validation_curve(
    Ridge(), X, y, param_name='alpha',
    param_range=np.logspace(-3, 3, 7), cv=5,
)
```

`alpha` (정칙화 강도) 가 작을 때는 *과적합*, 클 때는 *과소적합*.

### 3.3 챕터 정리

다항 회귀는 *모델 복잡도와 일반화* 의 트레이드오프를 가장 명확히 보여 주는 사례입니다.

---

## 4. 평가 지표

```
+-------+-------+--------------------+
| 지표  | 의미  | 강점/약점          |
+-------+-------+--------------------+
| MSE   | 제곱 평균 오차 | 큰 오차에 민감 |
| MAE   | 절대 평균 오차 | 이상치에 강건  |
| RMSE  | MSE 의 √ | 단위 일치        |
| R²    | 분산 설명력 | 0-1 범위         |
+-------+-------+--------------------+
```

선택 기준:

- *이상치 많음* → MAE
- *큰 오차에 강하게 페널티* → MSE
- *해석 쉬움* → RMSE 또는 R²

---

## 권 정리

- 선형 회귀 = 닫힌 해 존재
- 정칙화 (L1·L2·EN) = 과적합 제어
- 다항 회귀 = 복잡도-일반화 트레이드오프 시각화
- 평가 지표 = MSE/MAE/R² 도메인에 맞춰 선택

가장 기억에 남겨야 할 한 줄은 **"선형 회귀가 ML 의 시작점이며, 정칙화 사고는 모든 후속 모델로 이전된다."** 입니다.

다음 권은 [Volume 22 — 분류](./volume_22_classification.md) 입니다.

---

## 자가점검 키워드

`선형 회귀`, `정규 방정식`, `릿지`, `라쏘`, `엘라스틱넷`, `다항`, `MSE`, `R²`

## 자가점검 질문

1. 선형 회귀 닫힌 해 식을 NumPy 로 구현하십시오.
2. 릿지와 라쏘의 차이를 *가중치 분포* 관점에서 설명하십시오.
3. 다항 회귀에서 차수와 일반화의 관계를 그래프로 그리십시오.
4. MSE 와 MAE 의 적용 시점 차이를 적으십시오.

## 다음 권

[Volume 22 — 분류](./volume_22_classification.md)
