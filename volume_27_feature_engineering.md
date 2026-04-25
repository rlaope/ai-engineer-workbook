# Volume 27 — 특성 공학

> 이 권이 끝나면 정형 데이터를 받았을 때 *모델에 넣기 전에 무엇을 해야 하는가* 의 표준 체크리스트가 머릿속에 그려지게 됩니다.

## 목적

특성 공학 (Feature Engineering) 은 *원시 데이터를 모델이 잘 학습할 수 있는 형태로 변환* 하는 작업입니다. 정형 데이터에서는 *모델 알고리즘 선택보다 특성 공학이 결과에 더 큰 영향* 을 주는 경우가 많습니다. 이 권은 인코딩·스케일링·결측·이상치·파생 특성의 표준 기법을 다집니다.

## 선수 지식

- Volume 21, 22 완료
- 외부 지식: pandas DataFrame 기본

## 학습 결과

1. 범주형·수치형·시간 특성의 표준 처리 기법을 적용할 수 있습니다.
2. 결측치 처리 4 가지 전략을 비교할 수 있습니다.
3. 이상치 탐지·처리의 함정을 안다.
4. 파생 특성 (Feature Crosses, Polynomial) 의 효과를 알 수 있습니다.
5. *데이터 누수* 를 방지하는 표준 절차를 적용할 수 있습니다.

---

## 1. 범주형 특성 인코딩

### 1.1 One-Hot Encoding

```python
import pandas as pd
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})
encoded = pd.get_dummies(df['color'])
```

장점: 단순. 단점: *고차원 (카테고리 수가 큰 경우)*.

### 1.2 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded = le.fit_transform(['red', 'green', 'blue'])
# → [2, 1, 0]
```

장점: 저차원. 단점: *순서 의미가 잘못 도입* — 트리 모델에만 적합.

### 1.3 Target Encoding

각 카테고리를 *그 카테고리의 타겟 평균* 으로 인코딩.

장점: 고카디널리티 (수천 카테고리) 처리. 단점: *데이터 누수 위험* — 학습 셋의 타겟 평균을 사용하면 누수.

### 1.4 Embedding (학습 가능)

신경망에서 사용. 카테고리를 *학습 가능한 벡터* 로 표현. 예: 사용자 ID → 32 차원 임베딩.

---

## 2. 수치형 특성 처리

### 2.1 스케일링

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()   # 평균 0, 분산 1
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

스케일링이 필요한 모델: *KNN, SVM, Logistic Regression, 신경망*.
스케일링 무관: *트리 기반 (Random Forest, XGBoost)*.

### 2.2 변환

- *Log* — 양의 왜도 큰 데이터
- *Box-Cox / Yeo-Johnson* — 일반화된 정규화
- *Binning* — 연속 값을 구간으로

---

## 3. 결측치 처리

```
+----------+--------------------------+--------------------+
| 전략     | 방법                     | 적용 시점          |
+----------+--------------------------+--------------------+
| 삭제     | 결측 행/열 제거          | 결측이 적을 때     |
| 평균/중앙값 | 단순 통계로 채움       | 빠른 처리          |
| 모델 예측 | 다른 특성으로 예측       | 정확성 우선        |
| 새 값    | 특수값 또는 결측 표시     | 결측 자체가 의미   |
+----------+--------------------------+--------------------+
```

```python
from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy='median').fit(X_train)
X_filled = imputer.transform(X_train)
```

---

## 4. 이상치 처리

탐지: IQR, Z-score, Isolation Forest.
처리:
- *제거* — 명백한 측정 오류
- *Capping* — 임계값으로 잘라내기 (Winsorization)
- *Transformation* — log 등으로 영향 줄이기
- *유지* — 의미 있는 신호일 수 있음

---

## 5. 파생 특성

```python
df['ratio'] = df['a'] / df['b']
df['interaction'] = df['a'] * df['b']
df['diff'] = df['a'] - df['b']
df['polynomial'] = df['a'] ** 2
```

도메인 지식이 가장 큰 효과를 만듭니다. 시간 데이터의 경우:

```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

---

## 6. 데이터 누수 방지

### 6.1 흔한 누수 패턴

- *전체 데이터로 fit, 학습/테스트 모두 transform* — 테스트 통계가 학습에 영향
- *Target Encoding 학습 셋 사용* — 타겟이 특성으로 누수
- *시간 데이터 무작위 분할* — 미래 정보로 과거 예측

### 6.2 방어

- *Pipeline* 사용 — 학습/테스트에 동일한 변환 자동 적용
- *fit 은 학습 셋에만, transform 은 양쪽에*
- *시간 데이터는 시간 순으로 분할*

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression()),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

---

## 권 정리

- 범주형: One-Hot·Label·Target·Embedding
- 수치형: 스케일링·변환·Binning
- 결측: 삭제·통계·모델·특수값
- 이상치: 제거·Capping·변환·유지
- 파생: 비율·상호작용·시간 분해
- 데이터 누수: Pipeline 으로 방지

가장 기억할 한 줄: **"정형 데이터에서는 모델 선택보다 특성 공학이 결과를 더 크게 바꾼다."**

다음 권: [Volume 28 — 베이지안 머신러닝](./volume_28_bayesian_ml.md)

---

## 자가점검 키워드

`One-Hot`, `Target Encoding`, `StandardScaler`, `결측 4 전략`, `IQR/Z-score`, `Pipeline`, `데이터 누수`

## 자가점검 질문

1. 범주형 인코딩 4 가지의 적용 시점을 비교하십시오.
2. 스케일링이 필요한 모델과 무관한 모델을 분류하십시오.
3. Target Encoding 의 데이터 누수 위험을 설명하십시오.
4. 시간 데이터 분할 시 *왜 무작위 분할이 안 되는가* 설명하십시오.

## 다음 권

[Volume 28 — 베이지안 머신러닝](./volume_28_bayesian_ml.md)
