# Volume 23 — 앙상블

> 이 권이 끝나면 *여러 약한 학습기를 모으면 강한 학습기가 된다* 는 통찰을 코드로 증명할 수 있게 됩니다.

## 목적

앙상블 (Ensemble) 은 *여러 모델의 예측을 결합* 해 단일 모델보다 좋은 결과를 만드는 기법입니다. 정형 데이터 (테이블) 에서는 여전히 *XGBoost·LightGBM 같은 부스팅 앙상블이 SOTA* 인 경우가 많으며, Kaggle 경진대회 우승의 표준 도구입니다. 이 권은 배깅·부스팅·스태킹의 차이와 표준 도구를 다집니다.

## 선수 지식

- Volume 22 완료
- 외부 지식: 의사결정나무 직관

## 학습 결과

1. 배깅·부스팅·스태킹의 차이를 설명할 수 있습니다.
2. 랜덤포레스트의 *어떻게 분산을 줄이는가* 메커니즘을 알 수 있습니다.
3. AdaBoost·GBM·XGBoost·LightGBM 의 차이를 한 줄씩 설명할 수 있습니다.
4. 정형 데이터에서 부스팅이 여전히 강한 이유를 설명할 수 있습니다.

---

## 1. 배깅 (Bagging)

### 1.1 정의

*Bootstrap Aggregating*. 데이터를 *부트스트랩 (복원 추출)* 으로 여러 셋을 만들고, 각 셋에 *독립적으로 모델 학습* 후 예측 평균.

### 1.2 랜덤포레스트

배깅 + *각 트리에서 무작위 특성 부분집합* 사용. 트리 사이의 *상관성* 을 줄여 분산 감소 효과 극대화.

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X, y)
```

### 1.3 챕터 정리

배깅은 *분산 감소* 를 목표로 하며, 랜덤포레스트가 표준 구현입니다.

---

## 2. 부스팅 (Boosting)

### 2.1 정의

*순차적으로* 약한 학습기를 학습하되, *이전 학습기가 못 푼 샘플에 더 큰 가중치* 부여.

- 편향 감소 (배깅과 반대)
- 학습기 사이 *순차 의존성* 으로 병렬화 어려움

### 2.2 알고리즘 진화

- **AdaBoost** (1995) — 첫 실용적 부스팅. 잘못 분류된 샘플 가중치 증가.
- **GBM (Gradient Boosting Machine)** (1999) — 손실 함수의 *그래디언트 방향* 으로 약학습기 추가.
- **XGBoost** (2014) — GBM 의 효율 + 정칙화 + 병렬화 최적화. Kaggle 의 표준.
- **LightGBM** (2017) — Microsoft. 히스토그램 기반·잎 단위 분할로 더 빠름.
- **CatBoost** (2017) — Yandex. 범주형 특성 처리에 강점.

### 2.3 XGBoost 사용

```python
import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### 2.4 챕터 정리

부스팅은 *편향 감소* 가 목표이며, AdaBoost → GBM → XGBoost → LightGBM 순으로 진화했습니다.

---

## 3. 스태킹 (Stacking)

여러 모델의 예측을 *입력으로 받는 메타 모델* 학습. *서로 다른 종류* 의 모델 (Tree + Linear + Neural) 을 결합할 때 효과적.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(),
).fit(X, y)
```

---

## 4. 정형 데이터에서 부스팅이 강한 이유

- *이종 특성 자연 처리* (수치 + 범주)
- *결측치 처리 내장*
- *비선형 관계 자동 포착*
- *해석 가능 도구* (SHAP, Feature Importance) 풍부
- 같은 정확도에 *훨씬 적은 데이터* 필요

신경망이 정형 데이터에서 *약한 이유* 의 거울 이미지입니다.

---

## 권 정리

- 배깅 = 병렬·분산 감소 (랜덤포레스트)
- 부스팅 = 순차·편향 감소 (XGBoost·LightGBM)
- 스태킹 = 이종 모델 결합
- 정형 데이터 SOTA 는 여전히 부스팅

가장 기억해야 할 한 줄: **"이미지·텍스트는 신경망, 테이블은 부스팅 — 데이터 종류가 알고리즘을 결정한다."**

다음 권: [Volume 24 — SVM과 커널 방법](./volume_24_svm_kernel.md)

---

## 자가점검 키워드

`배깅`, `부스팅`, `랜덤포레스트`, `AdaBoost`, `XGBoost`, `LightGBM`, `스태킹`, `정형 데이터`

## 자가점검 질문

1. 배깅과 부스팅의 차이를 *분산·편향 관점* 에서 설명하십시오.
2. AdaBoost → GBM → XGBoost → LightGBM 의 진화를 한 줄씩 적으십시오.
3. 정형 데이터에서 부스팅이 신경망보다 강한 이유 4 가지를 적으십시오.

## 다음 권

[Volume 24 — SVM과 커널 방법](./volume_24_svm_kernel.md)
