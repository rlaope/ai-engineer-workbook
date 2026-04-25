# Volume 29 — 하이퍼파라미터 탐색

> 이 권이 끝나면 *수십 개의 실험을 자동으로 효율적으로 굴리는* 시스템을 설계할 수 있게 됩니다.

## 목적

ML 모델의 성능은 *하이퍼파라미터 선택* 에 크게 의존합니다. 학습률·배치 크기·정칙화 강도·아키텍처 결정 등을 *수동으로* 시도하면 시간이 매우 많이 듭니다. 자동 하이퍼파라미터 탐색 (HPO) 도구가 이 과정을 표준화합니다. Grid·Random·Bayesian·BOHB 의 차이와 Optuna 사용을 다집니다.

## 선수 지식

- Volume 23, 24 완료

## 학습 결과

1. Grid·Random·Bayesian·BOHB 의 차이를 알 수 있습니다.
2. Optuna 로 자동 탐색을 구현할 수 있습니다.
3. Early Stopping 과 Pruning 으로 비용을 절감할 수 있습니다.
4. *탐색 공간 정의* 의 함정을 안다.

---

## 1. 탐색 알고리즘

### 1.1 Grid Search

각 파라미터의 *모든 조합* 시도. 차원이 늘면 폭발.

```python
from sklearn.model_selection import GridSearchCV
grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
search = GridSearchCV(SVC(), grid, cv=5).fit(X, y)
```

3 × 2 = 6 조합. 5-fold 교차 검증 → 30 학습.

### 1.2 Random Search

탐색 공간에서 *무작위 샘플*. Grid 보다 효율적이라는 것이 *경험적 결과* (Bergstra & Bengio, 2012).

```python
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(SVC(), grid, n_iter=20).fit(X, y)
```

### 1.3 Bayesian Optimization

이전 시도의 결과로 *다음 시도를 추천*. 가우시안 프로세스 또는 TPE (Tree Parzen Estimator) 가 표준.

가장 적은 시도로 좋은 결과를 찾는 경향이 있음.

### 1.4 BOHB / ASHA

*Bayesian + Hyperband* 조합. 나쁜 시도를 *조기 종료* 해 자원을 좋은 후보에 집중.

---

## 2. Optuna 사용

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    n_layers = trial.suggest_int('n_layers', 1, 5)
    
    model = build_model(n_layers)
    val_loss = train(model, lr, batch_size)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(study.best_params)
```

Optuna 의 강점:

- *Define-by-Run* — 함수 안에서 파라미터 정의 (조건부 가능)
- *Pruning* — 중간 결과로 나쁜 시도 조기 종료
- *분산 실행* — 여러 머신에서 병렬

---

## 3. Pruning 으로 비용 절감

```python
def objective(trial):
    for epoch in range(100):
        train_one_epoch()
        intermediate = validate()
        trial.report(intermediate, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return final_val_loss

study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
```

중간에 *다른 시도들의 중앙값보다 나쁘면 종료*. 전체 비용 50%+ 절감 흔함.

---

## 4. 탐색 공간 정의의 함정

- *너무 좁음* — 최적이 공간 밖에 있을 수 있음
- *너무 넓음* — 효율 악화
- *log 스케일 누락* — 학습률은 *항상 log 스케일* 로 (1e-5 ~ 1e-1)
- *조건부 누락* — `n_layers >= 2` 일 때만 의미 있는 파라미터
- *중요도 무시* — 모든 파라미터에 같은 시간 배분

---

## 권 정리

- Grid·Random·Bayesian·BOHB — 효율 점진 향상
- Optuna = 산업 표준 (Define-by-Run, Pruning)
- 탐색 공간 정의가 *결과를 좌우*
- 학습률은 *항상 log 스케일*

가장 기억할 한 줄: **"수동 튜닝 시간을 자동 HPO 와 GPU 시간으로 바꿔라 — 거의 항상 더 싸다."**

다음 권: [Volume 30 — 퍼셉트론과 신경망의 기원](./volume_30_perceptron.md)

---

## 자가점검 키워드

`Grid`, `Random`, `Bayesian`, `BOHB/ASHA`, `Optuna`, `Pruning`, `log 스케일`

## 자가점검 질문

1. Grid 와 Random Search 의 효율 차이를 설명하십시오.
2. Bayesian Optimization 이 *적은 시도로* 좋은 결과를 찾는 메커니즘을 적으십시오.
3. Optuna 의 Pruning 동작을 한 문단으로 설명하십시오.
4. 탐색 공간 정의의 5 가지 함정을 적으십시오.

## 다음 권

[Volume 30 — 퍼셉트론과 신경망의 기원](./volume_30_perceptron.md)
