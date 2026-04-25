# Volume 93 — 모델 모니터링 깊이

> 이 권이 끝나면 *모델이 조용히 망가지고 있는지* 를 감지하는 자동화 시스템을 설계할 수 있게 됩니다.

## 목적

배포된 모델은 시간이 지나면서 *데이터 분포 변화·라벨 변화·예측 분포 변화* 에 의해 점진적으로 정확도가 떨어집니다. 단순 정확도 모니터링은 라벨이 즉시 도착할 때만 가능하므로, 산업 현장에서는 *간접 신호* 로 드리프트를 감지해야 합니다.

## 선수 지식

- Volume 38, 92 완료

## 학습 결과

1. 데이터·컨셉·라벨·예측 드리프트의 차이를 구분할 수 있습니다.
2. PSI·KS·MMD 같은 드리프트 통계를 계산할 수 있습니다.
3. 라벨 지연이 있을 때의 *프록시 메트릭* 설계를 할 수 있습니다.
4. Evidently·NannyML 의 사용 패턴을 알 수 있습니다.

---

## 1. 드리프트의 종류

### 1.1 데이터 드리프트 (Data Drift)

*입력 분포 변화*. 사용자 인구·계절·트렌드 변화로 발생.

### 1.2 컨셉 드리프트 (Concept Drift)

*입력→출력 관계 변화*. 같은 입력에 대한 정답이 시간에 따라 다름.

### 1.3 라벨 드리프트 (Label Drift)

*출력 분포 변화*. 클래스 비율이 시간에 따라 다름.

### 1.4 예측 드리프트 (Prediction Drift)

모델 출력 분포 변화. 입력·관계 변화의 *간접 신호*.

---

## 2. 드리프트 통계

### 2.1 PSI (Population Stability Index)

```python
def psi(expected, actual, bins=10):
    e_dist, _ = np.histogram(expected, bins=bins, density=True)
    a_dist, _ = np.histogram(actual, bins=bins, density=True)
    e_dist = e_dist + 1e-8
    a_dist = a_dist + 1e-8
    return np.sum((a_dist - e_dist) * np.log(a_dist / e_dist))

# 0.1 미만: 변화 없음
# 0.1-0.25: 작은 변화
# 0.25+: 큰 변화 (조사 필요)
```

### 2.2 KS (Kolmogorov-Smirnov)

두 분포의 *최대 차이*. p-value 로 통계적 유의성.

```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(reference, current)
```

### 2.3 MMD (Maximum Mean Discrepancy)

커널 기반. 다차원 분포 차이.

---

## 3. 라벨 지연 환경의 프록시

라벨이 *수일·수주 후* 도착하면 직접 정확도 측정 불가. 프록시:

- *예측 신뢰도 분포* 변화
- *예측 클래스 비율* 변화
- *입력 특성 분포* (PSI)
- *사용자 행동* (재시도·이탈)

---

## 4. Evidently

오픈소스 ML 모니터링.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("drift_report.html")
```

---

## 5. NannyML

라벨 지연 환경 특화. *프록시 기반 정확도 추정*.

---

## 6. 알람 설계

### 6.1 임계값

PSI > 0.25 → 경고. PSI > 0.5 → 즉시 조사.

### 6.2 지속 시간

*5 분 단발성* 은 무시. *30 분 이상 지속* 시 알람.

### 6.3 분기별 변화

계절성·이벤트 기반 변화는 *예외 처리*.

---

## 7. 자동 재학습 트리거

```
드리프트 임계 초과 + N 일 지속 → 자동 재학습 시작
                                  ↓
                              새 모델 → 평가 → 카나리 → 전체 롤아웃
```

이 자동화가 *AutoML 의 핵심 가치*.

---

## 권 정리

- 드리프트 4 종류 = 데이터·컨셉·라벨·예측
- PSI·KS·MMD = 통계적 측정
- 라벨 지연 시 프록시 메트릭
- Evidently·NannyML = 표준 도구
- 알람 = 임계 + 지속 시간 + 계절성 보정
- 자동 재학습 트리거

가장 기억할 한 줄: **"모델 모니터링은 라벨 없이도 드리프트를 감지하는 시스템이며, PSI + 프록시 메트릭이 표준 도구상자다."**

다음 권: [Volume 94 — AI 인시던트 사후 분석](./volume_94_postmortem.md)

---

## 자가점검 키워드

`데이터 드리프트`, `컨셉 드리프트`, `PSI`, `KS`, `MMD`, `프록시 메트릭`, `Evidently`, `NannyML`

## 자가점검 질문

1. 드리프트 4 종류를 적으십시오.
2. PSI 의 임계값과 의미를 설명하십시오.
3. 라벨 지연 시 프록시 메트릭 4 가지를 적으십시오.
4. 자동 재학습 트리거 흐름을 그리십시오.

## 다음 권

[Volume 94 — AI 인시던트 사후 분석](./volume_94_postmortem.md)
