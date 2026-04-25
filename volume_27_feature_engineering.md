# Volume 43 — 특성 공학

> 이 권이 끝나면 정형 데이터를 받았을 때 *모델에 넣기 전에 무엇을 해야 하는가* 의 표준 체크리스트가 머릿속에 그려지게 됩니다.

## 목적

특성 공학은 *모델 선택보다 더 큰 영향*을 결과에 미치는 경우가 많습니다. 누락 처리·인코딩·스케일링·이상치 처리·새 특성 생성 같은 단계가 데이터를 모델이 *학습 가능한 형태*로 바꿉니다. 이 권은 정형 데이터 ML 워크플로의 표준 전처리 단계를 정리합니다.

## 선수 지식

- Volume 13, 10 완료
- 외부 지식: pandas 기본

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 결측치 처리 방법(삭제·대체·예측 보완) 의 트레이드오프를 알 수 있습니다.
2. 범주형 인코딩(One-Hot·Ordinal·Target·Frequency) 의 적용 시점을 구분할 수 있습니다.
3. 스케일링(Min-Max·Standard·Robust) 이 어떤 모델에 영향을 주는지 알 수 있습니다.
4. 이상치 탐지·처리의 표준 기법을 적용할 수 있습니다.
5. 도메인 지식을 새 특성으로 변환하는 워크플로를 가질 수 있습니다.

## 챕터 목차

1. **데이터 검사 — 첫 30 분의 원칙**
2. **결측치 처리** — 삭제·대체·KNN Imputer·예측 모델
3. **범주형 인코딩** — One-Hot·Ordinal·Target·Frequency·Hash
4. **수치형 변환** — Log·Box-Cox·Yeo-Johnson·Binning
5. **스케일링** — Min-Max·Standard·Robust·Quantile
6. **이상치 탐지·처리** — IQR·Z-score·Isolation Forest
7. **시간 특성** — 주기·델타·집계
8. **타깃 누수의 함정**
9. **scikit-learn Pipeline 으로 정리하기**

## 자가점검 키워드

`결측치`, `One-Hot/Target Encoding`, `스케일링`, `Box-Cox`, `이상치`, `시간 특성`, `타깃 누수`, `Pipeline`

## 다음 권

[Volume 58 — 베이지안 머신러닝](./volume_28_bayesian_ml.md)
