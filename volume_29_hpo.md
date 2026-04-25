# Volume 45 — 하이퍼파라미터 탐색

> 이 권이 끝나면 *수십 개의 실험을 자동으로 효율적으로 굴리는* 시스템을 설계할 수 있게 됩니다.

## 목적

학습률·배치 크기·은닉층 차원·드롭아웃 같은 하이퍼파라미터는 모델 성능을 좌우합니다. 모두 손으로 시도하는 것은 비효율적이며, 그리드/랜덤 서치는 빠르지만 비효율적입니다. 베이지안 최적화·BOHB·Hyperband 같은 알고리즘은 *적은 시도로 좋은 영역을 찾는* 도구입니다. 이 권은 HPO 의 표준 알고리즘과 도구(Optuna·Ray Tune·W&B Sweeps) 를 정리합니다.

## 선수 지식

- Volume 37, 58 완료
- 외부 지식: 그리드 서치 경험

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 그리드·랜덤·베이지안 서치의 트레이드오프를 설명할 수 있습니다.
2. Hyperband·BOHB·ASHA 의 *조기 중단* 발상을 알 수 있습니다.
3. 베이지안 최적화의 획득 함수(EI·UCB) 의 직관을 그릴 수 있습니다.
4. Optuna 또는 Ray Tune 으로 HPO 실험을 자동화할 수 있습니다.
5. HPO 의 함정(검증 세트 누수·과탐색) 을 인식합니다.

## 챕터 목차

1. **하이퍼파라미터의 종류** — 학습·아키텍처·정칙화
2. **그리드 서치 vs 랜덤 서치**
3. **베이지안 최적화** — Surrogate Model + Acquisition Function
4. **Hyperband·ASHA·BOHB** — 조기 중단 알고리즘
5. **Optuna 로 시작하기**
6. **Ray Tune 과 분산 HPO**
7. **W&B Sweeps**
8. **HPO 의 함정** — 누수·과탐색·재현성

## 자가점검 키워드

`Grid/Random Search`, `Bayesian Opt`, `EI/UCB`, `Hyperband`, `ASHA`, `BOHB`, `Optuna`, `Ray Tune`

## 다음 권

[Volume 60 — 활성화 함수 깊이](./volume_39_activations.md)
