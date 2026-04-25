# Volume 66 — 학습률 스케줄 깊이

> 이 권이 끝나면 *학습률 스케줄을 바꾸는 것만으로 같은 모델의 성능을 수 % 끌어올릴 수 있다* 는 사실을 코드로 증명할 수 있게 됩니다.

## 목적

옵티마이저 자체보다 *학습률을 시간에 따라 어떻게 바꿀지*가 결과에 더 큰 영향을 줄 수 있습니다. Warmup·Cosine·OneCycle·Polynomial·ReduceOnPlateau 같은 스케줄은 각자 다른 학습 동학에 적합합니다. 이 권은 각 스케줄의 형태·동기·적용 시점을 다집니다.

## 선수 지식

- Volume 33 완료
- 외부 지식: 함수 그래프 변화

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Warmup 의 동기를 그래디언트 분산 관점에서 설명할 수 있습니다.
2. Cosine·Polynomial·Linear Decay 의 차이를 그래프로 그릴 수 있습니다.
3. OneCycle 정책의 *학습률·모멘텀 동시 변동* 발상을 이해합니다.
4. ReduceOnPlateau 의 적용 시점을 알 수 있습니다.
5. LR Range Test 로 적정 학습률을 빠르게 찾을 수 있습니다.

## 챕터 목차

1. **학습률이 학습에 주는 영향 복습**
2. **Step Decay·Exponential Decay**
3. **Cosine Annealing·Cosine with Restarts**
4. **Linear Warmup + Cosine Decay** — 트랜스포머의 표준
5. **OneCycle Policy**
6. **Polynomial Decay**
7. **ReduceOnPlateau**
8. **LR Range Test (lr_finder)**

## 자가점검 키워드

`Warmup`, `Cosine`, `OneCycle`, `Polynomial Decay`, `ReduceOnPlateau`, `LR Range Test`, `Restarts`, `Step Decay`

## 다음 권

[Volume 41 — 딥러닝 디버깅](./volume_41_dl_debugging.md)
