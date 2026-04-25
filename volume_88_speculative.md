# Volume 98 — Speculative Decoding

> 이 권이 끝나면 *작은 모델로 추측하고 큰 모델로 검증* 하는 발상이 어떻게 정확도 손실 없이 LLM 추론을 가속하는지 설명할 수 있게 됩니다.

## 목적

Speculative Decoding 은 LLM 추론의 *순차성*이라는 본질적 병목을 우회하는 방법입니다. 작은 *Draft 모델*이 여러 토큰을 추측한 뒤, 큰 *Target 모델*이 이들을 *한 번에* 검증·수정합니다. 검증된 토큰은 그대로 채택되므로 정확도는 그대로면서 처리량은 2-3 배 올라갑니다. Medusa·EAGLE·Lookahead 같은 변형이 다양하게 등장합니다. 이 권은 Speculative Decoding 의 알고리즘과 변형을 정리합니다.

## 선수 지식

- Volume 51, 48, 72 완료
- 외부 지식: 분기 예측의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Speculative Decoding 의 *추측 + 검증* 알고리즘을 손으로 따라갈 수 있습니다.
2. 정확도가 손상되지 않는 수학적 이유를 설명할 수 있습니다.
3. Draft 모델 선택의 트레이드오프를 알 수 있습니다.
4. Medusa·EAGLE·Lookahead 의 차별점을 구분할 수 있습니다.
5. *어떤 워크로드에서 가속이 큰가* 를 예측할 수 있습니다.

## 챕터 목차

1. **LLM 추론의 순차성 병목**
2. **표준 Speculative Decoding 알고리즘**
3. **수용/거절 확률** — 정확도 보존의 수학
4. **Draft 모델 선택** — 같은 계열 vs 외부
5. **Medusa** — 추가 헤드 기반 추측
6. **EAGLE / EAGLE-2** — 임베딩 기반 추측
7. **Lookahead Decoding** — N-gram 기반
8. **워크로드별 가속 효과**

## 자가점검 키워드

`Speculative`, `Draft 모델`, `Target 모델`, `수용/거절`, `Medusa`, `EAGLE`, `Lookahead`, `정확도 보존`

## 다음 권

[Volume 83 — 분산 학습](./volume_89_distributed_training.md)
