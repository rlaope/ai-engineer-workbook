# Volume 72 — LLM 디코딩 알고리즘

> 이 권이 끝나면 같은 모델·같은 프롬프트로도 *temperature 0.7 vs 1.2* 가 만드는 출력 차이를 직관적으로 예측할 수 있게 됩니다.

## 목적

LLM 의 출력은 *모델*뿐 아니라 *디코딩 알고리즘*이 함께 결정합니다. Greedy·Beam·Sampling·Top-k·Top-p·Speculative 는 각자 다른 트레이드오프를 가집니다. 이 권은 디코딩 전략의 수학적 동작과 *출력 스타일에 미치는 영향*을 다집니다.

## 선수 지식

- Volume 32, 39 완료
- 외부 지식: 확률 분포·샘플링

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Greedy·Beam·Sampling 의 차이를 한 그림으로 설명할 수 있습니다.
2. Top-k·Top-p(Nucleus) 의 차이를 알 수 있습니다.
3. Temperature 가 분포에 미치는 영향을 그래프로 그릴 수 있습니다.
4. Repetition·Frequency·Presence Penalty 의 동기를 알 수 있습니다.
5. Speculative Decoding 의 *추측 + 검증* 발상을 설명할 수 있습니다.

## 챕터 목차

1. **언어 모델의 출력 분포 복습**
2. **Greedy Decoding**
3. **Beam Search**
4. **Sampling — Random·Top-k·Top-p**
5. **Temperature 의 효과**
6. **Repetition·Frequency·Presence Penalty**
7. **Min-p / Mirostat 같은 최신 변형**
8. **Speculative Decoding** — 추측 모델 + 검증
9. **Constrained Generation 미리보기 (Vol 74 와 연결)**

## 자가점검 키워드

`Greedy`, `Beam`, `Top-k`, `Top-p`, `Temperature`, `Repetition Penalty`, `Speculative`, `Mirostat`

## 다음 권

[Volume 73 — 모델 사이즈 의사결정](./volume_73_model_sizing.md)
