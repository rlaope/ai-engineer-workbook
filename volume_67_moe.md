# Volume 67 — Mixture of Experts (MoE)

> 이 권이 끝나면 *Mixtral 이 47B 파라미터인데 13B 처럼 빠른* 이유를 설명할 수 있게 됩니다.

## 목적

Mixture of Experts 는 *입력마다 일부 전문가만 활성화*함으로써 파라미터 수와 계산량을 분리합니다. 이는 모델 용량은 키우되 추론 비용은 낮게 유지하는 방법으로, Switch Transformer·Mixtral·DeepSeek-V3 같은 최신 LLM 의 핵심 설계입니다. 이 권은 MoE 의 라우팅·로드 밸런싱·서빙 함의를 정리합니다.

## 선수 지식

- Volume 32, 67 직전 권 완료
- 외부 지식: 라우팅의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. MoE 의 라우터·전문가 구조를 그림으로 그릴 수 있습니다.
2. Top-k 라우팅과 그 트레이드오프를 설명할 수 있습니다.
3. 로드 밸런싱 손실의 동기를 알 수 있습니다.
4. MoE 모델의 메모리 요구량(전문가 전체 로드 필요) 을 계산할 수 있습니다.
5. MoE 가 *서빙 시 까다로운 이유* 를 인식합니다.

## 챕터 목차

1. **MoE 의 직관** — 전문가 분담
2. **라우터의 구조와 학습**
3. **Top-1·Top-2 라우팅**
4. **로드 밸런싱 손실** — 전문가 편중 방지
5. **Switch Transformer**
6. **Mixtral 8×7B 의 구조**
7. **DeepSeek-MoE / Granite MoE**
8. **MoE 서빙의 어려움** — 메모리·통신·부하 변동

## 자가점검 키워드

`MoE`, `라우터`, `Top-k`, `로드 밸런싱`, `Switch`, `Mixtral`, `DeepSeek-MoE`, `희소 활성화`

## 다음 권

[Volume 68 — 효율적 어텐션 변형](./volume_68_efficient_attention.md)
