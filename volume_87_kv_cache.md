# Volume 70 — KV 캐시 깊이

> 이 권이 끝나면 *왜 LLM 추론에서 KV 캐시가 메모리의 가장 큰 부분이 되는가* 와 *그 메모리를 줄이는 모든 기법*을 알게 됩니다.

## 목적

LLM 추론은 *지난 토큰의 K·V 를 다시 계산하지 않기 위해* 캐시합니다. 이 캐시는 시퀀스 길이·배치·헤드 수·차원의 곱이며, 긴 컨텍스트와 큰 배치에서는 모델 가중치보다 더 큰 메모리를 차지합니다. PagedAttention·MQA·GQA·KV 압축·Prefix Cache 가 이 메모리를 통제하는 도구입니다. 이 권은 KV 캐시의 모든 측면을 정리합니다.

## 선수 지식

- Volume 51, 47, 48 완료
- 외부 지식: 캐시·페이지 테이블의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. KV 캐시 메모리를 모델 차원에서 계산할 수 있습니다.
2. MQA·GQA 가 어떻게 KV 메모리를 줄이는지 그림으로 설명할 수 있습니다.
3. PagedAttention 의 페이지 테이블 발상을 알 수 있습니다.
4. Prefix Cache 와 Prompt Cache 의 차이를 알 수 있습니다.
5. KV 캐시 양자화·압축 기법을 인식합니다.

## 챕터 목차

1. **KV 캐시의 정의와 동기**
2. **메모리 계산 공식** — `2 × layers × heads × head_dim × seq_len × batch × dtype`
3. **Multi-Query Attention (MQA)**
4. **Grouped-Query Attention (GQA)**
5. **PagedAttention** — vLLM 의 핵심
6. **Prefix Cache·Prompt Cache** — 시스템 프롬프트 재사용
7. **KV 캐시 양자화** — INT8·FP8·INT4
8. **장기 컨텍스트의 KV 압축** — H2O·StreamingLLM

## 자가점검 키워드

`KV 캐시`, `MQA`, `GQA`, `PagedAttention`, `Prefix Cache`, `KV 양자화`, `H2O`, `StreamingLLM`

## 다음 권

[Volume 88 — Speculative Decoding](./volume_88_speculative.md)
