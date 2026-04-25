# Volume 94 — 추론 최적화

> 이 권이 끝나면 새 모델을 받았을 때 *지연 시간 또는 비용을 절반으로 줄이기 위해 무엇을 시도할지* 의 우선순위 표가 머릿속에 그려지게 됩니다.

## 목적

학습은 한 번이지만 추론은 매일 일어납니다. 모델 추론의 비용·지연을 줄이는 일은 AI 시스템 운영의 가장 큰 레버이며, 양자화·그래프 컴파일·KV 캐시·FlashAttention·CUDA Graph·배칭 같은 기법은 각각 다른 병목을 다룹니다. 이 권은 *어떤 병목에 어떤 카드*를 써야 하는지의 의사결정 표를 만듭니다.

## 선수 지식

- Volume 36, 32, 47 완료
- 외부 지식: 지연/처리량·캐시 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 지연(latency) 과 처리량(throughput) 의 트레이드오프를 설명할 수 있습니다.
2. PTQ·QAT·SmoothQuant·AWQ·GPTQ·FP8 양자화의 특징을 구분할 수 있습니다.
3. `torch.compile`·TensorRT·TensorRT-LLM 의 적용 시점을 알 수 있습니다.
4. KV 캐시의 메모리 비용을 추정할 수 있습니다.
5. FlashAttention 이 *왜 빠른가*를 메모리 계층 관점에서 설명할 수 있습니다.

## 챕터 목차

1. **지연 vs 처리량** — 두 다른 KPI
2. **양자화 1 — PTQ vs QAT**
3. **양자화 2 — SmoothQuant·AWQ·GPTQ·FP8**
4. **그래프 컴파일** — `torch.compile` 의 동작
5. **TensorRT / TensorRT-LLM**
6. **CUDA Graph** — 커널 런치 오버헤드 제거
7. **KV 캐시** — LLM 추론의 핵심 자산
8. **FlashAttention 1·2·3** — 메모리 절약 어텐션
9. **배칭 전략** — 정적·동적·연속·인플라이트
10. **모델 파티셔닝** — 텐서 병렬·파이프라인 병렬

## 자가점검 키워드

`Latency/Throughput`, `PTQ/QAT`, `AWQ/GPTQ`, `torch.compile`, `TensorRT`, `KV 캐시`, `FlashAttention`, `Continuous Batching`

## 다음 권

[Volume 86 — 추론 서빙 시스템](./volume_86_serving_systems.md)
