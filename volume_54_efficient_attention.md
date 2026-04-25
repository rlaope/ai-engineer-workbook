# Volume 14 — 효율적 어텐션 변형

> 이 권이 끝나면 어텐션을 *근사하거나 재구성하는* 다양한 발상의 지도를 머릿속에 가지게 됩니다.

## 목적

표준 어텐션은 정확하지만 느립니다. Linear Attention·Performer·Linformer 는 *수학적 근사*로 복잡도를 낮추고, FlashAttention 은 *알고리즘은 같지만 메모리 접근*을 최적화합니다. 이 둘의 차이를 이해하면, 새로운 어텐션 변형이 등장할 때마다 *근사인가 재배치인가* 만 식별하면 됩니다. 이 권은 그 분류 도구를 만듭니다.

## 선수 지식

- Volume 51, 47 완료
- 외부 지식: 행렬 곱의 결합법칙

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. *근사 어텐션*과 *정확하지만 더 빠른 어텐션*의 차이를 구분할 수 있습니다.
2. Linear Attention 의 핵심 트릭(소프트맥스 우회) 을 설명할 수 있습니다.
3. Performer 의 무작위 특성 근사 발상을 알 수 있습니다.
4. FlashAttention 1·2·3 의 메모리 계층 활용을 설명할 수 있습니다.
5. Multi-Query·Grouped-Query Attention 의 추론 가속 효과를 알 수 있습니다.

## 챕터 목차

1. **어텐션의 비용 분해** — 시간·메모리·메모리 대역폭
2. **Linear Attention** — 소프트맥스 우회
3. **Performer** — Random Feature 근사
4. **Linformer** — Key·Value 차원 축소
5. **Sparse Attention** — Sliding·Strided·Global
6. **FlashAttention** — 정확하지만 메모리 절약
7. **Multi-Query·Grouped-Query Attention**
8. **Paged Attention** — KV 캐시의 페이징

## 자가점검 키워드

`Linear Attention`, `Performer`, `Linformer`, `Sparse`, `FlashAttention`, `MQA/GQA`, `PagedAttention`, `근사 vs 재배치`

## 다음 권

[Volume 58 — 멀티모달 임베딩](./volume_58_multimodal_embedding.md)
