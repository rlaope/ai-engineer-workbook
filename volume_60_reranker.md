# Volume 39 — Reranker 깊이

> 이 권이 끝나면 *왜 Bi-Encoder 후 Cross-Encoder 로 재순위*하는 2 단계 검색이 표준이 되었는지를 설명할 수 있게 됩니다.

## 목적

벡터 검색은 빠르지만 정확도에 한계가 있습니다. Cross-Encoder Reranker 는 *질의-문서 쌍을 함께 인코딩*함으로써 정확도를 높이지만 속도는 느립니다. 이 둘을 *Retrieve → Rerank* 2 단계로 결합하는 것이 RAG·검색 시스템의 표준입니다. 이 권은 Cross-Encoder·ColBERT·LLM Reranker 의 차이와 적용 시점을 다집니다.

## 선수 지식

- Volume 57, 40 완료
- 외부 지식: 검색의 정밀도/재현율

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Bi-Encoder 와 Cross-Encoder 의 구조 차이를 그릴 수 있습니다.
2. ColBERT 의 *Late Interaction* 발상을 설명할 수 있습니다.
3. LLM Reranker 의 비용·정확도 트레이드오프를 알 수 있습니다.
4. 2 단계 검색 시스템의 *후보 수·차단 임계값* 을 결정할 수 있습니다.
5. Reranker 학습 데이터 구성을 설계할 수 있습니다.

## 챕터 목차

1. **Bi-Encoder 의 한계와 재순위의 동기**
2. **Cross-Encoder 의 구조** — 질의-문서 결합 인코딩
3. **ColBERT** — Late Interaction
4. **LLM Reranker** — Listwise·Pointwise·Pairwise
5. **2 단계 검색의 운영** — 후보 수·임계값
6. **재순위 평가 지표**
7. **Reranker 학습 데이터** — 양성·음성·하드 네거티브
8. **Cohere/Voyage Rerank API**

## 자가점검 키워드

`Bi-Encoder`, `Cross-Encoder`, `ColBERT`, `Late Interaction`, `Listwise/Pointwise`, `Hard Negative`, `Cohere Rerank`, `Voyage Rerank`

## 다음 권

[Volume 67 — LLM 디코딩 알고리즘](./volume_67_decoding.md)
