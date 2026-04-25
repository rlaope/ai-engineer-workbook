# Volume 35 — 벡터 검색과 ANN

> 이 권이 끝나면 수억 개의 임베딩 안에서 *밀리초 단위로 가장 가까운 이웃*을 찾는 방법을 설명할 수 있게 됩니다.

## 목적

임베딩이 있어도 *그것을 빠르게 검색할 수 없으면* RAG·추천·중복 제거 시스템은 만들 수 없습니다. 정확한 최근접 탐색은 데이터가 커지면 사실상 불가능하므로, 우리는 *근사 최근접 이웃(ANN)* 알고리즘에 의존합니다. FAISS·HNSW·IVF·PQ 가 그 핵심 도구입니다. 이 권은 백엔드 엔지니어에게 가장 친숙한 영역입니다.

## 선수 지식

- Volume 4, 34 완료
- 외부 지식: 인덱스의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 정확한 KNN 의 시간 복잡도와 ANN 의 트레이드오프를 설명할 수 있습니다.
2. IVF·HNSW·PQ 의 핵심 아이디어를 한 문단씩 정리할 수 있습니다.
3. FAISS 또는 ScaNN 으로 100 만 벡터 인덱스를 만들고 검색할 수 있습니다.
4. 벡터 DB(Pinecone·Weaviate·Qdrant·pgvector) 의 비교 기준을 갖게 됩니다.
5. Hybrid Search(키워드 + 벡터) 의 구조를 그릴 수 있습니다.

## 챕터 목차

1. **정확한 KNN 의 한계** — 차원의 저주가 검색에 미치는 영향
2. **ANN 의 정의** — 정확도/속도/메모리 트레이드오프
3. **IVF — Inverted File Index**
4. **HNSW — 계층적 항해 가능한 작은 세계 그래프**
5. **PQ — 곱 양자화로 메모리 절감**
6. **FAISS / ScaNN 사용 패턴**
7. **벡터 DB 비교** — Pinecone·Weaviate·Qdrant·Milvus·pgvector
8. **Hybrid Search 와 재순위**

## 자가점검 키워드

`KNN`, `ANN`, `IVF`, `HNSW`, `PQ`, `FAISS`, `벡터 DB`, `Hybrid Search`

## 다음 권

[Volume 36 — 토크나이저](./volume_36_tokenizer.md)
