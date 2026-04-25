# Volume 57 — 벡터 검색과 ANN

> 이 권이 끝나면 수억 개의 임베딩 안에서 *밀리초 단위로 가장 가까운 이웃* 을 찾는 방법을 설명할 수 있게 됩니다.

## 목적

RAG 의 *검색* 단계는 본질적으로 *고차원 벡터 공간에서 가장 가까운 이웃 K 개를 찾는 일* 입니다. 1 억 개의 1536 차원 벡터에서 *완전 탐색* 으로 거리를 계산하면 한 쿼리당 수 분이 걸립니다. ANN (Approximate Nearest Neighbor) 알고리즘이 이 비용을 *밀리초* 로 줄입니다.

## 선수 지식

- Volume 8, 56 완료

## 학습 결과

1. 완전 탐색의 비용을 알 수 있습니다.
2. ANN 알고리즘 (HNSW·IVF·PQ) 의 발상을 이해합니다.
3. FAISS·Qdrant·pgvector·Pinecone 의 차이를 알 수 있습니다.
4. *재현율 vs 속도* 트레이드오프를 적용할 수 있습니다.

---

## 1. 완전 탐색 vs ANN

### 1.1 완전 탐색

```python
import numpy as np

# N 개 임베딩, 차원 D
N, D = 100_000, 768
corpus = np.random.randn(N, D)
query = np.random.randn(D)

# 정규화 후 내적
corpus_n = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
query_n = query / np.linalg.norm(query)

scores = corpus_n @ query_n
top_k = np.argsort(scores)[::-1][:10]
```

100K 벡터·768 차원에서 한 쿼리당 *수십 ms*. 100M 으로 늘면 *수십 초*. 사용자 응답으로 부적합.

### 1.2 ANN 의 발상

*100% 정확한 답 대신 99% 정확한 답* 을 *1000 배 빠르게*. 검색 도메인에서는 거의 항상 합리적 트레이드오프.

---

## 2. HNSW (Hierarchical Navigable Small World)

### 2.1 발상

벡터들을 *계층적 그래프* 로 연결. 상위 층은 *멀리 보는 시야*, 하위 층은 *세밀한 조정*. 상위 → 하위로 *그리디 탐색*.

### 2.2 강점

- 매우 빠름
- 높은 재현율 (99%+)
- 메모리 사용량 적당

가장 인기 있는 ANN 알고리즘. 거의 모든 벡터 DB 의 기본.

---

## 3. IVF (Inverted File)

### 3.1 발상

벡터를 *K-Means 로 클러스터링* → 쿼리는 *가까운 몇 개 클러스터* 안에서만 탐색.

### 3.2 강점

- 메모리 효율
- 큰 데이터셋 (수십억) 에 강함

단점: HNSW 보다 일반적으로 느림 또는 정확도 낮음.

---

## 4. PQ (Product Quantization)

### 4.1 발상

벡터를 *작은 부분으로 나눠 각각 양자화* → 메모리 대폭 감소. 검색 시 *근사 거리* 계산.

768 차원 FP32 = 3KB → PQ 8 바이트로 *400 배 압축*.

### 4.2 강점

- 메모리 매우 적음
- 큰 데이터셋 가능

단점: 정확도 손실 큼. 보통 IVF + PQ 결합.

---

## 5. 벡터 DB 비교

```
+------------+-------------------+------------------+
| 도구       | 강점              | 적용 시점         |
+------------+-------------------+------------------+
| FAISS      | 라이브러리, 무료   | 자체 통합        |
| Qdrant     | Rust, REST, 빠름  | 프로덕션         |
| Weaviate   | GraphQL, 메타데이터| 복잡한 쿼리      |
| pgvector   | PostgreSQL 확장   | 기존 DB 통합      |
| Pinecone   | SaaS, 관리 X      | 빠른 시작        |
| Milvus     | 대규모, 분산      | 수십억 벡터       |
+------------+-------------------+------------------+
```

---

## 6. 재현율 vs 속도

ANN 알고리즘은 보통 *파라미터 (HNSW 의 efSearch 등)* 로 튜닝. 큰 값 = 정확, 작은 값 = 빠름.

운영 권장:
- *프로덕션 검색* — recall@10 = 95%+ 목표
- *오프라인 분석* — 완전 탐색 사용 가능

---

## 7. 미니 FAISS 사용

```python
import faiss
import numpy as np

D = 768
N = 100_000
corpus = np.random.randn(N, D).astype(np.float32)

# IndexFlatIP = 완전 탐색 (내적)
index = faiss.IndexFlatIP(D)
index.add(corpus)

# 검색
query = np.random.randn(1, D).astype(np.float32)
scores, indices = index.search(query, k=10)
print(indices)

# HNSW 로 변경
index = faiss.IndexHNSWFlat(D, 32)   # M=32 이웃
index.add(corpus)
```

---

## 권 정리

- 완전 탐색 = 100M 에서 수십 초, 부적합
- ANN = 99% 정확도 + 1000 배 빠름
- HNSW = 가장 인기, 그래프 기반
- IVF·PQ = 메모리·대규모 데이터
- 벡터 DB: FAISS·Qdrant·pgvector·Pinecone·Milvus
- 재현율 vs 속도 트레이드오프 튜닝

가장 기억할 한 줄: **"벡터 검색은 100% 정확함을 99%·1000 배 빠름과 교환하는 일이며, 거의 항상 합리적 거래이다."**

다음 권: [Volume 58 — 멀티모달 임베딩](./volume_58_multimodal_embedding.md)

---

## 자가점검 키워드

`완전 탐색`, `ANN`, `HNSW`, `IVF`, `PQ`, `FAISS`, `Qdrant`, `pgvector`, `재현율`

## 자가점검 질문

1. ANN 의 *재현율 vs 속도* 트레이드오프를 설명하십시오.
2. HNSW·IVF·PQ 의 차이를 적으십시오.
3. 벡터 DB 5 가지의 적용 시점을 비교하십시오.
4. FAISS 로 HNSW 인덱스를 만드는 코드를 적으십시오.

## 다음 권

[Volume 58 — 멀티모달 임베딩](./volume_58_multimodal_embedding.md)
