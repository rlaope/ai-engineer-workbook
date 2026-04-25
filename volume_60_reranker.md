# Volume 60 — Reranker 깊이

> 이 권이 끝나면 *왜 Bi-Encoder 후 Cross-Encoder 로 재순위* 하는 2 단계 검색이 표준이 되었는지를 설명할 수 있게 됩니다.

## 목적

벡터 검색 (Bi-Encoder) 은 *빠르지만 정확도 한계*. Cross-Encoder 로 *상위 K 개를 다시 순위 매기면* 정확도가 크게 향상됩니다. 이 2 단계 패턴이 *현대 RAG·검색 시스템의 표준* 입니다.

## 선수 지식

- Volume 56, 57 완료

## 학습 결과

1. Bi-Encoder 와 Cross-Encoder 의 차이를 알 수 있습니다.
2. *2 단계 검색 (Retrieve-Rerank)* 의 발상을 이해합니다.
3. ColBERT 의 *Late Interaction* 발상을 알 수 있습니다.
4. LLM Reranker 의 등장과 트레이드오프를 안다.

---

## 1. Bi-Encoder vs Cross-Encoder

### 1.1 Bi-Encoder

```
쿼리 → Encoder → 쿼리 임베딩
문서 → Encoder → 문서 임베딩 (사전 계산)
                          ↓
                   코사인 유사도
```

장점: *문서 임베딩 사전 계산 가능* → 검색 시 빠름.
단점: 쿼리·문서를 *독립적으로* 인코딩 → 미묘한 관계 놓침.

### 1.2 Cross-Encoder

```
[쿼리; 문서] → Encoder → 점수
```

쿼리와 문서를 *함께* 인코더에 입력. *어텐션이 둘을 결합* → 정확도 높음.

단점: 모든 (쿼리, 문서) 쌍을 *런타임에 계산* 해야 함 → 느림.

---

## 2. 2 단계 검색

```
[1 단계 — Bi-Encoder]
쿼리 → 임베딩 → 벡터 DB → 상위 K=100 개 검색

[2 단계 — Cross-Encoder]
쿼리·각 문서 → 점수 계산 → 정렬 → 상위 K'=10 개
```

장점: *Bi-Encoder 의 속도 + Cross-Encoder 의 정확도*.

표준 K = 100, K' = 10. *100 개 Cross-Encoder 호출* 은 *수백 ms 안에 가능*.

---

## 3. 사용 예

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 1 단계: Bi-Encoder
query_emb = bi_encoder.encode(query)
corpus_emb = bi_encoder.encode(corpus)
top_100_idx = np.argsort(corpus_emb @ query_emb)[::-1][:100]

# 2 단계: Cross-Encoder
candidates = [corpus[i] for i in top_100_idx]
scores = cross_encoder.predict([(query, c) for c in candidates])
top_10_idx = top_100_idx[np.argsort(scores)[::-1][:10]]
```

---

## 4. ColBERT — Late Interaction

ColBERT (2020) 는 Bi-Encoder 와 Cross-Encoder 의 *중간 형태*.

각 토큰의 *임베딩을 모두 보존* → 검색 시 *토큰 단위 매칭* (MaxSim).

장점: Cross-Encoder 보다 빠르면서 비슷한 정확도.
단점: *저장 공간 많이* 사용 (각 문서가 수백 개 토큰 임베딩).

---

## 5. LLM Reranker

최근 등장. *LLM 에 (쿼리, 문서) 를 주고 점수 매기게* 하는 방식.

장점: *최고 정확도*. 복잡한 쿼리·도메인 이해.
단점: *매우 느리고 비쌈*. 작은 K (10-30) 에만 적용 현실적.

대표: GPT-4·Claude 를 평가자로 사용.

---

## 권 정리

- Bi-Encoder = 빠름·정확도 한계
- Cross-Encoder = 느림·정확도 높음
- 2 단계 검색 = 100 후보 → 10 위 = 표준
- ColBERT = 중간 (Late Interaction)
- LLM Reranker = 최고 정확도, 비쌈

가장 기억할 한 줄: **"RAG 의 검색 정확도는 Bi-Encoder + Cross-Encoder 의 2 단계 패턴으로 가장 효율적으로 끌어올린다."**

다음 권: [Volume 61 — 임베딩 모델 직접 학습](./volume_61_embedding_training.md)

---

## 자가점검 키워드

`Bi-Encoder`, `Cross-Encoder`, `2 단계 검색`, `ColBERT`, `Late Interaction`, `LLM Reranker`

## 자가점검 질문

1. Bi-Encoder 와 Cross-Encoder 의 트레이드오프를 적으십시오.
2. 2 단계 검색의 흐름을 그리십시오.
3. ColBERT 의 *Late Interaction* 발상을 설명하십시오.
4. LLM Reranker 의 적용 시점을 적으십시오.

## 다음 권

[Volume 61 — 임베딩 모델 직접 학습](./volume_61_embedding_training.md)
