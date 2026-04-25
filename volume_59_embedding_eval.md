# Volume 59 — 임베딩 평가와 벤치마크

> 이 권이 끝나면 *우리 도메인에 어떤 임베딩 모델을 쓸지* 를 30 분 안에 결정할 수 있게 됩니다.

## 목적

수백 개의 임베딩 모델 중 *자기 도메인에 가장 잘 맞는 것* 을 고르는 일은 RAG 시스템 품질의 가장 큰 지렛대입니다. MTEB·BEIR 같은 표준 벤치마크와 *자기 도메인 평가* 를 결합하는 사고법을 다집니다.

## 선수 지식

- Volume 56, 57 완료

## 학습 결과

1. MTEB·BEIR 의 차이를 알 수 있습니다.
2. *Top-K Recall·MRR·NDCG* 메트릭을 적용할 수 있습니다.
3. 자기 도메인 평가 셋을 빠르게 구축할 수 있습니다.
4. *공개 벤치마크와 자기 도메인의 격차* 를 인식합니다.

---

## 1. MTEB

### 1.1 정의

*Massive Text Embedding Benchmark* (Hugging Face). 임베딩 모델의 표준 평가:

- 분류 (Classification)
- 클러스터링
- 검색 (Retrieval)
- 재순위
- 의미 유사도 (STS)
- 요약

100+ 개 데이터셋, 100+ 개 언어.

### 1.2 활용

[Hugging Face MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 에서 모델별 순위 확인.

상위 모델은 보통 *큰 사이즈 + 다양한 학습 데이터* 가 특징.

---

## 2. BEIR

*Benchmarking IR* — 검색에 특화. 18 개 검색 데이터셋. *Zero-shot* 검색 능력 평가.

---

## 3. 메트릭

### 3.1 Top-K Recall

상위 K 개 결과 안에 *정답이 있는가* 의 비율.

### 3.2 MRR (Mean Reciprocal Rank)

첫 정답의 *위치 역수* 평균. 1 위면 1.0, 10 위면 0.1.

### 3.3 NDCG (Normalized Discounted Cumulative Gain)

상위 결과의 *위치별 가중 점수*. 위치가 낮을수록 가중치 감소.

```python
def reciprocal_rank(ranking, gold):
    for i, item in enumerate(ranking, 1):
        if item == gold:
            return 1.0 / i
    return 0.0

mrr = np.mean([reciprocal_rank(r, g) for r, g in zip(rankings, golds)])
```

---

## 4. 자기 도메인 평가

### 4.1 평가 셋 구축

```
1. 자기 도메인의 (질의, 정답 문서) 쌍 50-200 개 수집
2. 도메인 전문가 또는 사용자 피드백으로 라벨링
3. 정기적으로 (월 1 회) 셋 갱신
```

50 쌍이면 *모델 비교는 가능*, 100+ 쌍이면 *통계적으로 유의미*.

### 4.2 비교 실험

```python
models = ['all-MiniLM-L6-v2', 'bge-large-en', 'text-embedding-3-large']
for model_name in models:
    model = SentenceTransformer(model_name)
    corpus_emb = model.encode(corpus)
    
    recalls = []
    for query, gold_id in eval_set:
        query_emb = model.encode(query)
        scores = corpus_emb @ query_emb
        top_10 = np.argsort(scores)[::-1][:10]
        recalls.append(gold_id in top_10)
    
    print(f"{model_name}: Recall@10 = {np.mean(recalls):.3f}")
```

---

## 5. 공개 벤치마크 vs 자기 도메인

MTEB 1 위가 *자기 도메인에서 1 위가 아닐 수 있습니다*.

이유:
- *학습 데이터 분포* 차이 — 의료·법률·코드 같은 특수 도메인
- *언어* — 한국어·일본어 등에서 다른 순위
- *길이 분포* — 짧은 쿼리·긴 문서 같은 특성

따라서 *반드시 자기 도메인 평가* 를 거쳐야 합니다.

---

## 권 정리

- MTEB·BEIR = 표준 벤치마크
- Top-K Recall·MRR·NDCG = 표준 메트릭
- 자기 도메인 평가 셋 50-200 쌍이 필수
- 공개 1 위가 자기 도메인 1 위가 아닐 수 있음

가장 기억할 한 줄: **"MTEB 순위로 후보 5 개를 좁히고, 자기 도메인 평가 셋으로 최종 선택한다."**

다음 권: [Volume 60 — Reranker 깊이](./volume_60_reranker.md)

---

## 자가점검 키워드

`MTEB`, `BEIR`, `Recall@K`, `MRR`, `NDCG`, `자기 도메인 평가`

## 자가점검 질문

1. MTEB 와 BEIR 의 차이를 적으십시오.
2. Recall@K·MRR·NDCG 의 차이를 설명하십시오.
3. 자기 도메인 평가 셋의 최소 크기와 권장 크기를 적으십시오.
4. *공개 벤치마크 1 위 ≠ 자기 도메인 1 위* 인 이유를 설명하십시오.

## 다음 권

[Volume 60 — Reranker 깊이](./volume_60_reranker.md)
