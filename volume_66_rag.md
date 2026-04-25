# Volume 66 — RAG — 검색 증강 생성

> 이 권이 끝나면 빈 회사 문서 더미를 받았을 때 *답변 가능한 챗봇* 을 만드는 표준 파이프라인을 그릴 수 있게 됩니다.

## 목적

RAG (Retrieval-Augmented Generation) 는 LLM 에 *외부 지식* 을 주입하는 가장 표준적인 방법입니다. *학습 데이터에 없는 사내 문서·최신 정보·도메인 특화 지식* 을 LLM 응답에 활용 가능하게 만들며, 산업 LLM 응용의 80% 가 RAG 기반입니다.

## 선수 지식

- Volume 56, 57, 60, 65 완료

## 학습 결과

1. RAG 의 *인덱싱·검색·답변 합성* 3 단계를 그릴 수 있습니다.
2. 청킹 전략의 4 가지 옵션을 비교할 수 있습니다.
3. *검색 정확도 vs 답변 품질* 의 분리 평가를 할 수 있습니다.
4. RAG 의 함정과 개선 패턴을 안다.

---

## 1. RAG 의 3 단계

```
[인덱싱 — 사전]
문서 → 청킹 → 임베딩 → 벡터 DB

[검색 — 런타임]
사용자 쿼리 → 임베딩 → 벡터 DB → 상위 K 청크

[답변 합성 — 런타임]
LLM 입력 = "다음 문서를 기반으로 답하라:
[청크 1]
[청크 2]
...
질문: {쿼리}"
→ LLM 답변
```

이 3 단계가 *모든 RAG 시스템의 골격* 입니다.

---

## 2. 청킹 전략

문서를 *어떻게 자를지* 가 검색 품질을 결정.

### 2.1 고정 길이

```python
chunks = [text[i:i+500] for i in range(0, len(text), 400)]   # 100 토큰 겹침
```

가장 단순. 의미 경계 무시 단점.

### 2.2 문장·문단 단위

문장 또는 문단으로 나눔. 의미 경계 보존.

### 2.3 의미 단위 (Semantic Chunking)

문장 임베딩의 *유사도가 끊기는* 지점에서 분할. 더 정확하지만 느림.

### 2.4 계층적 청킹

문단 + 문장의 *2 단계*. 검색은 문단으로, LLM 입력은 문장 또는 문단 결합.

---

## 3. 미니 RAG (50 줄)

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

# 1. 인덱싱
docs = ["...", "...", "..."]   # 사내 문서
def chunk(text, size=300, overlap=50):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

all_chunks = [c for d in docs for c in chunk(d)]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embs = embedder.encode(all_chunks, normalize_embeddings=True)

# 2. 검색
def retrieve(query, k=5):
    q_emb = embedder.encode(query, normalize_embeddings=True)
    scores = chunk_embs @ q_emb
    top_k = np.argsort(scores)[::-1][:k]
    return [all_chunks[i] for i in top_k]

# 3. 답변 합성
client = OpenAI()
def rag_answer(query):
    context = "\n\n".join(retrieve(query))
    prompt = f"""다음 문서를 기반으로 답하라.
출처에 없는 정보는 추측하지 말고 "모르겠습니다" 로 답하라.

문서:
{context}

질문: {query}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

print(rag_answer("회사 휴가 정책은?"))
```

---

## 4. 검색 vs 답변의 분리 평가

RAG 시스템의 정확도는 *두 부분의 곱*:

- *검색 정확도* — 정답 청크가 상위 K 안에 있는가
- *답변 품질* — 검색된 청크로 LLM 이 정확한 답을 만드는가

각자 따로 평가:

```python
# 검색 정확도
recall = sum(gold in retrieve(q) for q, gold in eval_set) / len(eval_set)

# 답변 품질 (LLM-as-Judge 또는 인간 평가)
answer_quality = ...
```

검색이 95%, 답변이 90% 면 전체 정확도 약 86%.

---

## 5. 흔한 함정

- *청킹이 너무 작음* — 컨텍스트 부족으로 답변 부정확
- *청킹이 너무 큼* — 무관한 정보로 노이즈
- *임베딩 모델이 도메인에 안 맞음* — 검색 정확도 낮음
- *상위 K 가 너무 적음* — 정답 누락
- *상위 K 가 너무 많음* — LLM 컨텍스트 오염
- *시간 정보 누락* — 오래된 문서 우선

---

## 6. 개선 패턴

- **Reranker** (Vol 60) — 상위 K 를 100 → 10 으로 줄이면서 정확도 향상
- **HyDE** — 가상의 답변을 먼저 생성 → 그것으로 검색
- **Multi-Query** — 쿼리를 N 개로 변형해 다양한 검색
- **Hybrid Search** — 벡터 + 키워드 (BM25) 결합
- **Conversational Memory** — 이전 대화로 쿼리 보강

---

## 권 정리

- RAG = 인덱싱 + 검색 + 답변 합성 3 단계
- 청킹 전략 (고정·문장·의미·계층) 이 결정적
- 검색·답변을 *각자 평가*
- Reranker·HyDE·Hybrid·Memory 가 표준 개선 패턴

가장 기억할 한 줄: **"RAG 는 LLM 에 외부 지식을 주입하는 표준이며, 청킹·임베딩·Reranker 의 조합이 정확도를 결정한다."**

다음 권: [Volume 67 — LLM 디코딩 알고리즘](./volume_67_decoding.md)

---

## 자가점검 키워드

`인덱싱·검색·합성`, `청킹`, `Reranker`, `HyDE`, `Hybrid Search`, `Conversational Memory`

## 자가점검 질문

1. RAG 의 3 단계를 그리십시오.
2. 청킹 전략 4 가지를 비교하십시오.
3. RAG 시스템의 정확도를 *검색·답변 분리 평가* 로 측정하는 방법을 적으십시오.
4. RAG 개선 패턴 5 가지를 나열하십시오.

## 다음 권

[Volume 67 — LLM 디코딩 알고리즘](./volume_67_decoding.md)
