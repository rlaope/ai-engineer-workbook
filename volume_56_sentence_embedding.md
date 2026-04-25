# Volume 56 — 문장·문서 임베딩

> 이 권이 끝나면 *어떤 텍스트든 벡터 한 줄로 변환해 검색·분류·클러스터링에 쓸 수 있는 사고법* 을 갖추게 됩니다.

## 목적

단어 임베딩에서 한 단계 위, *문장·문단·문서 전체* 를 *고정 차원 벡터* 로 표현하는 기법입니다. RAG·시멘틱 검색·문서 분류의 가장 직접적 기반이며, AI 엔지니어가 매일 사용하는 도구입니다.

## 선수 지식

- Volume 55 완료

## 학습 결과

1. Sentence-BERT 의 *Siamese 학습* 구조를 이해합니다.
2. OpenAI Embedding·E5·BGE 같은 표준 모델을 알 수 있습니다.
3. 임베딩 모델 선택 기준 (차원·언어·도메인) 을 적용할 수 있습니다.
4. Hugging Face `sentence-transformers` 로 임베딩을 만들 수 있습니다.

---

## 1. 단순 평균 임베딩의 한계

문장 임베딩의 *가장 단순한 시도*: 단어 임베딩들의 평균.

```python
sentence = "AI engineer is great"
emb = np.mean([word_emb[w] for w in sentence.split()], axis=0)
```

문제: *어순·구문 무시*. *"dog bites man"* 과 *"man bites dog"* 가 같은 임베딩.

---

## 2. Sentence-BERT (2019)

### 2.1 발상

BERT 의 *문장 임베딩* 을 *Siamese 구조 + 대조 학습* 으로 fine-tune. 두 문장 임베딩의 *코사인 유사도* 가 *의미적 유사도* 에 일치하도록.

### 2.2 사용

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["AI engineering is fun.", "Software development is creative."]
embeddings = model.encode(sentences)
print(embeddings.shape)   # (2, 384)
```

`all-MiniLM-L6-v2` 는 *6 층의 작은 모델*, 384 차원, 매우 빠름. 산업 표준 중 하나.

---

## 3. OpenAI Embedding

### 3.1 모델

- `text-embedding-3-small` — 1536 차원, 저렴
- `text-embedding-3-large` — 3072 차원, 정확도 높음

### 3.2 사용

```python
from openai import OpenAI
client = OpenAI()

emb = client.embeddings.create(
    input="AI engineering",
    model="text-embedding-3-small"
).data[0].embedding
print(len(emb))   # 1536
```

장점: *최고 수준 정확도*. 단점: API 비용·외부 의존.

---

## 4. 한국어·다국어

- **multilingual-e5** — 100+ 언어
- **bge-m3** (BAAI) — 다국어 + 다중 검색 모드
- **KoSimCSE** — 한국어 특화
- **KoSentenceBERT**

한국어 도메인은 *multilingual 모델 + 자기 도메인 fine-tune* 이 표준.

---

## 5. 임베딩 모델 선택 기준

```
+----------+-------------------+
| 기준     | 권장              |
+----------+-------------------+
| 빠른 시작 | OpenAI API        |
| 자체 운영 | sentence-transformers |
| 한국어    | bge-m3, KoSimCSE  |
| 도메인    | 자기 데이터 fine-tune |
| 큰 차원   | text-embedding-3-large |
| 작은 차원 | all-MiniLM-L6-v2  |
+----------+-------------------+
```

---

## 6. 임베딩 평가

벡터 검색 정확도로 평가. 표준 벤치마크: **MTEB (Massive Text Embedding Benchmark)**.

자기 도메인 평가는 *(질의, 정답 문서) 쌍* 을 만들어 *Top-K Recall*·*MRR* 측정.

---

## 권 정리

- 단순 평균 = 어순 무시 한계
- Sentence-BERT = Siamese + 대조 학습
- OpenAI Embedding = SaaS 표준
- 한국어는 multilingual + fine-tune
- MTEB 로 모델 평가

가장 기억할 한 줄: **"문장 임베딩은 RAG·검색·분류의 코어이며, sentence-transformers 또는 OpenAI Embedding 으로 즉시 사용 가능하다."**

다음 권: [Volume 57 — 벡터 검색과 ANN](./volume_57_vector_search.md)

---

## 자가점검 키워드

`Sentence-BERT`, `Siamese`, `대조 학습`, `OpenAI Embedding`, `multilingual-e5`, `MTEB`

## 자가점검 질문

1. 단순 평균 임베딩의 한계를 설명하십시오.
2. Sentence-BERT 의 학습 방식을 한 문단으로 적으십시오.
3. 한국어 도메인에 적합한 임베딩 모델 3 가지를 적으십시오.
4. 임베딩 모델 평가의 표준 벤치마크 이름과 메트릭을 적으십시오.

## 다음 권

[Volume 57 — 벡터 검색과 ANN](./volume_57_vector_search.md)
