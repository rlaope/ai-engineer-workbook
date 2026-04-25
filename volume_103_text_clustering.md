# Volume 103 — 텍스트 클러스터링·토픽 모델링

> 이 권이 끝나면 *수만 건의 문서를 라벨 없이 의미 있는 그룹으로 묶고* 그 그룹의 주제를 자동으로 명명할 수 있게 됩니다.

## 목적

라벨이 없는 대량 텍스트 (사용자 피드백·뉴스·검색 로그 등) 에서 *자연스러운 주제 그룹* 을 발견하는 일이 텍스트 클러스터링·토픽 모델링입니다. BERTopic 같은 현대 도구가 *임베딩 + 클러스터링 + LLM 명명* 을 통합합니다.

## 선수 지식

- Volume 25, 26, 56 완료

## 학습 결과

1. 텍스트 클러스터링의 표준 파이프라인 (임베딩 → 차원축소 → 클러스터링) 을 안다.
2. LDA 와 BERTopic 의 차이를 알 수 있습니다.
3. 클러스터에 *자동 이름 부여* 하는 LLM 활용을 적용합니다.
4. 산업 응용 (피드백 분석·뉴스 토픽) 을 알 수 있습니다.

---

## 1. 표준 파이프라인

```
원시 텍스트 → [임베딩] → [차원 축소 (UMAP)] → [클러스터링 (HDBSCAN)] → [LLM 명명]
```

### 1.1 임베딩

Sentence-BERT 또는 OpenAI Embedding (Vol 56).

### 1.2 차원 축소

UMAP 으로 *2-50 차원* 으로 축소. 클러스터링 효율·시각화 용도.

### 1.3 클러스터링

K-Means 또는 HDBSCAN. HDBSCAN 은 *K 사전 지정 불필요* + *이상치 자동 분리* 로 텍스트에 적합.

### 1.4 LLM 명명

각 클러스터의 *대표 문서들* 을 LLM 에 보여 *주제 이름* 자동 생성.

---

## 2. BERTopic

전체 파이프라인을 통합한 라이브러리.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=embedder)

topics, probs = topic_model.fit_transform(documents)
print(topic_model.get_topic_info())
```

각 토픽의 *대표 단어·대표 문서·크기* 자동 출력.

---

## 3. LDA — 고전 토픽 모델링

### 3.1 발상

각 문서가 *여러 토픽의 혼합*, 각 토픽이 *여러 단어의 분포*. 이중 분포를 학습.

```python
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

texts = [[word for word in doc.lower().split()] for doc in documents]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = LdaModel(corpus, num_topics=10, id2word=dictionary)
print(lda.print_topics())
```

### 3.2 한계

- *bag-of-words* 기반 — 어순·문맥 무시
- *토픽 수 사전 지정* 필요
- *짧은 문서에 약함*

BERTopic 이 등장하면서 *LDA 사용 줄어듦*.

---

## 4. LLM 으로 토픽 명명

```python
def name_topic(cluster_documents):
    sample = "\n---\n".join(cluster_documents[:5])
    prompt = f"""Below are documents from a single topic cluster.
Provide a 2-5 word topic name.

Documents:
{sample}

Topic name:"""
    return llm(prompt).strip()

for topic_id in topic_ids:
    docs = get_cluster_docs(topic_id)
    name = name_topic(docs)
    print(f"Topic {topic_id}: {name}")
```

자동 명명으로 *수동 라벨링 비용 0*.

---

## 5. 동적 토픽 모델링

시간에 따른 *토픽 변화 추적*. BERTopic 이 지원.

```python
topic_model.topics_over_time(documents, timestamps)
```

응용: *뉴스 트렌드·고객 관심사 변화* 분석.

---

## 6. 산업 응용

- **고객 피드백 분류** — 수만 건 피드백 → 자동 카테고리화
- **뉴스 트렌드** — 오늘 가장 화제인 주제
- **연구 논문 군집** — 새 연구 분야 발견
- **검색 로그** — 사용자 의도 분류

---

## 7. 평가

라벨 없으므로 *직접 평가 어려움*. 간접 평가:

- *일관성 (Coherence)* — 같은 토픽 단어들의 의미 유사도
- *분리도 (Separation)* — 다른 토픽 사이 거리
- *수동 검토* — 도메인 전문가가 토픽 의미 확인

---

## 권 정리

- 표준 파이프라인 = 임베딩 → UMAP → HDBSCAN → LLM 명명
- BERTopic = 통합 라이브러리
- LDA = 고전, BERTopic 으로 대체됨
- LLM 자동 명명으로 수동 라벨링 0
- 동적 토픽 = 시간 추적
- 평가 = Coherence·Separation·수동 검토

가장 기억할 한 줄: **"BERTopic 한 줄로 라벨 없는 수만 건 텍스트를 의미 있는 토픽으로 묶고 자동 명명까지 가능하다."**

다음 권: [Volume 104 — 데이터셋 종합 실습 워크북](./volume_104_dataset_workbook.md)

---

## 자가점검 키워드

`BERTopic`, `LDA`, `UMAP`, `HDBSCAN`, `LLM 명명`, `Coherence`

## 자가점검 질문

1. 텍스트 클러스터링 표준 파이프라인 4 단계를 적으십시오.
2. BERTopic 과 LDA 의 차이를 적으십시오.
3. LLM 자동 명명의 효용을 설명하십시오.
4. 자기 도메인의 텍스트 클러스터링 응용 사례를 적으십시오.

## 다음 권

[Volume 104 — 데이터셋 종합 실습 워크북](./volume_104_dataset_workbook.md)
