# Volume 102 — 텍스트 분류 4 접근 비교

> 이 권이 끝나면 새 분류 작업을 받았을 때 *4 가지 접근 중 어떤 것을 시도할지* 를 30 초 안에 결정할 수 있게 됩니다.

## 목적

같은 텍스트 분류 작업을 *표현 모델 직접 분류·작업 특화 fine-tuned 모델·임베딩 + 단순 분류기·생성 모델 zero/few-shot* 의 4 가지 방법으로 풀 수 있습니다. 각 방법은 데이터 양·정확도·지연·비용에서 다른 트레이드오프를 가집니다.

## 선수 지식

- Volume 22, 56, 65 완료

## 학습 결과

1. 4 가지 접근의 *입력·출력·학습 비용·추론 비용* 을 표로 비교할 수 있습니다.
2. 데이터 양에 따른 권장 접근을 제시할 수 있습니다.
3. 같은 데이터셋으로 4 접근을 모두 구현·비교할 수 있습니다.
4. 산업 사용 사례별 표준 선택 매트릭스를 갖습니다.

---

## 1. 4 가지 접근

### 1.1 표현 모델 직접 분류 (BERT 등)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
# 학습 데이터로 미세조정
# ...
```

장점: 정확도 높음. 단점: 미세조정 필요 (최소 수천 샘플).

### 1.2 작업 특화 사전학습 모델

이미 *유사한 작업으로 학습된 모델* 사용. 미세조정 없이 *zero-shot*.

```python
sentiment = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
result = sentiment("I love this movie!")
```

장점: 즉시 사용. 단점: 정확히 일치하는 작업 모델이 있어야.

### 1.3 임베딩 + 단순 분류기

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_emb = embedder.encode(texts)

clf = LogisticRegression()
clf.fit(X_emb, labels)
```

장점: 적은 데이터 (100-1000) 로 학습. 빠름.

### 1.4 생성 모델 Zero/Few-shot

```python
prompt = """Classify the sentiment as positive or negative.

Examples:
"Amazing!" -> positive
"Terrible." -> negative

Text: "I love this!"
Sentiment:"""

response = llm(prompt)
```

장점: 데이터 0 또는 매우 적은 양. 단점: 지연·비용 큼.

---

## 2. 데이터 양별 권장

```
+----------------+------------------+
| 데이터 양      | 권장 접근         |
+----------------+------------------+
| 0 (없음)       | 4. LLM Zero-shot|
| 10-100         | 4. LLM Few-shot |
| 100-1K         | 3. 임베딩 + LR  |
| 1K-10K         | 1 또는 3        |
| 10K+           | 1. BERT 미세조정 |
| 작업 특화 모델 있음| 2. 즉시 사용  |
+----------------+------------------+
```

데이터가 늘어나면 *작은 모델 미세조정* 이 더 효율적.

---

## 3. 비용·지연·정확도 비교

```
+----+--------+--------+--------+--------+
| 접근| 학습비| 추론비 | 지연   | 정확도 |
+----+--------+--------+--------+--------+
| 1  | 중간   | 매우 낮음| 매우 빠름| 높음   |
| 2  | 0      | 매우 낮음| 매우 빠름| 도메인 의존|
| 3  | 매우 낮음| 매우 낮음| 매우 빠름| 중간 |
| 4  | 0      | 높음    | 느림    | 매우 높음|
+----+--------+--------+--------+--------+
```

---

## 4. 같은 데이터셋 4 접근 비교 (IMDB)

```python
# 1. BERT 미세조정 → 정확도 92%
# 2. 작업 특화 (SST-2) → 정확도 88% (도메인 차이)
# 3. 임베딩 + LR → 정확도 87%
# 4. GPT-4 Zero-shot → 정확도 90%
# 4. GPT-4 Few-shot (10) → 정확도 92%

대규모 트래픽: 1 또는 3 추천 (비용)
프로토타입: 4 추천 (즉시)
```

---

## 5. 의사결정 매트릭스

```
도메인 데이터 충분?
   YES → 트래픽 큼?
            YES → 1. BERT 미세조정
            NO  → 4. LLM Few-shot
   NO  → 작업 특화 모델 있음?
            YES → 2. 즉시 사용
            NO  → 4. LLM Zero/Few-shot
```

---

## 권 정리

- 4 접근 = 표현 모델·작업 특화·임베딩·LLM
- 데이터 양이 적을수록 LLM, 많을수록 미세조정
- 트래픽 큰 프로덕션은 작은 모델 권장
- 의사결정 매트릭스로 30 초 결정

가장 기억할 한 줄: **"새 텍스트 분류 작업은 데이터 양과 트래픽으로 4 접근 중 하나가 자동 결정된다."**

다음 권: [Volume 103 — 텍스트 클러스터링·토픽 모델링](./volume_103_text_clustering.md)

---

## 자가점검 키워드

`BERT 분류`, `작업 특화`, `임베딩 + LR`, `LLM Few-shot`, `의사결정 매트릭스`

## 자가점검 질문

1. 4 접근의 비용·지연·정확도를 표로 적으십시오.
2. 데이터 양별 권장 접근을 적으십시오.
3. 의사결정 매트릭스를 그리십시오.

## 다음 권

[Volume 103 — 텍스트 클러스터링·토픽 모델링](./volume_103_text_clustering.md)
