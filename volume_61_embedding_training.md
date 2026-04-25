# Volume 61 — 임베딩 모델 직접 학습

> 이 권이 끝나면 자기 도메인에 특화된 임베딩 모델을 처음부터 학습·평가·서빙할 수 있게 됩니다.

## 목적

공개 임베딩 모델로 *80% 의 정확도* 가 나오면 충분한 경우가 많지만, *나머지 20%* 가 비즈니스 결정에 중요하다면 *자기 도메인 미세조정* 이 필요합니다. 이 권은 대조 학습으로 임베딩 모델을 직접 학습하는 흐름을 다룹니다.

## 선수 지식

- Volume 56, 59 완료

## 학습 결과

1. *대조 학습 (Contrastive Loss)* 의 동작을 이해합니다.
2. (anchor, positive, negative) 쌍을 만들 수 있습니다.
3. Hard Negative Mining 이 왜 중요한지 알 수 있습니다.
4. PyTorch + sentence-transformers 로 fine-tune 할 수 있습니다.

---

## 1. 대조 학습

### 1.1 발상

같은 의미의 두 텍스트는 *임베딩이 가깝게*, 다른 의미는 *멀게*.

손실 함수 (InfoNCE):

$$L = -\log \frac{\exp(s(a, p)/\tau)}{\sum_{n} \exp(s(a, n)/\tau)}$$

- $a$: anchor
- $p$: positive (anchor 와 같은 의미)
- $n$: negatives (다른 의미)
- $\tau$: temperature

### 1.2 데이터 형태

```
(anchor, positive, negative1, negative2, ...)

예시 — 검색 도메인:
anchor:    "What is machine learning?"
positive:  "ML is a subset of AI..."
negatives: ["Sports news today", "Recipe for pasta", ...]
```

---

## 2. 데이터 구축

### 2.1 양성 쌍의 출처

- 사용자 검색 로그: (쿼리, 클릭한 문서)
- FAQ: (질문, 답변)
- 라벨링: 도메인 전문가가 (질문, 적합 문서) 작성
- LLM 생성: GPT 로 (문서 → 합성 질문) 생성

### 2.2 음성 쌍의 출처

- 무작위 다른 문서
- *Hard Negative* — 의미가 비슷하지만 정답이 아닌 문서

Hard Negative 가 *학습 효과를 크게 올림*.

### 2.3 Hard Negative Mining

```python
# 1. 베이스 모델로 모든 anchor 의 상위 K 결과 미리 계산
# 2. 정답이 아닌 상위 결과를 hard negative 로 선택

for anchor, positive in train_data:
    candidates = top_k_search(anchor, k=20)
    hard_negatives = [c for c in candidates if c != positive][:5]
```

---

## 3. PyTorch 로 학습

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 베이스 모델
model = SentenceTransformer('all-MiniLM-L6-v2')

# 학습 데이터
train_examples = [
    InputExample(texts=[anchor, positive])
    for anchor, positive in pairs
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Multiple Negatives Ranking Loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# 학습
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./my-embedder',
)
```

`MultipleNegativesRankingLoss` 는 *배치 안의 다른 샘플* 을 negatives 로 자동 사용.

---

## 4. 평가

학습 후 자기 도메인 평가 셋 (Vol 59) 으로 *Recall@K, MRR* 측정. 베이스 모델보다 *유의미한 향상* 이 있어야 채택.

---

## 5. 서빙

학습된 모델을 표준 sentence-transformers 인터페이스로 서빙:

```python
model = SentenceTransformer('./my-embedder')
embeddings = model.encode(documents)
```

또는 ONNX·TensorRT 로 변환해 *추론 속도* 가속.

---

## 권 정리

- 대조 학습 = anchor·positive·negatives
- 데이터: 검색 로그·FAQ·라벨링·LLM 합성
- Hard Negative Mining 이 핵심
- sentence-transformers 의 MultipleNegativesRankingLoss 가 표준
- 자기 도메인 평가로 검증 후 채택

가장 기억할 한 줄: **"공개 모델의 80% 가 충분하지 않다면 자기 데이터로 fine-tune — Hard Negative 가 학습 효과를 결정한다."**

다음 권: [Volume 62 — 토크나이저](./volume_62_tokenizer.md)

---

## 자가점검 키워드

`대조 학습`, `InfoNCE`, `Hard Negative Mining`, `MultipleNegativesRankingLoss`, `합성 데이터`

## 자가점검 질문

1. 대조 학습의 손실 함수를 적으십시오.
2. (anchor, positive) 쌍의 출처 4 가지를 적으십시오.
3. Hard Negative Mining 의 의미와 메커니즘을 설명하십시오.
4. sentence-transformers 로 fine-tune 하는 5 줄 코드를 적으십시오.

## 다음 권

[Volume 62 — 토크나이저](./volume_62_tokenizer.md)
