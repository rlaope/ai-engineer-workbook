# Volume 67 — LLM 디코딩 알고리즘

> 이 권이 끝나면 같은 모델·같은 프롬프트로도 *temperature 0.7 vs 1.2* 가 만드는 출력 차이를 직관적으로 예측할 수 있게 됩니다.

## 목적

LLM 의 출력은 *모델* 뿐 아니라 *디코딩 알고리즘* 이 함께 결정합니다. Greedy·Beam·Sampling·Top-k·Top-p·Speculative 는 각자 다른 트레이드오프를 가집니다.

## 선수 지식

- Volume 51, 65 완료

## 학습 결과

1. Greedy·Beam·Sampling 의 차이를 한 그림으로 설명할 수 있습니다.
2. Top-k·Top-p (Nucleus) 의 차이를 알 수 있습니다.
3. Temperature 가 분포에 미치는 영향을 그래프로 그릴 수 있습니다.
4. Speculative Decoding 의 *추측 + 검증* 발상을 설명할 수 있습니다.

---

## 1. 언어 모델의 출력 분포

매 시점에서 LLM 은 *어휘의 모든 토큰에 대한 확률 분포* 를 출력합니다. 디코딩 알고리즘은 이 분포에서 *어떤 토큰을 선택할지* 결정합니다.

```
다음 토큰 분포:
"the" → 0.3
"a"   → 0.2
"an"  → 0.1
"this"→ 0.05
...
```

---

## 2. Greedy

매번 *확률이 가장 높은 토큰* 선택. 결정론적.

```python
next_token = logits.argmax()
```

장점: 단순·빠름·결정론적.
단점: *반복 패턴*·*창의성 결여*.

---

## 3. Beam Search

매 시점에서 *상위 N 개 후보 시퀀스* 유지. 끝까지 가장 높은 누적 확률 시퀀스 선택.

장점: Greedy 보다 더 좋은 시퀀스 발견.
단점: 비싼 계산. 응답이 *지나치게 안전·반복적*.

번역·요약에서 사용. 자유로운 생성에는 부적합.

---

## 4. Sampling

분포에서 *확률에 비례해 무작위 선택*.

```python
next_token = torch.multinomial(probs, num_samples=1)
```

장점: *다양성*. 단점: *낮은 확률 토큰까지 선택* 해 부적절한 출력 가능.

### 4.1 Top-K

상위 K 개 토큰만 후보로:

```python
top_k_logits, top_k_indices = logits.topk(k=50)
top_k_probs = top_k_logits.softmax()
selected = torch.multinomial(top_k_probs, 1)
next_token = top_k_indices[selected]
```

### 4.2 Top-P (Nucleus)

누적 확률 P 까지의 토큰만:

```python
sorted_probs, sorted_indices = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(0)
mask = cumsum <= 0.9
filtered = sorted_probs * mask
filtered /= filtered.sum()
```

Top-P 가 더 자연스러움 (분포의 모양에 적응).

---

## 5. Temperature

```python
adjusted_logits = logits / T
probs = adjusted_logits.softmax()
```

- T = 1 — 원래 분포
- T < 1 — *분포 sharpening*, 확실한 응답
- T > 1 — *분포 flattening*, 창의적

기본값 0.7 이 균형. 코드 생성은 0.0-0.3, 창작은 0.7-1.2.

---

## 6. Repetition / Frequency Penalty

같은 토큰의 *반복을 페널티*. 무한 반복 방지.

```python
for token in generated_so_far:
    logits[token] -= penalty
```

---

## 7. Speculative Decoding

### 7.1 발상

작은 *Draft 모델* 이 *N 개 토큰을 빠르게 추측* → 큰 *Verifier 모델* 이 *동시에 검증*. 맞으면 채택, 틀리면 다시.

장점: 같은 결과 + 2-4 배 빠름.

대표: Medusa, EAGLE, Lookahead Decoding.

### 7.2 vLLM 사용

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    speculative_model="meta-llama/Llama-3-8b-instruct",
)
```

`[VERIFY: vLLM 의 정확한 인자명]`

---

## 8. 권장 조합

```
+--------+-------------+----------+--------+
| 작업   | temperature | top_p    | k 또는 beam|
+--------+-------------+----------+--------+
| 번역   | 0.0         | -        | beam=4 |
| 요약   | 0.3         | -        | beam=4 |
| 코드   | 0.0-0.3     | 0.95     | -      |
| QA     | 0.3-0.7     | 0.9      | -      |
| 창작   | 0.7-1.2     | 0.95     | -      |
| 챗봇   | 0.7         | 0.9      | -      |
+--------+-------------+----------+--------+
```

---

## 권 정리

- Greedy = 결정론·반복
- Beam = 안전·번역
- Sampling + Top-K/P = 다양성
- Temperature = 확실성·창의성 조절
- Speculative Decoding = 같은 결과·2-4 배 빠름

가장 기억할 한 줄: **"디코딩은 모델만큼 출력을 결정하며, temperature·top_p 만 잘 골라도 응답 품질이 크게 달라진다."**

다음 권: [Volume 68 — 모델 사이즈 의사결정](./volume_68_model_sizing.md)

---

## 자가점검 키워드

`Greedy`, `Beam`, `Top-K`, `Top-P`, `Temperature`, `Speculative Decoding`

## 자가점검 질문

1. Greedy·Beam·Sampling 의 차이를 적으십시오.
2. Top-K 와 Top-P 의 차이를 설명하십시오.
3. Temperature 0 과 1 의 출력 차이를 설명하십시오.
4. Speculative Decoding 의 동작 원리를 한 문단으로 적으십시오.

## 다음 권

[Volume 68 — 모델 사이즈 의사결정](./volume_68_model_sizing.md)
