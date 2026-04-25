# Volume 88 — Speculative Decoding

> 이 권이 끝나면 *작은 모델로 추측하고 큰 모델로 검증* 하는 발상이 어떻게 정확도 손실 없이 LLM 추론을 가속하는지 설명할 수 있게 됩니다.

## 목적

LLM 추론은 *순차적 (autoregressive)* 이라 본질적으로 느립니다. Speculative Decoding 은 *병렬화 가능한 검증* 으로 *2-4 배 가속* 을 달성합니다. Medusa·EAGLE·Lookahead Decoding 같은 변형이 표준입니다.

## 선수 지식

- Volume 67, 86 완료

## 학습 결과

1. Speculative Decoding 의 *추측 + 검증* 발상을 이해합니다.
2. Draft Model·Medusa·EAGLE 의 차이를 알 수 있습니다.
3. *기대 가속률* 을 계산할 수 있습니다.
4. vLLM·TensorRT-LLM 에서 활성화하는 방법을 안다.

---

## 1. 자기회귀의 한계

LLM 은 *한 토큰씩 순차 생성*. 병렬화 불가능 → GPU 가 *대부분 시간 idle*.

```
스텝 1: [past] → 토큰 1
스텝 2: [past, 1] → 토큰 2
스텝 3: [past, 1, 2] → 토큰 3
...
```

각 스텝에서 *한 토큰만 계산*. GPU 의 처리량 활용 매우 낮음.

---

## 2. Speculative Decoding

### 2.1 발상

작은 *Draft Model* 이 *N 개 토큰을 빠르게 추측* → 큰 *Verifier (Target) Model* 이 *그 N 개를 한 번에 검증*.

```
1. Draft: 빠르게 5 개 토큰 추측 → "the cat sat on the"
2. Verifier: 5 개를 동시 forward → 모두 맞으면 5 개 채택
            틀린 시점부터 다시
```

### 2.2 핵심: 병렬 검증

Verifier 가 5 개 토큰을 *한 번의 forward* 로 검증할 수 있어, *순차 5 번 호출 비용 ≈ 1 번 호출 비용*.

### 2.3 정확도 손실 없음

*수학적으로 증명* — 결과 분포가 *Verifier 의 분포와 동일*. 가속만 있고 품질 손실 없음.

---

## 3. 기대 가속률

```
가속률 = (수락된 토큰 수) / (Verifier 호출 수)

α = Draft 의 수락률 (0-1)
N = Draft 가 추측하는 토큰 수

기대 가속률 ≈ (1 - α^(N+1)) / (1 - α)
```

α=0.7, N=5 → 약 3 배 가속.

Draft 와 Verifier 가 비슷할수록 α 가 높아 가속이 큼.

---

## 4. Medusa

### 4.1 발상

별도 Draft Model 대신, *Verifier 모델에 추가 head* 를 학습. *Verifier 자체가 추측도 함*.

장점: 별도 모델 관리 불필요. 메모리 효율.

---

## 5. EAGLE

Medusa 의 후계자. *더 정확한 추측*. 더 높은 가속률.

---

## 6. Lookahead Decoding

*draft 모델 없이도 가속*. *과거 생성 패턴* 으로 추측. 임시 가속 효과.

---

## 7. vLLM 사용

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    speculative_model="meta-llama/Llama-3-8b-instruct",
    num_speculative_tokens=5,
)
```

Verifier (70B) + Draft (8B) → 정확도 70B + 가속 효과.

---

## 8. 함정

### 8.1 Draft 와 Verifier 의 어휘 일치

같은 토크나이저를 써야 함. 다른 모델 가족이면 어려움.

### 8.2 작은 배치에서 효과 큼

큰 배치에서는 *Verifier 가 이미 GPU 를 풀로 활용* 해 가속 효과 작음.

### 8.3 Draft 모델 호스팅 비용

GPU 메모리에 두 모델 모두 로드.

---

## 권 정리

- 자기회귀 = 순차, GPU idle
- Speculative = Draft (빠른 추측) + Verifier (병렬 검증)
- 정확도 손실 없음 (수학적 증명)
- Medusa·EAGLE = 별도 Draft 없이
- vLLM·TensorRT-LLM 에서 한 줄 활성화
- 작은 배치에서 효과 큼

가장 기억할 한 줄: **"Speculative Decoding 은 정확도 손실 없이 2-4 배 가속을 주며, 큰 모델 + 작은 배치에서 가장 효과적이다."**

다음 권: [Volume 89 — 분산 학습](./volume_89_distributed_training.md)

---

## 자가점검 키워드

`Draft Model`, `Verifier`, `병렬 검증`, `Medusa`, `EAGLE`, `Lookahead`, `수락률`

## 자가점검 질문

1. Speculative Decoding 의 *추측 + 검증* 흐름을 그리십시오.
2. 정확도 손실이 없는 이유를 설명하십시오.
3. α=0.8, N=4 일 때 기대 가속률을 계산하십시오.
4. Medusa 가 별도 Draft Model 을 대체하는 메커니즘을 적으십시오.

## 다음 권

[Volume 89 — 분산 학습](./volume_89_distributed_training.md)
