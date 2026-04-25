# Volume 53 — Mixture of Experts (MoE)

> 이 권이 끝나면 *Mixtral 이 47B 파라미터인데 13B 처럼 빠른* 이유를 설명할 수 있게 됩니다.

## 목적

MoE (Mixture of Experts) 는 *모델의 일부만* 매 토큰에 활성화하는 구조입니다. *총 파라미터는 크지만 토큰당 실행 비용은 작은* 모델을 만들 수 있어, 큰 LLM 의 *효율 vs 품질 트레이드오프* 를 깨뜨립니다.

## 선수 지식

- Volume 51 완료

## 학습 결과

1. MoE 의 *Sparse Activation* 발상을 그릴 수 있습니다.
2. Router (Gating) 의 역할을 이해합니다.
3. Switch Transformer·Mixtral·DeepSeek 의 차이를 알 수 있습니다.
4. MoE 의 *학습 안정성·로드 밸런싱* 도전을 안다.

---

## 1. MoE 의 발상

### 1.1 표준 FFN vs MoE FFN

표준: 모든 토큰이 *같은 FFN* 통과.

MoE: *N 개의 expert FFN* 중 *Router 가 선택한 K 개* 만 통과.

```
입력 토큰 → Router → expert 1 (활성)
                  → expert 2 (활성)
                  → expert 3 (비활성)
                  → expert 4 (비활성)
                  → ...
                  → expert N (비활성)
                          ↓
                    선택된 expert 출력 가중합
```

### 1.2 효과

- *총 파라미터*: N 개 expert × 각 크기
- *토큰당 활성*: K 개 expert (보통 2 개)

Mixtral 8x7B = *총 47B*, 토큰당 활성 *13B*. 47B 모델의 능력 + 13B 모델의 추론 속도.

---

## 2. Router

```python
# 의사코드
def moe_layer(x, experts, router):
    # x: (B, L, D)
    logits = router(x)            # (B, L, N) — N 개 expert 점수
    top_k = logits.topk(k=2)      # 상위 2 개
    weights = softmax(top_k.values)
    
    output = 0
    for i in range(2):
        expert_idx = top_k.indices[..., i]
        expert_out = experts[expert_idx](x)
        output += weights[..., i:i+1] * expert_out
    return output
```

Router 는 *작은 Linear* 로 충분.

---

## 3. 모델 사례

### 3.1 Switch Transformer (2021)

Google 의 첫 대규모 MoE. *expert 1 개만 활성* 하는 단순한 변형.

### 3.2 Mixtral 8x7B (2023)

Mistral. *8 개 expert × 7B*, 토큰당 *2 개 활성*. 오픈 모델로 공개.

### 3.3 DeepSeek-V2/V3 (2024)

*수백 개 expert*. 정교한 라우팅. 비용 효율 높음.

### 3.4 GPT-4 추정

*MoE 라는 추측* 이 다수. 정확한 구조는 비공개.

---

## 4. 도전

### 4.1 로드 밸런싱

Router 가 *몇 개 expert 만 선호* 하면 다른 expert 는 학습 안 됨. *Auxiliary Loss* 로 균등 분배 유도.

### 4.2 학습 불안정성

Router 의 *비미분적 top-k 선택* 이 그래디언트 흐름을 어렵게.

### 4.3 추론 인프라 복잡

Expert 가 *여러 GPU 에 분산* 되어 있으면 *통신 비용* 발생. *Expert Parallelism* 같은 별도 분산 전략 필요.

---

## 권 정리

- MoE = Sparse Activation 으로 큰 모델·빠른 추론
- Router 가 K 개 expert 선택
- Switch (1 expert) → Mixtral (2/8) → DeepSeek (수백)
- 도전: 로드 밸런싱·학습 안정성·추론 인프라

가장 기억할 한 줄: **"MoE 는 모델의 일부만 활성화해 큰 능력 + 빠른 추론을 동시에 달성하는 구조이다."**

다음 권: [Volume 54 — 효율적 어텐션 변형](./volume_54_efficient_attention.md)

---

## 자가점검 키워드

`Sparse Activation`, `Router/Gating`, `Top-K`, `Switch Transformer`, `Mixtral`, `DeepSeek`, `Load Balancing`

## 자가점검 질문

1. MoE 의 *Sparse Activation* 발상을 한 문단으로 설명하십시오.
2. Mixtral 8x7B 의 총 파라미터와 토큰당 활성 파라미터를 적으십시오.
3. Router 의 역할과 단순한 구현을 적으십시오.
4. MoE 의 3 가지 도전을 나열하십시오.

## 다음 권

[Volume 54 — 효율적 어텐션 변형](./volume_54_efficient_attention.md)
