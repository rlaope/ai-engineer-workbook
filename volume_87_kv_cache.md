# Volume 87 — KV 캐시 깊이

> 이 권이 끝나면 *왜 LLM 추론에서 KV 캐시가 메모리의 가장 큰 부분이 되는가* 와 *그 메모리를 줄이는 모든 기법* 을 알게 됩니다.

## 목적

자기회귀 LLM 추론에서 *KV 캐시* 는 매 시점의 *Key·Value 행렬* 을 저장합니다. 시퀀스가 길어지고 배치가 커질수록 KV 캐시가 *모델 가중치보다 큰 메모리* 를 차지하게 됩니다. PagedAttention·MQA/GQA·KV 압축 같은 기법이 이 문제를 해결합니다.

## 선수 지식

- Volume 50, 54, 84, 85 완료

## 학습 결과

1. KV 캐시 메모리 계산을 손으로 할 수 있습니다.
2. PagedAttention 의 *가상 메모리 발상* 을 이해합니다.
3. MQA/GQA 가 KV 캐시를 줄이는 메커니즘을 안다.
4. KV 압축·prefix 캐시 같은 추가 기법을 설명합니다.

---

## 1. KV 캐시의 정체

### 1.1 자기회귀 추론

토큰을 *하나씩 생성* 할 때, 매 시점에서 *모든 과거 토큰의 K·V* 가 필요. 매번 다시 계산하면 비효율 → *캐시*.

### 1.2 메모리 계산

```
KV 캐시 = 2 × layers × hidden_dim × seq_len × batch × bytes
        ↑                                                  ↑
    K and V                                          FP16=2

LLaMA 3 8B:
2 × 32 × 4096 × 2048 × 1 × 2 = 1 GB (배치 1, seq 2048)
2 × 32 × 4096 × 8192 × 16 × 2 = 64 GB (배치 16, seq 8192)
```

배치·시퀀스가 커지면 *모델 가중치 (16 GB) 보다 KV 캐시가 큼*.

---

## 2. PagedAttention (vLLM)

### 2.1 문제

기존 KV 캐시는 *연속된 메모리 블록*. 다양한 길이의 요청이 섞이면 *fragmentation*.

### 2.2 발상

OS 의 *가상 메모리·페이징* 적용. KV 캐시를 *고정 크기 페이지 (예: 16 토큰)* 로 나누고 *논리 페이지 → 물리 페이지* 매핑.

### 2.3 효과

- 메모리 fragmentation 제거
- *3-5 배 처리량* 향상
- 다양한 길이의 요청 동시 처리 가능

vLLM 의 핵심 기능.

---

## 3. MQA / GQA

Vol 54 에서 다룸. KV 를 *모든 head 가 공유* 또는 *그룹별로 공유* 해 KV 캐시 H 배 감소.

LLaMA 2/3, Mistral 등 표준 채택.

---

## 4. KV 압축

### 4.1 양자화

KV 캐시도 *INT8/FP8 로 양자화* 가능. 메모리 1/2-1/4.

### 4.2 Sliding Window

오래된 KV 를 *버림*. 최근 N 개만 유지. 컨텍스트 일부 손실.

### 4.3 Selective KV (H2O 등)

*중요한 토큰의 KV 만 보존*. 어텐션 점수 기반 선택.

---

## 5. Prefix Cache

여러 요청이 *같은 시스템 프롬프트* 를 공유하면, *그 부분의 KV 를 재사용*.

```
요청 1: [시스템 프롬프트 1000 토큰] + [질문 A]
요청 2: [시스템 프롬프트 1000 토큰] + [질문 B]
       → 1000 토큰 KV 재사용 → 큰 비용 절감
```

OpenAI·Anthropic 의 *프롬프트 캐싱* API 가 이 패턴.

---

## 6. Multi-Query Attention 의 다른 변형

- **Grouped-Query Attention (GQA)** — 가장 흔한 절충
- **Multi-Latent Attention (DeepSeek-V2)** — KV 를 *압축된 잠재* 로 유지
- **Sliding Window + Sink** — Mistral 의 패턴

---

## 권 정리

- KV 캐시 = 2·L·D·seq·B·bytes (긴 시퀀스에서 폭발)
- PagedAttention = 가상 메모리 발상, fragmentation 제거
- MQA/GQA = 헤드 간 KV 공유
- 양자화·Sliding Window·Selective = 추가 압축
- Prefix Cache = 공통 프롬프트 재사용

가장 기억할 한 줄: **"긴 컨텍스트·큰 배치에서 KV 캐시가 모델 가중치보다 큰 메모리를 잡아먹으며, PagedAttention + GQA + Prefix Cache 가 표준 해결책이다."**

다음 권: [Volume 88 — Speculative Decoding](./volume_88_speculative.md)

---

## 자가점검 키워드

`KV 캐시`, `PagedAttention`, `MQA/GQA`, `KV 양자화`, `Prefix Cache`, `MLA`

## 자가점검 질문

1. LLaMA 3 8B 의 *batch 16, seq 8192* KV 캐시 크기를 계산하십시오.
2. PagedAttention 의 가상 메모리 발상을 설명하십시오.
3. MQA·GQA·MHA 의 KV 캐시 크기를 비교하십시오.
4. Prefix Cache 가 가장 큰 효과를 내는 응용을 적으십시오.

## 다음 권

[Volume 88 — Speculative Decoding](./volume_88_speculative.md)
