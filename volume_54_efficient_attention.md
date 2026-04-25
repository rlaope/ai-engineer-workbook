# Volume 54 — 효율적 어텐션 변형

> 이 권이 끝나면 어텐션을 *근사하거나 재구성하는* 다양한 발상의 지도를 머릿속에 가지게 됩니다.

## 목적

표준 어텐션은 O(n²) 비용을 가지므로, 더 효율적인 변형이 다수 제안되었습니다. FlashAttention·Linear Attention·Performer·Multi-Query/Grouped-Query Attention 같은 기법이 대표적입니다.

## 선수 지식

- Volume 50, 52 완료

## 학습 결과

1. FlashAttention 의 *메모리 최적화* 발상을 이해합니다.
2. Linear Attention 의 *커널 트릭* 사용을 알 수 있습니다.
3. MQA·GQA 의 *KV 공유* 발상을 설명합니다.
4. 각 기법의 적용 시점을 알 수 있습니다.

---

## 1. FlashAttention

### 1.1 발상

표준 어텐션은 *큰 어텐션 행렬을 GPU 메모리에 저장* 해 메모리 대역폭이 병목.

FlashAttention (Tri Dao, 2022): *블록 단위로 SRAM 에서 처리* 해 큰 행렬을 한 번에 만들지 않음.

### 1.2 효과

- 메모리: O(n²) → O(n)
- 속도: 2-4 배 빠름
- *수학적으로는 표준 어텐션과 동등* (근사 아님)

PyTorch 2.0+ 에서 자동 사용 (`scaled_dot_product_attention`).

### 1.3 FlashAttention-2/3

후속 버전으로 더 빠름. H100 의 새 기능 (TMA·WGMMA) 활용.

---

## 2. Linear Attention

### 2.1 발상

소프트맥스를 *커널 함수* 로 근사하면 어텐션을 *선형 시간* 으로 계산 가능.

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot (\phi(K)^T \cdot V)$$

행렬 결합 순서를 바꾸면 O(n²) → O(n·d).

### 2.2 단점

근사이므로 *정확도 손실*. 표준 트랜스포머만큼 잘 학습되지 않음. 일부 응용에서만 사용.

대표: Performer, Linformer, Linear Transformer.

---

## 3. Multi-Query / Grouped-Query Attention

### 3.1 발상

표준 MHA 는 H 개 head 가 *각자 K·V* 를 가짐. *추론 시 KV 캐시* 가 H 배 크기.

MQA: 모든 head 가 *같은 K·V* 공유. KV 캐시 H 배 작음.
GQA: H 개 head 를 G 그룹으로 나누고 *각 그룹이 K·V 공유*. MQA 와 MHA 의 중간.

```
MHA: Q (H), K (H), V (H)
GQA: Q (H), K (G), V (G)   # G < H
MQA: Q (H), K (1), V (1)
```

장점: *KV 캐시 메모리 대폭 감소* → 큰 배치 가능.

LLaMA 2/3, Mistral 등이 GQA 채택.

---

## 4. Sparse Attention

Vol 52 (Long Context) 에서 다룬 LongFormer·BigBird 도 효율적 어텐션의 한 형태.

---

## 5. 어떤 것을 언제 쓰는가

```
+--------------+------------------+
| 상황         | 권장              |
+--------------+------------------+
| 표준 사용     | FlashAttention   |
| 추론 KV 절감  | GQA              |
| 매우 긴 시퀀스| Sparse Attention |
| 학술 실험     | Linear Attention |
+--------------+------------------+
```

대부분의 산업 시스템은 *FlashAttention + GQA* 가 기본.

---

## 권 정리

- FlashAttention = 메모리 최적화, 수학적으로 동등
- Linear Attention = 근사, 학술 위주
- MQA/GQA = KV 캐시 절감, 산업 표준
- Sparse Attention = 매우 긴 시퀀스용

가장 기억할 한 줄: **"FlashAttention 과 GQA 가 현대 LLM 어텐션의 두 표준이며, 둘은 메모리·계산을 다른 차원에서 줄인다."**

다음 권: [Volume 55 — 단어 임베딩](./volume_55_word_embedding.md)

---

## 자가점검 키워드

`FlashAttention`, `Linear Attention`, `MQA`, `GQA`, `KV 캐시`, `메모리 대역폭`

## 자가점검 질문

1. FlashAttention 의 *메모리 최적화* 메커니즘을 설명하십시오.
2. MQA·GQA·MHA 의 K·V 공유 정도를 표로 정리하십시오.
3. FlashAttention 과 Linear Attention 의 *근사 vs 정확* 차이를 설명하십시오.
4. 산업 LLM 의 표준 조합을 적으십시오.

## 다음 권

[Volume 55 — 단어 임베딩](./volume_55_word_embedding.md)
