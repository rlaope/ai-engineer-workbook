# Volume 52 — Long Context 기법

> 이 권이 끝나면 *왜 트랜스포머가 긴 시퀀스에서 약한가* 와 *그 한계를 어떻게 우회하는가* 를 모두 설명할 수 있게 됩니다.

## 목적

표준 어텐션은 시퀀스 길이의 *제곱에 비례* 하는 시간·메모리를 사용합니다. 1 만 토큰 이상의 컨텍스트를 다루려면 새로운 발상이 필요합니다.

## 선수 지식

- Volume 50, 51 완료

## 학습 결과

1. 표준 어텐션의 O(n²) 복잡도를 시간·메모리 양쪽으로 설명할 수 있습니다.
2. Sliding Window·LongFormer·BigBird 의 희소 패턴을 그릴 수 있습니다.
3. Mamba/SSM 이 *RNN 의 부활* 인 이유를 알 수 있습니다.
4. *Lost in the Middle* 같은 Long Context 의 함정을 인식합니다.

---

## 1. O(n²) 의 본질

어텐션 행렬 $QK^T$ 는 (n × n). n=4096 이면 16M 원소, n=32768 이면 1G 원소.

시간: O(n²·d), 메모리: O(n²).

128K 컨텍스트는 *어텐션 행렬만 64GB* (FP16). H100 80GB 도 어려움.

---

## 2. 희소 어텐션

### 2.1 Sliding Window

각 토큰이 *주변 W 개* 만 봄. 복잡도 O(n·W).

### 2.2 LongFormer

Sliding Window + 일부 *Global Token* (모든 토큰과 연결).

### 2.3 BigBird

Random + Window + Global 의 조합. 이론적으로 *Full Attention 과 동등한 표현력*.

---

## 3. State Space Models (SSM) — Mamba

### 3.1 발상

RNN 의 *순차 계산* 한계를 *병렬화 가능한 형태* 로 재발명.

수학: 연속 시간 SSM 의 이산화. 선형 RNN 으로 표현 가능 → *FFT 로 병렬 계산*.

### 3.2 Mamba (2024)

선택적 SSM. *입력에 따라 상태 동학을 변경*. 트랜스포머와 비슷한 성능 + *선형 복잡도*.

`[VERIFY: Gu & Dao 2024, Mamba: Linear-Time Sequence Modeling with Selective State Spaces]`

### 3.3 RetNet·RWKV

다른 형태의 RNN-like 부활. 모두 *선형 복잡도 + 트랜스포머 성능* 추구.

---

## 4. Position Encoding 외삽

학습은 4K 컨텍스트로 했는데 *추론 시 32K* 같은 외삽이 필요할 때:

### 4.1 RoPE Scaling

RoPE 의 회전 주기를 *늘려* 더 긴 시퀀스에 적용.

기법: Linear Scaling, NTK-aware Scaling, YaRN.

### 4.2 ALiBi

위치 페널티 방식이라 *외삽이 자연스럽게 가능*.

---

## 5. 외부화 — RAG

긴 컨텍스트의 *대안* — 모델 안에 모든 정보를 넣지 말고 *외부 검색* 으로 필요한 부분만 컨텍스트에 주입.

장점: *무한대 정보* 다룰 수 있음. 단점: *검색 품질이 정확도 결정*.

상세는 Vol 66 (RAG) 에서.

---

## 6. Long Context 의 함정

### 6.1 Lost in the Middle

긴 컨텍스트의 *중간에 있는 정보를 모델이 잘 활용하지 못함* 현상 (Liu et al. 2023).

함의: *중요한 정보는 컨텍스트 시작 또는 끝에* 두는 프롬프트 전략이 효과적.

### 6.2 정보 희석

컨텍스트가 길어질수록 *각 토큰의 어텐션 가중치* 가 작아짐 → 신호 약화.

---

## 권 정리

- O(n²) = 트랜스포머의 본질적 한계
- 희소 어텐션 (Sliding/LongFormer/BigBird)
- SSM (Mamba/RetNet/RWKV) = RNN 부활
- RoPE Scaling·ALiBi = 위치 외삽
- RAG = 외부화 대안
- Lost in the Middle 같은 함정

가장 기억할 한 줄: **"Long Context 는 어텐션의 O(n²) 한계와 싸움이며, 희소 어텐션·SSM·RAG·외삽 기법으로 우회한다."**

다음 권: [Volume 53 — Mixture of Experts (MoE)](./volume_53_moe.md)

---

## 자가점검 키워드

`O(n²)`, `Sliding Window`, `LongFormer`, `Mamba/SSM`, `RoPE Scaling`, `Lost in the Middle`, `RAG`

## 자가점검 질문

1. 트랜스포머의 O(n²) 가 메모리에 미치는 영향을 128K 예시로 계산하십시오.
2. Sliding Window 와 BigBird 의 차이를 적으십시오.
3. Mamba 가 *RNN 의 부활* 이라 불리는 이유를 설명하십시오.
4. Lost in the Middle 의 실무적 함의를 적으십시오.

## 다음 권

[Volume 53 — Mixture of Experts (MoE)](./volume_53_moe.md)
