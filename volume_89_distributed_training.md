# Volume 89 — 분산 학습

> 이 권이 끝나면 *왜 70B 모델 학습에 GPU 한 장이 아니라 수천 장이 필요한가* 와 *그 수천 장이 어떻게 협업하는가* 를 그림으로 그릴 수 있게 됩니다.

## 목적

대형 모델은 한 GPU 메모리에 들어가지 않습니다. *데이터·모델 자체·옵티마이저 상태* 를 여러 GPU 에 분산해야 하며, 이것이 데이터 병렬·텐서 병렬·파이프라인 병렬·시퀀스 병렬·ZeRO 의 세계입니다.

## 선수 지식

- Volume 36, 84, 85 완료

## 학습 결과

1. DDP·FSDP·ZeRO 의 차이를 메모리 관점에서 설명할 수 있습니다.
2. 텐서 병렬과 파이프라인 병렬의 통신 패턴을 그릴 수 있습니다.
3. 3D 병렬 (DP × TP × PP) 의 결합 발상을 이해합니다.
4. NCCL·AllReduce·AllGather 의 의미를 알 수 있습니다.

---

## 1. 분산 학습의 동기

### 1.1 메모리 한계

```
70B 모델 학습 메모리 (FP16, AdamW):
- 가중치: 140 GB
- 그래디언트: 140 GB
- AdamW 모멘트 1·2: 560 GB
- 활성화 (배치·시퀀스 따라): 100 GB+
합계: ~940 GB
```

H100 80GB 한 장으로 *불가능*. 분산 필수.

### 1.2 시간 한계

70B 학습 = 수조 토큰. 한 GPU 로 *수년*. 수천 GPU 로 *수개월*.

---

## 2. Data Parallel (DP / DDP)

### 2.1 발상

각 GPU 가 *모델 전체 복사본* + *다른 데이터*. 그래디언트를 *AllReduce* 로 평균.

```
GPU 0: 데이터 1-32 → 손실·그래디언트
GPU 1: 데이터 33-64 → 손실·그래디언트
...
            ↓
        AllReduce (그래디언트 평균)
            ↓
       각 GPU 가 같은 갱신
```

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

장점: 단순. 단점: *각 GPU 가 모델 전체* — 큰 모델은 안 됨.

---

## 3. FSDP (Fully Sharded Data Parallel)

### 3.1 발상

DDP 의 단점 (모델 전체 복사) 해결. 모델 가중치·그래디언트·옵티마이저 상태를 *각 GPU 에 분할 저장*. forward/backward 시 *필요할 때만 모음*.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

장점: *큰 모델 학습 가능*. 단점: *통신 비용 증가*.

---

## 4. ZeRO

DeepSpeed 의 분산 학습. 3 단계:

- **ZeRO-1** — 옵티마이저 상태만 분할
- **ZeRO-2** — 옵티마이저 상태 + 그래디언트 분할
- **ZeRO-3** — 가중치까지 분할 (FSDP 와 거의 동일)

PyTorch FSDP ≈ DeepSpeed ZeRO-3.

---

## 5. Tensor Parallel (TP)

### 5.1 발상

한 *층의 행렬* 을 *여러 GPU 에 분할*. 한 forward 가 GPU 간 *AllReduce 통신* 필요.

```
Linear 층 (D=8192) → 4 GPU 분할
GPU 0: D=2048 부분
GPU 1: D=2048 부분
...
```

장점: 매우 큰 층도 처리. 단점: GPU 간 *고대역 (NVLink)* 필수.

Megatron-LM 이 표준 구현.

---

## 6. Pipeline Parallel (PP)

### 6.1 발상

*층을 GPU 에 분할*. 데이터가 *파이프라인처럼* 흐름.

```
GPU 0: 1-20 층
GPU 1: 21-40 층
GPU 2: 41-60 층
GPU 3: 61-80 층

데이터 → GPU 0 → GPU 1 → GPU 2 → GPU 3
```

도전: *Pipeline Bubble* — GPU 가 다른 GPU 를 기다림. 마이크로 배치로 완화.

---

## 7. 3D 병렬

대형 학습에서는 *세 가지를 모두 결합*:

```
DP × TP × PP

예: 1024 GPU = 4 (DP) × 8 (TP) × 32 (PP)
```

각 차원이 *다른 도전을 해결*.

---

## 8. NCCL 통신 원시연산

### 8.1 표준 연산

- **AllReduce** — 모두가 자기 값 합쳐 결과를 모두에게
- **AllGather** — 모두가 데이터를 모아 모두에게
- **Broadcast** — 한 GPU 가 값을 모두에게
- **ReduceScatter** — 합친 결과를 분할

### 8.2 통신 인프라

- *같은 노드 내 GPU* — NVLink (수백 GB/s)
- *노드 간* — InfiniBand (400 Gbps)

이 인프라가 *분산 학습 효율* 의 결정적 요인.

---

## 9. Activation Checkpointing

활성화 메모리를 *저장 안 하고 재계산*. 메모리 30-50% 절감, 시간 20% 증가.

```python
from torch.utils.checkpoint import checkpoint

def forward(x):
    return checkpoint(layer, x)
```

대형 모델 학습의 표준.

---

## 권 정리

- DDP = 데이터 병렬, 모델 전체 복사
- FSDP/ZeRO-3 = 가중치까지 분할
- TP = 행렬 분할, NVLink 필요
- PP = 층 분할, Pipeline Bubble 도전
- 3D 병렬 = DP × TP × PP
- NCCL AllReduce·AllGather 가 통신 표준
- Activation Checkpointing = 메모리·시간 trade

가장 기억할 한 줄: **"대형 모델 학습은 데이터·텐서·파이프라인 병렬의 3D 결합으로 이뤄지며, GPU 간 통신 인프라가 효율의 결정자이다."**

다음 권: [Volume 90 — AI 가속기 비교](./volume_90_accelerators.md)

---

## 자가점검 키워드

`DDP`, `FSDP`, `ZeRO`, `TP`, `PP`, `3D Parallelism`, `NCCL`, `AllReduce`, `Activation Checkpointing`

## 자가점검 질문

1. 70B 모델 학습 메모리를 손으로 계산하십시오.
2. DDP 와 FSDP 의 차이를 적으십시오.
3. TP 와 PP 의 차이를 통신 패턴 관점에서 설명하십시오.
4. 3D 병렬의 *DP × TP × PP* 분배의 합리적 비율을 적으십시오.

## 다음 권

[Volume 90 — AI 가속기 비교](./volume_90_accelerators.md)
