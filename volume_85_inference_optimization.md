# Volume 85 — 추론 최적화

> 이 권이 끝나면 새 모델을 받았을 때 *지연 시간 또는 비용을 절반으로 줄이기 위해 무엇을 시도할지* 의 우선순위 표가 머릿속에 그려지게 됩니다.

## 목적

LLM 추론은 *제품이 살아 있는 동안 영원히* 발생하는 비용입니다. 양자화·컴파일·KV 캐시 최적화·FlashAttention 같은 기법이 *같은 모델을 2-10 배 빠르게* 만듭니다.

## 선수 지식

- Volume 51, 84 완료

## 학습 결과

1. 추론 최적화 기법의 *우선순위 표* 를 갖게 됩니다.
2. 양자화 (PTQ·QAT·SmoothQuant·AWQ·GPTQ) 의 차이를 알 수 있습니다.
3. `torch.compile`·TensorRT 의 효과를 안다.
4. KV 캐시·FlashAttention·Speculative Decoding 의 결합 효과를 설명합니다.

---

## 1. 최적화 우선순위

```
+----+--------------------+--------+--------+
| 순위| 기법              | 효과   | 비용   |
+----+--------------------+--------+--------+
| 1  | 작은 모델로 교체   | 5-10x  | 정확도 |
| 2  | 양자화 (FP8/INT8)  | 2-3x   | 약간    |
| 3  | torch.compile/TRT  | 1.5-2x | 컴파일 시간 |
| 4  | FlashAttention     | 1.5-2x | 자동 (PyTorch 2+) |
| 5  | KV 캐시 (PagedAttn)| 메모리 | 구현 |
| 6  | Continuous Batching| 처리량 | vLLM 등 사용 |
| 7  | Speculative Decoding| 1.5-3x | 두 모델   |
| 8  | Tensor Parallelism | 큰 모델| 멀티 GPU |
+----+--------------------+--------+--------+
```

위에서 아래로 시도. *가장 큰 효과를 가장 작은 비용으로* 먼저.

---

## 2. 양자화

### 2.1 종류

- **PTQ (Post-Training)** — 학습 후 양자화. 가장 단순. 정확도 손실 약간.
- **QAT (Quantization-Aware Training)** — 학습 중 양자화 시뮬레이션. 정확도 손실 적음. 학습 비용.
- **SmoothQuant** — 활성화·가중치 분포를 *균등화* 후 양자화. 큰 활성화 outlier 처리.
- **AWQ (Activation-aware Weight Quantization)** — 중요한 가중치 (활성화가 큰 것에 대응) 만 보호.
- **GPTQ** — 행렬 분해 + 보정.

### 2.2 사용

```python
# AutoGPTQ 또는 bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b",
    quantization_config=bnb_config,
)
```

### 2.3 효과

- FP16 → INT8: 메모리 1/2, 속도 1.5-2x
- FP16 → INT4: 메모리 1/4, 속도 2-3x

정확도 손실은 보통 *1-2 %p* 이내.

---

## 3. torch.compile

PyTorch 2.0 부터. 모델을 *그래프로 컴파일* 해 *fused kernel* 생성.

```python
model = torch.compile(model)
```

한 줄로 1.5-2x 가속. 처음 호출 시 컴파일 시간 (수 분).

---

## 4. TensorRT-LLM

NVIDIA 의 LLM 전용 추론 엔진. *FP8 + 자동 그래프 최적화 + Custom Kernels*.

```python
# TensorRT-LLM 빌드
trtllm-build --checkpoint_dir ./llama-70b --output_dir ./engines
```

H100 에서 *최고 성능*. 단, *복잡한 빌드 절차* 와 *모델별 호환성* 도전.

---

## 5. FlashAttention

Vol 54 에서 다룸. PyTorch 2+ 에서 자동 사용.

---

## 6. KV 캐시 + PagedAttention

### 6.1 KV 캐시

자기회귀 생성 시 *과거 토큰의 K/V 를 재계산하지 않고 캐시*. 표준 최적화.

### 6.2 PagedAttention (vLLM)

KV 캐시를 *페이지 단위로 관리* — OS 의 가상 메모리와 비슷. *메모리 fragmentation 제거* 로 *3-5 배 처리량* 향상.

vLLM 이 표준.

---

## 7. Continuous Batching

기존 배칭은 *같은 길이의 요청만 묶음*. Continuous Batching 은 *다른 길이의 요청을 동적으로 묶고 풀음*. 처리량 *2-3 배* 향상.

vLLM, TGI, Triton 등이 지원.

---

## 8. Speculative Decoding

Vol 67 에서 다룸. *작은 모델 추측 + 큰 모델 검증* 으로 2-4x 가속.

---

## 9. Tensor Parallelism

큰 모델을 *여러 GPU 에 분산*. 70B 모델이 한 H100 에 안 들어가면 *2-4 GPU 에 텐서 병렬*.

vLLM, TensorRT-LLM 등이 지원.

```bash
vllm serve meta-llama/Llama-3-70b --tensor-parallel-size 4
```

---

## 권 정리

- 최적화 우선순위 = 작은 모델 → 양자화 → 컴파일 → 캐시 → 배칭 → 추측 → 분산
- 양자화 = FP16 → INT8 → INT4, 메모리·속도 절약
- torch.compile + FlashAttention = 자동 1.5-2x
- vLLM (PagedAttention + Continuous Batching) = 표준 서빙 도구
- 매우 큰 모델 = Tensor Parallelism

가장 기억할 한 줄: **"추론 최적화는 작은 모델 → 양자화 → 컴파일 → 캐시 → 배칭 의 순서로 시도하면 거의 항상 2-10 배 가속이 가능하다."**

다음 권: [Volume 86 — 추론 서빙 시스템](./volume_86_serving_systems.md)

---

## 자가점검 키워드

`PTQ/QAT`, `SmoothQuant/AWQ/GPTQ`, `torch.compile`, `TensorRT-LLM`, `PagedAttention`, `Continuous Batching`, `Tensor Parallel`

## 자가점검 질문

1. 추론 최적화 우선순위 7 가지를 적으십시오.
2. 4 가지 양자화 방식의 차이를 적으십시오.
3. PagedAttention 이 메모리 효율을 향상시키는 메커니즘을 설명하십시오.
4. Continuous Batching 의 발상을 적으십시오.

## 다음 권

[Volume 86 — 추론 서빙 시스템](./volume_86_serving_systems.md)
