# Volume 98 — PEFT — LoRA·QLoRA·Adapter·Prompt Tuning

> 이 권이 끝나면 *7B 모델을 24GB GPU 하나로 미세조정* 하는 방법을 코드 수준에서 설명할 수 있게 됩니다.

## 목적

PEFT (Parameter-Efficient Fine-Tuning) 는 *전체 모델의 0.1-1% 만 학습* 해 거의 같은 결과를 얻는 미세조정 기법입니다. LoRA·QLoRA·Adapter·Prompt Tuning 같은 변형이 *산업 표준 미세조정 도구* 가 되었습니다.

## 선수 지식

- Volume 9 (선형대수 2), 64 완료

## 학습 결과

1. LoRA 의 *낮은 랭크 분해* 발상을 이해합니다.
2. QLoRA 의 *4비트 양자화 + LoRA* 결합을 알 수 있습니다.
3. Adapter·Prompt Tuning 같은 다른 PEFT 변형을 안다.
4. PEFT 라이브러리로 미세조정 코드를 작성할 수 있습니다.

---

## 1. 표준 미세조정의 한계

70B 모델 전체 미세조정:
- 가중치 + 그래디언트 + AdamW 모멘트 = ~700GB
- H100 8 장 + FSDP 필요
- 학습 비용·시간 막대

작은 도메인 적응에 *과도한 비용*.

---

## 2. LoRA — 낮은 랭크 분해

### 2.1 발상

미세조정 시 가중치 변화량 ΔW 가 *낮은 랭크* 라고 가정. 두 작은 행렬의 곱으로 표현:

$$\Delta W = A \cdot B, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$$

r = 8, 16 같은 작은 값. 학습 파라미터 수가 *원래의 0.1-1%*.

### 2.2 PyTorch + PEFT

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4.2M || all params: 8.0B || trainable%: 0.05%
```

전체 파라미터의 *0.05%* 만 학습.

### 2.3 어댑터 합치기

추론 시 LoRA 어댑터를 *베이스 가중치에 합쳐* 추가 비용 없음:

```python
merged = model.merge_and_unload()
```

---

## 3. QLoRA — 4비트 양자화 + LoRA

### 3.1 발상

베이스 모델을 *4비트로 양자화* 해 메모리 1/4 → LoRA 학습 가능.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config,
)
model = get_peft_model(model, lora_config)
```

### 3.2 효과

70B 모델을 *48GB GPU 하나* 로 미세조정 가능 (이전엔 H100 다수 필요). 학습 정확도 *Full FT 와 거의 동등*.

산업의 *디팩토 표준 미세조정 방식*.

---

## 4. Adapter

### 4.1 발상

각 트랜스포머 층 사이에 *작은 Adapter 모듈* 삽입. Adapter 만 학습.

```
Layer N → [Adapter (D → r → D)] → Layer N+1
                  ↑ 학습 대상
```

LoRA 와 비슷한 효과. LoRA 가 더 인기 있어 점차 사용 줄어듦.

---

## 5. Prompt Tuning / Prefix Tuning

### 5.1 발상

모델 가중치는 동결. *학습 가능한 가상 토큰 임베딩* 만 학습. 프롬프트의 앞부분에 추가.

```
[학습된 가상 토큰들] + [실제 프롬프트] → LLM
```

장점: 매우 가벼움 (수 KB). 단점: 표현력 제한.

---

## 6. 비교

```
+--------------+-----------+--------+--------+
| 방법         | 학습 비율 | 효과   | 적용   |
+--------------+-----------+--------+--------+
| Full FT      | 100%      | 최고   | 무거움  |
| LoRA         | 0.1-1%    | 거의 같음| 표준   |
| QLoRA        | 0.1-1% (4bit)| 거의 같음 | 표준 |
| Adapter      | 0.5-2%    | 좋음   | 줄어듦 |
| Prompt Tuning| 0.01%     | 제한적 | 가벼운 적응|
+--------------+-----------+--------+--------+
```

---

## 7. 어댑터 관리

LoRA 어댑터는 *수 MB*. 같은 베이스 모델 + 다양한 어댑터 → *수많은 도메인 특화 모델* 가능.

```
base/llama-3-8B (16 GB)
  ├── adapter/medical (10 MB)
  ├── adapter/legal (10 MB)
  ├── adapter/code (10 MB)
  └── adapter/customer-support (10 MB)
```

런타임에 *어댑터 동적 교체* 가능 (vLLM 의 LoRA 지원).

---

## 권 정리

- LoRA = 낮은 랭크 분해, 0.1-1% 만 학습
- QLoRA = 4비트 양자화 + LoRA, 70B 를 1 GPU 로
- Adapter·Prompt Tuning = 다른 PEFT 변형
- 어댑터 합치기로 추론 비용 0
- 어댑터 동적 교체로 다도메인 운영

가장 기억할 한 줄: **"산업 미세조정의 표준은 QLoRA 이며, 70B 모델도 단일 GPU 로 미세조정 가능하게 만들었다."**

다음 권: [Volume 99 — 모델 병합과 멀티태스크](./volume_99_model_merging.md)

---

## 자가점검 키워드

`LoRA`, `QLoRA`, `Adapter`, `Prompt Tuning`, `BitsAndBytes`, `merge_and_unload`

## 자가점검 질문

1. LoRA 의 낮은 랭크 분해 식 ΔW = AB 의 의미를 설명하십시오.
2. QLoRA 가 70B 를 단일 GPU 로 미세조정 가능하게 만드는 메커니즘을 적으십시오.
3. PEFT 4 방법의 학습 비율과 적용 시점을 표로 정리하십시오.
4. 어댑터 동적 교체의 산업 응용을 적으십시오.

## 다음 권

[Volume 99 — 모델 병합과 멀티태스크](./volume_99_model_merging.md)
