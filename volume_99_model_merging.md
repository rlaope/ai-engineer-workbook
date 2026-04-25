# Volume 99 — 모델 병합과 멀티태스크

> 이 권이 끝나면 *서로 다른 미세조정 모델을 합쳐 더 좋은 모델을 만든다* 는 발상의 원리와 한계를 알게 됩니다.

## 목적

같은 베이스 모델에서 미세조정한 *여러 모델의 가중치를 평균하거나 결합* 하면 *각자보다 더 좋은 모델* 이 만들어지는 경우가 있습니다. Model Soup·Task Arithmetic·SLERP 같은 기법이 이 영역의 표준입니다.

## 선수 지식

- Volume 9, 64, 98 완료

## 학습 결과

1. Model Soup 의 *가중치 평균* 발상을 이해합니다.
2. Task Arithmetic 의 *task vector* 개념을 알 수 있습니다.
3. SLERP·DARE·TIES 같은 병합 기법의 차이를 안다.
4. mergekit 라이브러리로 직접 병합할 수 있습니다.

---

## 1. Model Soup

### 1.1 발상

같은 베이스 + 다른 하이퍼파라미터로 학습한 *여러 모델 가중치를 단순 평균*. 평균 모델이 *각자보다 일반화 성능이 좋은* 경우가 흔함.

```python
def model_soup(models):
    avg = {}
    for key in models[0].state_dict():
        avg[key] = sum(m.state_dict()[key] for m in models) / len(models)
    return avg
```

`[VERIFY: Wortsman et al. 2022]`

### 1.2 동작 가설

다른 학습이 *다른 지역 최소값* 에 도달. 평균은 *그 지역들 사이의 평탄 지점* 으로 간주. *손실 풍경의 평탄 지역* 이 일반화에 좋음.

---

## 2. Task Arithmetic

### 2.1 발상

미세조정의 *변화량 (task vector)* 을 *벡터로 다룸*:

```
τ_task = θ_finetuned - θ_base
```

이 task vector 를 *더하거나 빼* 모델 능력 조작.

### 2.2 응용

```
θ_multitask = θ_base + τ_translation + τ_summarization
                     ← 두 작업 능력 모두 가짐

θ_unlearned = θ_base - τ_unwanted_behavior
            ← 특정 행동 제거
```

---

## 3. SLERP (Spherical Linear Interpolation)

두 모델의 가중치를 *구면 보간* — 단순 평균보다 *방향 보존*.

LLM 병합에서 자주 사용. mergekit 의 표준 옵션.

---

## 4. TIES·DARE

병합 시 *충돌하는 가중치 처리*:

- **TIES** — 부호 충돌 해결, 작은 변화는 무시
- **DARE** — 무작위 drop + rescale

복잡한 병합에서 *결과 품질* 향상.

---

## 5. mergekit 사용

오픈소스 모델 병합 도구.

```yaml
# merge.yml
slices:
  - sources:
      - model: meta-llama/Llama-3-8B-Instruct
        layer_range: [0, 32]
      - model: NousResearch/Nous-Hermes-Llama-3-8B
        layer_range: [0, 32]
merge_method: slerp
parameters:
  t:
    - filter: self_attn
      value: 0.5
    - filter: mlp
      value: 0.5
dtype: bfloat16
```

```bash
mergekit-yaml merge.yml output_path
```

Hugging Face 에 *수많은 병합 모델* 이 mergekit 으로 만들어짐.

---

## 6. 한계

- *베이스 모델이 같아야* — 다른 베이스는 병합 불가
- *과도한 병합은 성능 손실*
- *예측이 어려움* — 어떤 조합이 좋은지 실험 필요

---

## 권 정리

- Model Soup = 가중치 평균, 일반화 향상
- Task Arithmetic = task vector 산술
- SLERP = 구면 보간, LLM 표준
- TIES·DARE = 충돌 처리
- mergekit = 표준 도구
- 한계 = 같은 베이스, 실험적

가장 기억할 한 줄: **"모델 병합은 같은 베이스에서 미세조정된 모델들을 가중치 산술로 결합하는 기법이며, mergekit + SLERP 가 산업 표준이다."**

다음 권: [Volume 100 — 데이터셋 엔지니어링 종합](./volume_100_dataset_engineering.md)

---

## 자가점검 키워드

`Model Soup`, `Task Arithmetic`, `SLERP`, `TIES`, `DARE`, `mergekit`

## 자가점검 질문

1. Model Soup 의 동작 가설을 설명하십시오.
2. Task Arithmetic 의 task vector 산술 응용을 적으십시오.
3. SLERP 가 단순 평균보다 좋은 이유를 적으십시오.
4. mergekit 의 사용 흐름을 적으십시오.

## 다음 권

[Volume 100 — 데이터셋 엔지니어링 종합](./volume_100_dataset_engineering.md)
