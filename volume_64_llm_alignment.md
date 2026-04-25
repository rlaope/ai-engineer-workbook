# Volume 64 — LLM 정렬

> 이 권이 끝나면 *왜 사전학습된 LLM 만으로는 챗봇을 만들 수 없는가* 를 설명할 수 있게 됩니다.

## 목적

사전학습된 LLM 은 *다음 토큰 예측* 만 학습되어, 사용자 의도에 맞는 응답을 잘 만들지 못합니다. *정렬 (Alignment)* 단계가 LLM 을 *유용한 어시스턴트* 로 변환합니다. SFT·RLHF·DPO·Constitutional AI 같은 기법을 다집니다.

## 선수 지식

- Volume 12, 63 완료

## 학습 결과

1. SFT (Supervised Fine-Tuning) 의 동작을 이해합니다.
2. RLHF 의 3 단계 (보상 모델 학습 → PPO 정렬 → 평가) 를 알 수 있습니다.
3. DPO 가 RLHF 를 *단순화* 한 방식임을 설명합니다.
4. Constitutional AI 의 발상을 알 수 있습니다.

---

## 1. 사전학습 모델의 한계

GPT-3 (사전학습만) 에 *프롬프트* "What is AI?" 를 주면:

- 가능한 응답 1: "What is AI? What is ML? What is DL?" (다음 질문 패턴)
- 가능한 응답 2: 백과사전 스타일 긴 글
- 가능한 응답 3: 무관한 텍스트

*어떤 응답을 줄지 통제 불가*. 이를 *어시스턴트* 로 만들려면 정렬 필요.

---

## 2. SFT (Supervised Fine-Tuning)

### 2.1 동작

*고품질 (질문, 답변) 쌍* 으로 사전학습 모델을 *추가 학습*. 손실은 사전학습과 같음 (다음 토큰 예측).

```
입력: "What is AI? Answer: "
정답: "AI is the field of building systems that..."
```

### 2.2 데이터 양

수천-수만 개의 *높은 품질* 쌍이면 충분. *양보다 질*.

대표 데이터셋: Alpaca, Dolly, OpenAssistant.

---

## 3. RLHF

### 3.1 3 단계

1. **SFT** — 위와 같음
2. **보상 모델 학습** — 인간이 *두 응답을 비교* 하는 데이터로 *보상 모델* 학습
3. **PPO 정렬** — 보상 모델로 *정책 (LLM) 을 강화학습*

```
사람: 응답 A 와 B 중 어느 것이 더 좋은가?
→ 응답 A
→ Reward Model 학습 데이터: (prompt, A, B, A 가 좋음)
```

### 3.2 PPO 의 핵심

LLM 이 *고보상 응답을 더 자주 생성* 하도록 학습. 단, *사전학습 분포에서 너무 멀어지지 않게* KL 페널티.

$$\max_\pi E[R(x)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$$

이 KL 페널티가 없으면 *Reward Hacking* — 모델이 보상만 노리는 이상한 응답을 만듦.

### 3.3 결과

ChatGPT (2022) 가 RLHF 의 첫 대중적 성공.

---

## 4. DPO (Direct Preference Optimization)

### 4.1 발상

RLHF 의 *PPO 단계가 복잡하고 불안정*. DPO 는 *PPO 없이 직접 선호 데이터로 정렬*.

수학적으로 RLHF 의 최적 정책을 *닫힌 형태* 로 표현. 학습이 *훨씬 단순·안정*.

### 4.2 사용

```python
from trl import DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,   # (prompt, chosen, rejected)
    tokenizer=tokenizer,
)
trainer.train()
```

LLaMA-3, Mistral 등 최신 모델이 DPO 또는 변형 사용.

---

## 5. Constitutional AI

Anthropic 의 발상. *사람의 비교 라벨 대신 AI 가 자기 응답을 비교*. *Constitution (헌법)* 으로 정의된 원칙에 따라.

장점: *인간 라벨링 비용 감소*, *원칙의 투명성*.

Claude 의 정렬 방식.

---

## 6. 정렬의 본질적 도전

- *Reward Hacking* — 보상 모델의 헛점을 노리는 응답
- *Sycophancy* — 사용자에게 동의만 하는 *비위 맞추기*
- *능력 vs 안전* — 안전을 강조하면 능력이 떨어짐
- *문화적 편향* — 어느 문화의 *좋음* 인가

이 도전들은 *현재 활발한 연구 영역*.

---

## 권 정리

- 사전학습 ≠ 어시스턴트 (정렬 필요)
- SFT = 고품질 (질문, 답변) 학습
- RLHF = SFT + 보상 모델 + PPO
- DPO = RLHF 의 단순화
- Constitutional AI = 인간 라벨 대체 (Claude)
- 도전: Reward Hacking·Sycophancy·문화적 편향

가장 기억할 한 줄: **"사전학습된 LLM 은 다음 토큰 예측기일 뿐이며, 정렬 단계가 그것을 어시스턴트로 변환한다."**

다음 권: [Volume 65 — 프롬프트와 In-Context Learning](./volume_65_prompting.md)

---

## 자가점검 키워드

`SFT`, `RLHF`, `보상 모델`, `PPO`, `DPO`, `Constitutional AI`, `Reward Hacking`

## 자가점검 질문

1. 사전학습 모델만으로 챗봇이 어려운 이유를 적으십시오.
2. RLHF 의 3 단계를 적으십시오.
3. DPO 가 RLHF 를 단순화한 점을 설명하십시오.
4. Reward Hacking 의 의미와 방어책 (KL 페널티) 을 적으십시오.

## 다음 권

[Volume 65 — 프롬프트와 In-Context Learning](./volume_65_prompting.md)
