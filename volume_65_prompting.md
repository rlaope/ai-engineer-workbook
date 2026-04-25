# Volume 65 — 프롬프트와 In-Context Learning

> 이 권이 끝나면 *프롬프트는 새로운 프로그래밍 언어* 라는 명제의 의미를 정확히 이해하게 됩니다.

## 목적

LLM 의 동작은 프롬프트에 의해 크게 달라집니다. 프롬프트 엔지니어링은 *마법의 주문* 이 아니라 *모델의 컨텍스트 윈도우에 어떤 정보를 어떤 순서로 배치해 어떤 패턴을 유도할 것인가* 의 공학입니다. Few-shot·Chain-of-Thought·Self-Consistency·ReAct 같은 기법은 모두 *컨텍스트 조작* 의 변형입니다.

## 선수 지식

- Volume 51, 64 완료

## 학습 결과

1. Zero-shot·Few-shot·In-Context Learning 의 차이를 설명할 수 있습니다.
2. Chain-of-Thought 가 추론 성능을 올리는 메커니즘을 그릴 수 있습니다.
3. Self-Consistency·Tree-of-Thought·ReAct 의 사고 패턴을 구분합니다.
4. 생성 파라미터 (temperature·top-k·top-p) 가 출력에 미치는 영향을 알 수 있습니다.

---

## 1. 프롬프트 구조

### 1.1 System / User / Assistant

OpenAI·Anthropic 같은 API 의 표준 메시지 양식:

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Give an example."},
]
```

System 프롬프트가 *모델의 역할·제약·스타일* 을 정의.

---

## 2. Zero-shot vs Few-shot

### 2.1 Zero-shot

예시 없이 *직접 작업 지시*:

```
"Translate to French: Hello world."
→ "Bonjour le monde."
```

### 2.2 Few-shot

*몇 개의 예시* 를 함께 제공:

```
"English: Hello world. → French: Bonjour le monde.
English: Good morning. → French: Bonjour.
English: Thank you. → French: ?"
```

GPT-3 의 가장 충격적 발견: *몇 개의 예시만으로도 새 작업 학습*. 이를 *In-Context Learning* 이라 부릅니다.

---

## 3. Chain-of-Thought (CoT)

### 3.1 발상

복잡한 추론을 *단계별로 풀어쓰면* 정확도가 크게 향상됩니다.

```
질문: 23 × 17 은?

Zero-shot: "391"
CoT:      "23 × 17 = 23 × (10 + 7) = 230 + 161 = 391"
```

CoT 트리거: *"Let's think step by step."* 한 줄 추가만으로도 효과적 (Zero-shot CoT).

### 3.2 효과

수학·논리·다단계 추론에서 정확도 *수십 %p 개선*. GPT-4 의 강한 추론 능력의 한 원인.

---

## 4. Self-Consistency

CoT 응답을 *여러 번 (다른 random seed)* 생성한 뒤 *다수결*. 노이즈 평균화로 정확도 향상.

```python
answers = [llm(prompt, temperature=0.7) for _ in range(10)]
final = most_common(answers)
```

비용은 N 배지만 정확도 향상은 큼.

---

## 5. Tree-of-Thought

복잡한 문제를 *트리 구조로 탐색*. 각 단계에서 여러 후보 → 평가 → 가지치기.

```
[문제]
   ├── 접근 A
   │     ├── 단계 A1 (좋음, 계속)
   │     └── 단계 A2 (나쁨, 가지치기)
   └── 접근 B
         └── 단계 B1
```

장점: 매우 복잡한 추론 가능.
단점: 비싼 비용, 구현 복잡.

---

## 6. ReAct

*Reasoning + Acting* — 추론과 행동 (도구 호출) 을 *번갈아*.

```
Thought: 사용자가 날씨를 물었다. 검색 도구를 써야 한다.
Action: search("Seoul weather today")
Observation: "Sunny, 22°C"
Thought: 날씨 정보를 받았다. 응답을 만든다.
Final: "오늘 서울은 맑고 22°C 입니다."
```

에이전트 (Vol 72-77) 의 핵심 패턴.

---

## 7. 생성 파라미터

### 7.1 Temperature

- 0 — *결정론적*, 가장 확률 높은 토큰만
- 0.7 — 균형 (대부분 사용)
- 1.0+ — 창의적·예측 불가

### 7.2 Top-K

상위 K 개 토큰만 후보로.

### 7.3 Top-P (Nucleus)

누적 확률 P 까지의 토큰만 후보로.

### 7.4 Repetition Penalty

같은 토큰 반복 페널티.

```python
response = llm.complete(
    prompt,
    temperature=0.7,
    top_p=0.9,
    max_tokens=500,
    frequency_penalty=0.5,
)
```

---

## 권 정리

- System/User/Assistant = API 표준
- Zero-shot vs Few-shot (= In-Context Learning)
- CoT = 단계별 풀어쓰기로 추론 정확도 향상
- Self-Consistency = 다수결로 노이즈 제거
- Tree-of-Thought = 트리 탐색
- ReAct = 추론 + 도구 호출 (에이전트 패턴)
- 생성 파라미터 = temperature·top-p·penalty

가장 기억할 한 줄: **"프롬프트 엔지니어링은 컨텍스트 조작의 공학이며, CoT·Self-Consistency·ReAct 같은 패턴이 정확도를 결정한다."**

다음 권: [Volume 66 — RAG — 검색 증강 생성](./volume_66_rag.md)

---

## 자가점검 키워드

`Zero/Few-shot`, `In-Context Learning`, `CoT`, `Self-Consistency`, `Tree-of-Thought`, `ReAct`, `temperature`, `top-p`

## 자가점검 질문

1. Zero-shot 과 Few-shot 의 차이를 적으십시오.
2. Chain-of-Thought 가 추론 정확도를 향상시키는 메커니즘을 설명하십시오.
3. Self-Consistency 의 동작과 비용을 적으십시오.
4. ReAct 패턴을 자기 도메인 예시로 그리십시오.
5. temperature 0 과 0.7 의 출력 차이를 설명하십시오.

## 다음 권

[Volume 66 — RAG — 검색 증강 생성](./volume_66_rag.md)
