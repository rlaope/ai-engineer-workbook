# Volume 71 — LLM-as-Judge 평가 자동화

> 이 권이 끝나면 *모델이 모델을 평가한다* 는 패턴의 강점과 함정을 모두 알게 됩니다.

## 목적

LLM 응답을 *자동 평가* 하는 표준 방식이 *LLM-as-Judge* 입니다. GPT-4·Claude 같은 강한 모델로 *다른 모델의 응답을 평가* 하면 *인간 평가의 80-90% 정확도* 를 *훨씬 낮은 비용* 으로 달성합니다.

## 선수 지식

- Volume 38, 64, 65 완료

## 학습 결과

1. LLM-as-Judge 의 표준 패턴을 적용할 수 있습니다.
2. 인간 평가와의 *상관성* 을 측정할 수 있습니다.
3. 평가의 *편향과 함정* (Position Bias·Length Bias) 을 안다.
4. Pairwise·Single·Reference 평가의 차이를 적용합니다.

---

## 1. 평가 종류

### 1.1 Single Evaluation

한 응답에 *점수 (1-5)* 또는 *Pass/Fail*.

```python
prompt = f"""Rate the following answer 1-5 for accuracy.
Question: {question}
Answer: {answer}
Score:"""
score = int(llm(prompt).strip())
```

장점: 단순. 단점: *기준이 모호*.

### 1.2 Pairwise Evaluation

두 응답을 *비교* 해 어느 것이 좋은가 결정.

```python
prompt = f"""Compare two answers. Which is better?
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Verdict: A or B"""
```

장점: *상대적 판단이 더 정확*. 단점: 두 배 비용.

### 1.3 Reference-based

*정답 (reference) 과 비교*.

```python
prompt = f"""Compare answer to reference. Score 1-5.
Reference: {reference}
Answer: {answer}"""
```

가장 정확. 단점: *reference 작성 비용*.

---

## 2. 표준 평가 프레임

```python
def llm_judge(question, answer, criteria):
    prompt = f"""You are an expert evaluator.

Criteria: {criteria}

Question: {question}
Answer: {answer}

Provide:
1. Score (1-5)
2. Reasoning (1 sentence)

Format:
Score: <number>
Reasoning: <text>"""
    
    response = llm(prompt, model="gpt-4o", temperature=0)
    return parse_score(response)
```

`temperature=0` 이 결정론적 평가에 권장.

---

## 3. 인간 평가와의 상관성

LLM-as-Judge 의 정확도를 *인간 평가와의 상관* 으로 측정:

```python
# 100 개 샘플에 인간·LLM 평가 둘 다
human_scores = [...]
llm_scores = [llm_judge(q, a) for q, a in samples]

correlation = np.corrcoef(human_scores, llm_scores)[0, 1]
print(f"Pearson correlation: {correlation}")
```

GPT-4 의 LLM-as-Judge 는 *대부분 도메인에서 r=0.8+* 를 달성.

---

## 4. 편향과 함정

### 4.1 Position Bias

Pairwise 평가에서 *첫 번째 응답을 더 높게* 평가하는 경향.

방어: *두 순서로 평가하고 평균*.

### 4.2 Length Bias

*긴 응답을 더 좋게* 평가하는 경향.

방어: 평가 프롬프트에 *간결성도 중요* 명시.

### 4.3 Self-Preference Bias

같은 모델 가족 (GPT 가 GPT 응답) 을 *더 높게* 평가하는 경향.

방어: *다른 모델 가족 (예: Claude)* 으로 GPT 응답 평가.

### 4.4 Verbosity Bias

평가자도 LLM 이라 *지시 없이는 형식이 자유로워* 파싱 어려움.

방어: 명시적 출력 형식 강제 (JSON Mode).

---

## 5. Arena 패턴

여러 모델의 출력을 *Pairwise 평가* 로 *Elo 레이팅* 산출. LMSYS Chatbot Arena 가 표준.

```python
# 두 모델 비교
result = pairwise_judge(question, model_a_answer, model_b_answer)
update_elo(model_a, model_b, result)
```

수천 비교가 쌓이면 *모델 순위* 가 명확해짐.

---

## 6. 운영 팁

- *평가 모델은 평가받는 모델보다 강해야* (GPT-4 로 GPT-3.5 평가)
- *평가 셋 100-1000 개* 면 통계적으로 의미 있음
- *주간 단위 자동 실행* 으로 회귀 감지
- *인간 평가 100 개* 와 정기 상관 측정

---

## 권 정리

- LLM-as-Judge = 인간 평가의 80-90% 정확도를 1/10 비용에
- Single·Pairwise·Reference 의 3 종류
- temperature=0 + 명시적 형식 권장
- Position·Length·Self-Preference·Verbosity 편향 인식
- Arena 패턴으로 모델 순위
- 인간 평가와의 정기 상관 검증

가장 기억할 한 줄: **"LLM-as-Judge 는 인간 평가의 자동화 대안이며, 편향을 인식한 채 사용하면 평가 비용을 1/10 로 줄인다."**

다음 권: [Volume 72 — 에이전트의 본질](./volume_72_agent_essence.md)

---

## 자가점검 키워드

`Single/Pairwise/Reference`, `Position Bias`, `Length Bias`, `Self-Preference`, `Arena`, `Elo`

## 자가점검 질문

1. 3 가지 평가 방식 (Single/Pairwise/Reference) 의 트레이드오프를 적으십시오.
2. 인간 평가와의 상관성 측정 방법을 적으십시오.
3. LLM-as-Judge 의 4 가지 편향과 방어책을 표로 정리하십시오.
4. Arena 패턴이 모델 순위를 만드는 메커니즘을 설명하십시오.

## 다음 권

[Volume 72 — 에이전트의 본질](./volume_72_agent_essence.md)
