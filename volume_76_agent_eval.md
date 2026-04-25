# Volume 76 — Agent 평가와 벤치마크

> 이 권이 끝나면 *우리 에이전트가 잘 동작하는가* 라는 모호한 질문을 *어떤 작업·어떤 메트릭으로 측정할 것인가* 로 좁힐 수 있게 됩니다.

## 목적

에이전트 평가는 LLM 단독 평가보다 어렵습니다. *최종 결과만이 아니라 도구 사용·중간 단계·비용* 까지 평가해야 합니다. GAIA·SWE-Bench·OSWorld·WebArena 같은 표준 벤치마크와 자기 도메인 평가법을 다집니다.

## 선수 지식

- Volume 71, 72 완료

## 학습 결과

1. 에이전트 평가의 *4 가지 차원* 을 알 수 있습니다.
2. GAIA·SWE-Bench·OSWorld·WebArena 의 차이를 알 수 있습니다.
3. 자기 에이전트의 평가 셋을 설계할 수 있습니다.
4. *Trace 분석* 으로 실패 원인을 좁힐 수 있습니다.

---

## 1. 평가의 4 차원

```
1. 정확도 (Accuracy)   — 최종 답이 맞는가
2. 효율 (Efficiency)   — 도구 호출 수, 비용
3. 안전 (Safety)       — 위험한 행동을 했는가
4. 견고성 (Robustness) — 도구 실패·노이즈에 강한가
```

각 차원이 다른 메트릭 필요.

---

## 2. 표준 벤치마크

### 2.1 GAIA (Meta, 2023)

*General AI Assistants* 벤치마크. 466 개 질문, 도구·웹 검색·다단계 추론 필요.

GPT-4 + 도구가 *약 30% 정확도* (인간 92%).

### 2.2 SWE-Bench (2023)

*GitHub Issue 를 코드 수정으로 해결*. 실제 OSS 프로젝트의 수천 개 이슈.

Claude Sonnet 4 가 *80%+* 달성 (2025 시점).

### 2.3 OSWorld

*실제 OS (Linux·MacOS·Windows) 에서의 작업 수행*. Computer Use 평가.

### 2.4 WebArena

*실제 웹사이트 (E-Commerce·CMS 등) 에서의 작업*. Browser Agent 평가.

---

## 3. 자기 도메인 평가

### 3.1 평가 셋 구축

```python
eval_set = [
    {
        "task": "고객의 환불 요청 처리",
        "context": {"order_id": 123, "reason": "defective"},
        "success_criteria": [
            "refund_initiated",
            "customer_notified",
            "ticket_closed",
        ],
    },
    ...
]
```

성공 기준을 *명시적·자동 검증 가능* 하게.

### 3.2 평가 흐름

```python
for task in eval_set:
    trace = agent.run(task["task"], task["context"])
    score = check_criteria(trace, task["success_criteria"])
    metrics.append({
        "score": score,
        "tool_calls": len(trace.tool_calls),
        "cost": trace.total_cost,
        "latency": trace.duration,
    })
```

---

## 4. Trace 분석

에이전트의 *모든 단계 (Thought·Action·Observation)* 를 기록한 *Trace* 를 분석해 실패 원인 좁힘.

```
Trace:
1. Thought: "검색이 필요하다"
2. Action: search(query="customer refund policy")
3. Observation: "[]"
4. Thought: "결과가 없다. 다른 쿼리 시도."
5. Action: search(query="refund process")
6. ...
```

실패 패턴:
- *잘못된 도구 선택*
- *잘못된 인자* (검색어 오타 등)
- *결과 오해* (LLM 이 빈 결과를 정상으로 해석)

---

## 5. 도구

- **Helicone** — LLM 호출 로깅·분석
- **Langfuse** — 에이전트 trace 시각화
- **Braintrust** — 평가 + Trace 통합

---

## 권 정리

- 4 평가 차원: 정확도·효율·안전·견고성
- 표준 벤치마크: GAIA·SWE-Bench·OSWorld·WebArena
- 자기 도메인 평가 셋 + 자동 성공 기준
- Trace 분석으로 실패 원인 좁힘
- Helicone·Langfuse·Braintrust 가 표준 도구

가장 기억할 한 줄: **"에이전트 평가는 최종 정확도만이 아니라 효율·안전·견고성까지 다차원으로 측정해야 한다."**

다음 권: [Volume 77 — Computer Use·Browser Agent](./volume_77_computer_use.md)

---

## 자가점검 키워드

`정확도/효율/안전/견고성`, `GAIA`, `SWE-Bench`, `Trace 분석`, `Helicone`, `Langfuse`

## 자가점검 질문

1. 에이전트 평가의 4 차원을 적으십시오.
2. 표준 벤치마크 4 가지를 적으십시오.
3. 자기 도메인 평가 셋의 양식을 적으십시오.
4. Trace 분석으로 좁힐 수 있는 실패 패턴 3 가지를 나열하십시오.

## 다음 권

[Volume 77 — Computer Use·Browser Agent](./volume_77_computer_use.md)
