# Volume 74 — 멀티에이전트와 워크플로

> 이 권이 끝나면 *언제 멀티에이전트가 단일 에이전트보다 나은가* 라는 까다로운 질문에 답할 수 있게 됩니다.

## 목적

멀티에이전트는 *분업과 검토* 가 모델 한 대보다 결과가 좋을 때만 가치가 있습니다. 잘못 설계하면 토큰 비용만 늘고 일관성은 떨어집니다. Planner-Executor·Critic·Self-Reflection·Swarm 같은 패턴을 다집니다.

## 선수 지식

- Volume 72, 73 완료

## 학습 결과

1. Planner-Executor 패턴의 의도를 설명할 수 있습니다.
2. Critic / Self-Reflection 의 효과와 비용을 알 수 있습니다.
3. 멀티에이전트가 *오버헤드만 늘리는* 안티 패턴을 식별합니다.
4. 메시지 큐·이벤트 기반 워크플로를 설계할 수 있습니다.

---

## 1. 단일 vs 멀티 트레이드오프

단일 에이전트가 충분한데 멀티로 가면 *비용만 N 배·일관성 저하*. 멀티가 정당한 경우는:

- *역할 전문화* — 각 에이전트가 다른 도메인 전문성
- *검토 단계 분리* — 작성과 검토를 다른 에이전트가
- *병렬 가능* — 독립적 하위 작업
- *복잡한 분기* — 단일 에이전트가 결정하기에 너무 복잡

---

## 2. Planner-Executor 패턴

### 2.1 구조

```
사용자 요청 → [Planner] → 계획 (단계 리스트) → [Executor] → 각 단계 실행
```

Planner 는 *큰 그림을 짠다*, Executor 는 *각 단계를 충실히 실행*.

### 2.2 장점

- 큰 작업 분해
- 디버깅이 쉬움 (계획 단계가 명시적)

### 2.3 사용 예

```python
# Planner
plan = llm.complete(f"""Break this task into steps:
{user_query}

Output JSON: [{{step: 1, description: "..."}}, ...]""")

# Executor
for step in plan:
    result = execute_step(step)
```

---

## 3. Critic / Self-Reflection

### 3.1 발상

작성 후 *다른 에이전트 (또는 같은 에이전트) 가 검토*. 문제점 발견 시 수정.

```
[Writer] → 초안
   ↓
[Critic] → 피드백
   ↓
[Writer] → 수정안
   ↓ (수렴할 때까지 반복)
```

### 3.2 효과

- 정확도 *수 % 향상*
- 비용 *2-N 배*

자기 검토가 *항상 도움이 되지는 않음*. 작은 작업에는 오버헤드.

---

## 4. 역할 분담 — Researcher·Coder·Reviewer

복잡한 시스템 개발:

```
[Researcher] — 기존 코드·문서 탐색
[Planner]    — 변경 계획
[Coder]      — 실제 코드 작성
[Reviewer]   — 코드 리뷰·테스트 작성
[Tester]     — 실행·검증
```

각 에이전트가 *다른 시스템 프롬프트·다른 도구* 사용.

---

## 5. 이벤트 기반 워크플로

큰 시스템은 *동기 호출* 대신 *메시지 큐* 로 비동기:

```
사용자 → 큐 → [Worker 1] → 큐 → [Worker 2] → ...
```

장점: *일부 실패가 전체 실패가 아님*, *수평 확장 가능*.

도구: Temporal, Airflow, Prefect, Celery.

---

## 6. 프레임워크

- **Swarm (OpenAI)** — 단순한 멀티 에이전트 핸드오프
- **CrewAI** — 역할 기반 멀티 에이전트
- **AutoGen (Microsoft)** — 대화 기반 멀티 에이전트
- **LangGraph** — 그래프 기반 (Vol 73)

---

## 7. 흔한 안티 패턴

- *과도한 분업* — 작은 작업도 5 명이 처리
- *명확하지 않은 역할* — 각 에이전트가 같은 것을 다시 함
- *공유 메모리 누수* — 한 에이전트의 컨텍스트가 다른 에이전트로 부적절하게 흐름
- *무한 합의 루프* — Critic 이 영원히 만족 안 함

---

## 권 정리

- 단일이 충분하면 단일 — 멀티는 정당화 필요
- Planner-Executor = 큰 작업 분해
- Critic/Self-Reflection = 검토 (오버헤드 비용)
- 역할 분담 = Researcher·Coder·Reviewer 등
- 이벤트 기반 = 큰 시스템에 필수
- 프레임워크: Swarm·CrewAI·AutoGen·LangGraph

가장 기억할 한 줄: **"멀티 에이전트는 분업·검토·병렬·복잡 분기가 정당화될 때만 가치가 있으며, 그 외에는 비용만 늘린다."**

다음 권: [Volume 75 — Agent Memory 시스템](./volume_75_agent_memory.md)

---

## 자가점검 키워드

`Planner-Executor`, `Critic`, `Self-Reflection`, `역할 분담`, `이벤트 기반`, `Swarm`, `CrewAI`

## 자가점검 질문

1. 멀티 에이전트가 정당화되는 4 가지 경우를 적으십시오.
2. Planner-Executor 패턴의 흐름을 그리십시오.
3. Critic 패턴의 효과와 비용을 설명하십시오.
4. 멀티 에이전트의 4 가지 안티 패턴을 적으십시오.

## 다음 권

[Volume 75 — Agent Memory 시스템](./volume_75_agent_memory.md)
