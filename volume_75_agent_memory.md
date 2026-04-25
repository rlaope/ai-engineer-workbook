# Volume 75 — Agent Memory 시스템

> 이 권이 끝나면 *에이전트가 어제 말한 것을 오늘 기억하게 만드는* 시스템을 직접 설계할 수 있게 됩니다.

## 목적

LLM 자체는 *상태가 없습니다*. 매 호출이 독립적이며, 컨텍스트 윈도우 안의 정보만 알 수 있습니다. *지속 메모리* 시스템이 에이전트에게 *지속성·맥락 유지·개인화* 를 부여합니다.

## 선수 지식

- Volume 56, 66, 72 완료

## 학습 결과

1. *Episodic vs Semantic vs Procedural* 메모리의 차이를 알 수 있습니다.
2. Mem0·Letta 같은 메모리 시스템의 발상을 이해합니다.
3. *컨텍스트 길이 vs 메모리 검색* 의 트레이드오프를 안다.
4. 메모리의 *추출·저장·검색·갱신* 4 단계를 설계할 수 있습니다.

---

## 1. 메모리의 종류

심리학에서 빌려온 분류:

- **Episodic** — *과거 사건* 기억. "어제 사용자가 블루베리 머핀 요리법을 물었다"
- **Semantic** — *일반 사실* 기억. "사용자는 비건이다"
- **Procedural** — *방법* 기억. "이 사용자는 짧은 답변을 선호한다"

각각 *다른 저장·검색* 방식이 적합.

---

## 2. 표준 4 단계

```
[추출] LLM 이 대화에서 *기억할 만한 정보* 추출
   ↓
[저장] 적절한 형식으로 메모리에 보존 (벡터 DB, KV 스토어 등)
   ↓
[검색] 새 대화에서 관련 메모리 가져오기
   ↓
[갱신] 새 정보가 옛 정보와 충돌하면 갱신·통합
```

이 4 단계가 *모든 메모리 시스템의 골격*.

---

## 3. 단순 구현 — 대화 요약

```python
# 매 N 턴마다 대화를 요약해 저장
def summarize_and_store(conversation):
    summary = llm.complete(f"Summarize the key facts from this conversation:\n{conversation}")
    vector_db.add(summary, embedding=embed(summary))

# 새 대화 시작 시 관련 요약 검색
def get_relevant_memory(query):
    return vector_db.search(embed(query), k=5)
```

---

## 4. Mem0

오픈소스 메모리 라이브러리. *자동 추출·저장·검색* 통합.

```python
from mem0 import Memory

m = Memory()
m.add("My name is Alice and I love Python", user_id="alice")
m.add("I prefer dark mode", user_id="alice")

memories = m.search("What does Alice like?", user_id="alice")
```

Vector DB + LLM 추출이 결합된 구조.

---

## 5. Letta (구 MemGPT)

*OS-like 메모리 관리*. *Working Memory* (작은 컨텍스트) + *Long-term Memory* (외부 저장) 의 두 계층.

LLM 이 직접 *메모리 도구를 호출* 해 저장·검색.

---

## 6. 컨텍스트 vs 메모리 트레이드오프

긴 컨텍스트 모델 (200K+) 이 등장하면서, *모든 것을 컨텍스트에* 넣는 옵션도 가능. 그러나:

- *비용*: 컨텍스트가 길수록 비쌈
- *지연*: TTFT 증가
- *Lost in the Middle*: 긴 컨텍스트의 중간 정보 활용 약함

→ *메모리 시스템 + 짧은 컨텍스트* 가 *비용·정확도* 양쪽에서 우수한 경우가 많음.

---

## 7. 메모리 갱신·삭제

새 정보가 옛 정보와 충돌:

```
옛 정보: "Alice 는 Python 을 좋아한다"
새 대화: "이제 Rust 로 옮겼어요"
→ 갱신: "Alice 는 Rust 를 사용한다 (이전: Python)"
```

LLM 이 *충돌 감지·통합 결정*. 이 단계가 *메모리 품질의 핵심*.

또한 *사용자가 명시적으로 삭제 요청* 가능해야 (GDPR 등 규제).

---

## 권 정리

- 메모리 종류: Episodic·Semantic·Procedural
- 4 단계: 추출·저장·검색·갱신
- Mem0 = 통합 라이브러리, Letta = OS 방식
- 컨텍스트 vs 메모리 트레이드오프
- 갱신·삭제가 메모리 품질의 핵심

가장 기억할 한 줄: **"에이전트 메모리는 추출·저장·검색·갱신의 4 단계 시스템이며, 긴 컨텍스트만으로는 비용·정확도 양쪽에서 부족하다."**

다음 권: [Volume 76 — Agent 평가와 벤치마크](./volume_76_agent_eval.md)

---

## 자가점검 키워드

`Episodic`, `Semantic`, `Procedural`, `Mem0`, `Letta`, `메모리 갱신`

## 자가점검 질문

1. 메모리 3 종류의 차이를 적으십시오.
2. 메모리 시스템의 4 단계를 적으십시오.
3. 컨텍스트 vs 메모리 트레이드오프를 설명하십시오.
4. 메모리 갱신 시 충돌 처리 사고를 적으십시오.

## 다음 권

[Volume 76 — Agent 평가와 벤치마크](./volume_76_agent_eval.md)
