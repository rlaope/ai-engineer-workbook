# Volume 48 — 프롬프트와 In-Context Learning

> 이 권이 끝나면 *프롬프트는 새로운 프로그래밍 언어*라는 명제의 의미를 정확히 이해하게 됩니다.

## 목적

LLM 의 동작은 프롬프트에 의해 크게 달라집니다. 프롬프트 엔지니어링은 *마법의 주문*이 아니라 *모델의 컨텍스트 윈도우에 어떤 정보를 어떤 순서로 배치해 어떤 패턴을 유도할 것인가* 의 공학입니다. Few-shot·Chain-of-Thought·Tree-of-Thought·Self-Consistency 같은 기법은 모두 *컨텍스트 조작*의 변형입니다. 이 권은 그 사고 도구상자를 정리합니다.

## 선수 지식

- Volume 63, 38 완료
- 외부 지식: API 호출 경험

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Zero-shot·Few-shot·In-Context Learning 의 차이를 설명할 수 있습니다.
2. Chain-of-Thought 가 추론 성능을 올리는 메커니즘을 그릴 수 있습니다.
3. Self-Consistency·Tree-of-Thought·ReAct 의 사고 패턴을 구분할 수 있습니다.
4. System/User/Assistant 메시지 구조의 의미를 설명할 수 있습니다.
5. 생성 파라미터(temperature·top-k·top-p) 가 출력에 미치는 영향을 알 수 있습니다.

## 챕터 목차

1. **프롬프트의 구조** — System·User·Assistant
2. **Zero-shot vs Few-shot vs Many-shot**
3. **In-Context Learning** — 학습 없이 학습하는 것처럼
4. **Chain-of-Thought (CoT)**
5. **Self-Consistency** — 다수결 추론
6. **Tree-of-Thought** — 분기 탐색
7. **ReAct** — 추론과 행동의 인터리브
8. **생성 파라미터** — temperature·top-k·top-p·repetition penalty
9. **프롬프트 평가와 자동 최적화**

## 자가점검 키워드

`Zero/Few-shot`, `ICL`, `CoT`, `Self-Consistency`, `ToT`, `ReAct`, `temperature`, `top-p`

## 다음 권

[Volume 66 — RAG — 검색 증강 생성](./volume_66_rag.md)
