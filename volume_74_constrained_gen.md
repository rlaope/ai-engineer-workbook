# Volume 74 — Constrained Generation

> 이 권이 끝나면 LLM 의 출력을 *항상 유효한 JSON*·*특정 정규식*·*특정 함수 시그니처* 로 강제하는 방법을 코드로 구현할 수 있게 됩니다.

## 목적

LLM 출력을 서비스가 신뢰하려면 *형식이 항상 유효*해야 합니다. 자유 텍스트로 *JSON 형식으로 답해줘* 라고 부탁하는 것은 깨지기 쉬우며, 진정한 신뢰성은 *디코딩 단계에서 형식을 강제*할 때 나옵니다. JSON Schema·정규식·CFG 강제 디코딩, Outlines·Guidance·OpenAI Structured Outputs 같은 도구가 이 영역의 표준입니다.

## 선수 지식

- Volume 39, 41, 72 완료
- 외부 지식: JSON Schema, 정규식

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. *프롬프트 강제* 와 *디코딩 강제* 의 신뢰성 차이를 설명할 수 있습니다.
2. JSON Schema 기반 출력의 동작 원리를 알 수 있습니다.
3. 토큰 단위 마스킹의 알고리즘을 이해합니다.
4. CFG(Context-Free Grammar) 기반 강제의 사용 시점을 알 수 있습니다.
5. Function Calling 의 내부가 Constrained Generation 임을 설명할 수 있습니다.

## 챕터 목차

1. **자유 생성의 신뢰성 한계**
2. **JSON Mode·Structured Outputs** — OpenAI/Anthropic 표준
3. **토큰 마스킹의 원리**
4. **정규식 강제 디코딩**
5. **CFG 기반 강제** — Lark·EBNF
6. **Outlines / Guidance / LMQL**
7. **Function Calling 의 내부**
8. **함정** — Tokenization 경계·낮은 확률 구간

## 자가점검 키워드

`JSON Mode`, `Structured Outputs`, `토큰 마스킹`, `정규식 디코딩`, `CFG`, `Outlines`, `Guidance`, `Function Calling`

## 다음 권

[Volume 75 — Agent Memory 시스템](./volume_75_agent_memory.md)
