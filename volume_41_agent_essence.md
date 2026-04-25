# Volume 41 — 에이전트의 본질

> 이 권이 끝나면 *AI 에이전트는 결국 LLM + 도구 + 루프*라는 한 줄 요약에 동의하게 됩니다.

## 목적

AI 에이전트는 *LLM 이 도구를 호출하고, 그 결과를 다시 입력으로 받아 다음 행동을 결정하는 루프*입니다. 이 한 가지 기본 패턴 위에 ReAct·Tool Use·Function Calling·Plan-Execute 같은 변형이 얹힙니다. 이 권은 에이전트 설계의 가장 본질적인 원리만을 다지고, 다음 권에서 프레임워크를 다룹니다.

## 선수 지식

- Volume 39, 40 완료
- 외부 지식: 함수 호출·콜백의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Tool Use 와 Function Calling 의 관계를 설명할 수 있습니다.
2. ReAct 패턴의 *Thought·Action·Observation* 루프를 그릴 수 있습니다.
3. JSON 스키마 기반 도구 정의의 표준 형태를 알 수 있습니다.
4. 에이전트 종료 조건(스톱 토큰·최대 스텝·목표 달성) 의 설계를 다룰 수 있습니다.
5. 에이전트 디버깅 시 *어디를 봐야 하는지* 알 수 있습니다.

## 챕터 목차

1. **에이전트의 한 줄 정의** — LLM + 도구 + 루프
2. **Function Calling 의 표준 형태** — JSON 스키마
3. **Tool Use vs Function Calling** — 미묘한 차이
4. **ReAct — Reasoning + Acting**
5. **Plan-then-Execute** — 분리된 두 단계
6. **메모리 — 단기·장기·에피소드**
7. **종료 조건과 안전 장치**
8. **에이전트 디버깅** — 추론·행동·관측의 추적

## 자가점검 키워드

`Function Calling`, `Tool Use`, `ReAct`, `Plan-Execute`, `JSON Schema`, `메모리`, `종료 조건`, `Trace`

## 다음 권

[Volume 42 — 에이전트 프레임워크](./volume_42_agent_frameworks.md)
