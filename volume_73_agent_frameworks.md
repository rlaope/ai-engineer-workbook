# Volume 68 — 에이전트 프레임워크

> 이 권이 끝나면 새 에이전트 프로젝트를 시작할 때 *프레임워크 없이 짤지·LangGraph 로 짤지·MCP 위에 올릴지* 를 30 초 안에 결정할 수 있게 됩니다.

## 목적

에이전트 프레임워크는 *루프·도구 호출·메모리·관측성* 같은 공통 부품을 제공합니다. 그러나 추상화 비용도 함께 따라옵니다. LangChain·LlamaIndex·LangGraph·MCP·Claude Agent SDK 의 강점과 약점을 모두 알면, 프로젝트 규모와 통제 수준에 따라 적절한 도구를 선택할 수 있게 됩니다. 이 권은 그 의사결정 워크시트를 만듭니다.

## 선수 지식

- Volume 72 완료
- 외부 지식: 의존성 관리·SDK 사용 경험

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. LangChain·LlamaIndex·LangGraph 의 위상 차이를 설명할 수 있습니다.
2. MCP(Model Context Protocol) 의 등장 동기와 위치를 알 수 있습니다.
3. Claude Agent SDK 의 핵심 추상화를 이해합니다.
4. 직접 짠 미니 에이전트와 프레임워크 기반 에이전트의 트레이드오프를 비교할 수 있습니다.
5. 관측성 도구(LangSmith·Helicone) 의 사용 시점을 알 수 있습니다.

## 챕터 목차

1. **LangChain — 가장 넓은 생태계**
2. **LlamaIndex — 인덱스/검색 중심**
3. **LangGraph — 그래프 기반 워크플로**
4. **MCP — 클라이언트-서버 표준**
5. **Claude Agent SDK**
6. **OpenAI Assistants API / Responses API**
7. **프레임워크 없이 직접 짜기**
8. **관측성 도구** — LangSmith·Helicone·Langfuse

## 자가점검 키워드

`LangChain`, `LlamaIndex`, `LangGraph`, `MCP`, `Agent SDK`, `Assistants API`, `LangSmith`, `Tracing`

## 다음 권

[Volume 43 — 멀티에이전트와 워크플로](./volume_74_multi_agent.md)
