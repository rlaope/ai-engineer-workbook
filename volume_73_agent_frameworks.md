# Volume 73 — 에이전트 프레임워크

> 이 권이 끝나면 새 에이전트 프로젝트를 시작할 때 *프레임워크 없이 짤지·LangGraph 로 짤지·MCP 위에 올릴지* 를 30 초 안에 결정할 수 있게 됩니다.

## 목적

에이전트 프레임워크는 *반복 코드를 줄이고 패턴을 표준화* 합니다. LangChain·LlamaIndex·LangGraph·MCP·Claude Agent SDK 같은 도구의 강약을 비교합니다.

## 선수 지식

- Volume 72 완료

## 학습 결과

1. 주요 프레임워크 5 가지의 차별점을 알 수 있습니다.
2. 프레임워크가 *해결하는 문제와 안 하는 문제* 를 구분합니다.
3. *프레임워크 없이 직접 짜기* vs *프레임워크 사용* 의 트레이드오프를 안다.
4. MCP 의 *도구 표준화* 발상을 이해합니다.

---

## 1. LangChain

가장 인기 있는 LLM 애플리케이션 프레임워크. *체인·도구·메모리·에이전트* 를 통합.

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(name="Search", func=search_fn, description="Web search"),
    Tool(name="Calc", func=calc_fn, description="Math calculation"),
]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What is 23 * 17?")
```

장점: *모든 것* 이 들어 있음. 큰 커뮤니티.
단점: *추상화가 두꺼워* 디버깅 어려움. *과도한 추상화* 비판.

---

## 2. LlamaIndex

RAG 에 특화. *문서 인덱싱·검색·답변* 의 패턴이 잘 정리.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
print(query_engine.query("What does the document say about ...?"))
```

장점: RAG 가 매우 단순. 단점: 일반 에이전트는 LangChain 이 더 풍부.

---

## 3. LangGraph

LangChain 팀의 *상태 기반 에이전트 프레임워크*. *복잡한 워크플로 (분기·반복·다중 에이전트)* 를 그래프로 표현.

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
builder.add_node("planner", planner_fn)
builder.add_node("executor", executor_fn)
builder.add_edge("planner", "executor")
builder.add_conditional_edges("executor", should_continue)
graph = builder.compile()
```

장점: *복잡한 흐름이 명시적*. 디버깅 쉬움.

---

## 4. MCP (Model Context Protocol)

Anthropic 의 *도구 표준 프로토콜* (2024). *LLM 과 도구 사이의 표준 인터페이스* 정의.

장점: *MCP 서버 한 번 만들면* 모든 MCP 호환 클라이언트 (Claude Desktop, Claude Agent SDK 등) 에서 사용 가능.

```json
// MCP 서버 manifest 예
{
  "name": "weather-server",
  "tools": [{
    "name": "get_weather",
    "description": "Get weather",
    "inputSchema": {...}
  }]
}
```

빠르게 산업 표준이 되어 가는 중.

---

## 5. Claude Agent SDK

Anthropic 의 *Claude 특화 에이전트 SDK*. MCP 와 통합. Claude Code 가 이 SDK 위에 만들어짐.

```python
from claude_agent_sdk import Agent

agent = Agent(model="claude-sonnet-4")
response = agent.run("Help me debug this code")
```

---

## 6. 비교 표

```
+-------------+-------------------+------------------+
| 프레임워크  | 강점              | 적용 시점         |
+-------------+-------------------+------------------+
| 직접 작성    | 100% 통제         | 작은 시스템·학습  |
| LangChain   | 풍부한 컴포넌트     | 빠른 프로토타입   |
| LlamaIndex  | RAG 특화           | RAG 중심 시스템   |
| LangGraph   | 복잡한 워크플로     | 멀티 에이전트     |
| MCP         | 도구 표준화         | 도구 재사용       |
| Claude Agent SDK | Claude 통합   | Claude 위주 환경  |
+-------------+-------------------+------------------+
```

---

## 7. 권장 흐름

1. *프로토타입* — 50 줄 직접 코드 또는 LangChain
2. *RAG 중심* — LlamaIndex
3. *복잡한 워크플로* — LangGraph
4. *Claude 환경* — Claude Agent SDK + MCP
5. *프로덕션* — 점진적으로 직접 코드로 마이그레이션 (제어권 회복)

---

## 권 정리

- LangChain = 가장 풍부, 추상화 두꺼움
- LlamaIndex = RAG 특화
- LangGraph = 복잡한 워크플로
- MCP = 도구 표준 프로토콜
- Claude Agent SDK = Claude 통합
- 프로토타입은 프레임워크, 프로덕션은 점진 직접화

가장 기억할 한 줄: **"프레임워크는 프로토타입을 빠르게 만들지만, 프로덕션은 결국 자기 코드로 통제권을 가져오는 경향이 있다."**

다음 권: [Volume 74 — 멀티에이전트와 워크플로](./volume_74_multi_agent.md)

---

## 자가점검 키워드

`LangChain`, `LlamaIndex`, `LangGraph`, `MCP`, `Claude Agent SDK`

## 자가점검 질문

1. 5 가지 프레임워크의 강약을 표로 정리하십시오.
2. 직접 작성 vs 프레임워크 사용의 트레이드오프를 적으십시오.
3. MCP 가 *도구 표준 프로토콜* 인 이유를 설명하십시오.

## 다음 권

[Volume 74 — 멀티에이전트와 워크플로](./volume_74_multi_agent.md)
