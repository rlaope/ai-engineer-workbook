# Volume 72 — 에이전트의 본질

> 이 권이 끝나면 *AI 에이전트는 결국 LLM + 도구 + 루프* 라는 한 줄 요약에 동의하게 됩니다.

## 목적

AI 에이전트는 *마법 같은 자율 시스템* 처럼 들리지만, 본질은 매우 단순합니다. *LLM 이 어떤 도구를 쓸지 결정 → 도구 실행 → 결과를 LLM 에 다시 입력 → 반복*. 이 권은 그 단순한 핵심을 다집니다.

## 선수 지식

- Volume 65, 69 완료

## 학습 결과

1. *LLM + 도구 + 루프* 의 에이전트 구조를 그릴 수 있습니다.
2. Function Calling·Tool Use 의 표준 인터페이스를 이해합니다.
3. ReAct 패턴을 직접 구현할 수 있습니다.
4. 에이전트의 흔한 실패 (무한 루프·잘못된 도구·환각) 를 안다.

---

## 1. 에이전트의 한 줄 정의

```
while not done:
    next_action = llm.decide(context, tools)
    if next_action.is_final_answer:
        done = True
    else:
        result = execute_tool(next_action)
        context.append(result)
return context.final_answer
```

이 7 줄이 *모든 에이전트의 골격* 입니다.

---

## 2. Function Calling

### 2.1 도구 정의

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}]
```

### 2.2 LLM 호출

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

if response.choices[0].message.tool_calls:
    # 도구 실행
    for call in response.choices[0].message.tool_calls:
        result = execute(call.function.name, call.function.arguments)
        messages.append({"role": "tool", "content": result, "tool_call_id": call.id})
    # 다시 LLM 호출
    response = client.chat.completions.create(...)
```

이 패턴이 *모든 에이전트 시스템의 코어*.

---

## 3. ReAct 패턴

*Reasoning + Acting* 의 명시적 인터리빙:

```
Thought: 사용자가 서울 날씨를 묻고 있다. 검색이 필요하다.
Action: get_weather(city="Seoul")
Observation: {"temp": 22, "condition": "sunny"}
Thought: 정보를 받았다. 응답을 만든다.
Final: "오늘 서울은 맑고 22°C 입니다."
```

LLM 이 *내부 사고를 명시적으로 출력* 하면 디버깅·해석이 쉬워집니다.

---

## 4. 미니 에이전트 (50 줄)

```python
import json
from openai import OpenAI

client = OpenAI()

tools = [...]   # 위와 같은 도구 정의

def execute_tool(name, args):
    args = json.loads(args)
    if name == "search_web":
        return search_engine(args["query"])
    elif name == "get_weather":
        return weather_api(args["city"])

def run_agent(user_query, max_iter=10):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": user_query},
    ]
    
    for i in range(max_iter):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            return msg.content
        
        for call in msg.tool_calls:
            result = execute_tool(call.function.name, call.function.arguments)
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": call.id,
            })
    
    return "Max iterations reached."

print(run_agent("What's the weather in Seoul today?"))
```

이 50 줄이 *완전한 에이전트* 입니다.

---

## 5. 흔한 실패 패턴

### 5.1 무한 루프

LLM 이 *같은 도구를 같은 인자로 반복* 호출. 결과가 같아 진전 없음.

방어: *반복 감지* + *max_iterations 제한*.

### 5.2 잘못된 도구

LLM 이 *부적합한 도구* 선택. 예: 날씨 질문에 검색 도구.

방어: *도구 description 을 명확히* + *system prompt 에 가이드*.

### 5.3 환각된 도구 호출

LLM 이 *존재하지 않는 도구* 또는 *잘못된 인자* 로 호출.

방어: *Function Calling (Constrained Generation)* 으로 형식 강제.

### 5.4 도구 실행 실패 처리

도구가 오류를 반환하면 LLM 에게 *오류 정보* 주고 *다시 시도* 또는 *우회*.

```python
try:
    result = execute_tool(...)
except Exception as e:
    result = f"Error: {e}. Try a different approach."
```

---

## 권 정리

- 에이전트 = LLM + 도구 + 루프 (7 줄 골격)
- Function Calling = 도구 인터페이스 표준
- ReAct = 추론·행동 인터리브
- 미니 에이전트 = 50 줄
- 실패 패턴 = 무한 루프·잘못된 도구·환각·실행 실패

가장 기억할 한 줄: **"AI 에이전트는 LLM 이 도구를 선택하고 결과를 다시 보는 루프이며, 그 골격은 50 줄로 충분하다."**

다음 권: [Volume 73 — 에이전트 프레임워크](./volume_73_agent_frameworks.md)

---

## 자가점검 키워드

`Function Calling`, `Tool Use`, `ReAct`, `루프`, `Tool Description`, `무한 루프`

## 자가점검 질문

1. 에이전트의 7 줄 골격을 적으십시오.
2. Function Calling 의 표준 인터페이스를 적으십시오.
3. ReAct 패턴의 4 단계 (Thought/Action/Observation/Final) 를 자기 도메인 예시로 그리십시오.
4. 에이전트 실패 4 패턴과 방어책을 적으십시오.

## 다음 권

[Volume 73 — 에이전트 프레임워크](./volume_73_agent_frameworks.md)
