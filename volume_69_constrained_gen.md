# Volume 69 — Constrained Generation

> 이 권이 끝나면 LLM 의 출력을 *항상 유효한 JSON*·*특정 정규식*·*특정 함수 시그니처* 로 강제하는 방법을 코드로 구현할 수 있게 됩니다.

## 목적

LLM 출력을 서비스가 신뢰하려면 *형식이 항상 유효* 해야 합니다. 자유 텍스트로 *JSON 형식으로 답해줘* 라고 부탁하는 것은 깨지기 쉬우며, 진정한 신뢰성은 *디코딩 단계에서 형식을 강제* 할 때 나옵니다.

## 선수 지식

- Volume 65, 67 완료

## 학습 결과

1. *프롬프트 강제* 와 *디코딩 강제* 의 신뢰성 차이를 설명할 수 있습니다.
2. JSON Schema 기반 출력의 동작 원리를 알 수 있습니다.
3. 토큰 단위 마스킹의 알고리즘을 이해합니다.
4. Function Calling 의 내부가 Constrained Generation 임을 설명할 수 있습니다.

---

## 1. 자유 생성의 한계

```python
prompt = "Return a JSON object: {name, age}"
response = llm(prompt)
# 가능한 응답:
# 1. {"name": "Alice", "age": 30}   ← OK
# 2. ```json\n{"name": "Alice", "age": 30}\n```   ← 코드 블록 포함
# 3. Sure! Here's the JSON: {"name": ...}    ← 추가 텍스트
# 4. {"name": "Alice", age: 30}    ← 따옴표 누락
```

자유 생성은 *깨지기 쉽습니다*. 100 번 호출하면 1-5 번은 형식이 깨짐.

---

## 2. JSON Mode / Structured Outputs

OpenAI·Anthropic 의 *공식 구조화 출력* 기능:

```python
from openai import OpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about Alice, 30."}],
    response_format=Person,
)
person = response.choices[0].message.parsed
print(person.name, person.age)
```

내부적으로 *디코딩 시 형식 강제* — 100% 유효한 JSON 보장.

---

## 3. 토큰 마스킹의 원리

LLM 의 다음 토큰 분포에서 *형식에 맞지 않는 토큰의 확률을 0 으로* 마스킹:

```python
def constrained_decode(model, prompt, schema):
    tokens = []
    while not done:
        logits = model(prompt + tokens)
        valid_tokens = schema.allowed_next(tokens)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[valid_tokens] = True
        logits[~mask] = -float('inf')
        next_token = sample(logits)
        tokens.append(next_token)
    return tokens
```

이 방식이 *형식이 항상 유효* 함을 보장.

---

## 4. CFG 기반 강제

JSON 외 *임의 문법* 도 강제 가능. EBNF 또는 PEG 같은 형식 문법으로:

```
expr   := number | "(" expr op expr ")"
op     := "+" | "-" | "*" | "/"
number := [0-9]+
```

라이브러리: **Outlines**, **Guidance**, **LMQL**.

```python
import outlines

model = outlines.models.transformers("meta-llama/Llama-3-8B-Instruct")
generator = outlines.generate.regex(model, r"\d{3}-\d{3}-\d{4}")
phone_number = generator("Generate a phone number: ")
```

---

## 5. Function Calling 의 내부

OpenAI·Anthropic 의 함수 호출도 *Constrained Generation*:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    tools=tools,
)
```

LLM 이 *함수 시그니처에 맞는 JSON* 을 반환할 것이 보장됩니다.

---

## 6. 함정

### 6.1 토큰 경계

JSON 의 `:` 가 *한 토큰* 또는 *두 토큰* 일 수 있음. 토크나이저 차이에 따라 강제가 어색해질 수 있음.

### 6.2 낮은 확률 구간

강제로 *낮은 확률 토큰* 만 선택 가능한 상황이 되면 *모델 품질이 떨어짐*. 모델이 자연스럽게 출력할 수 있는 형식을 사용.

### 6.3 컨텍스트 길이

복잡한 JSON Schema 는 *프롬프트 길이* 를 키워 비용 증가.

---

## 권 정리

- 자유 생성 = 1-5% 형식 깨짐
- JSON Mode = 100% 유효 보장
- 토큰 마스킹 = 디코딩 시 형식 강제
- Outlines/Guidance/LMQL = 임의 문법 강제
- Function Calling = 본질적으로 Constrained Generation

가장 기억할 한 줄: **"프로덕션에서 LLM 출력을 신뢰하려면 프롬프트 강제가 아닌 디코딩 강제 (JSON Mode/Function Calling) 를 써야 한다."**

다음 권: [Volume 70 — 방어적 프롬프트 엔지니어링](./volume_70_defensive_prompting.md)

---

## 자가점검 키워드

`JSON Mode`, `Structured Outputs`, `토큰 마스킹`, `정규식 디코딩`, `CFG`, `Outlines`, `Function Calling`

## 자가점검 질문

1. 자유 생성과 디코딩 강제의 신뢰성 차이를 설명하십시오.
2. 토큰 마스킹의 동작을 의사코드로 적으십시오.
3. Function Calling 의 내부가 Constrained Generation 임을 설명하십시오.
4. Constrained Generation 의 3 가지 함정을 적으십시오.

## 다음 권

[Volume 70 — 방어적 프롬프트 엔지니어링](./volume_70_defensive_prompting.md)
