# Volume 77 — Computer Use·Browser Agent

> 이 권이 끝나면 *모델이 화면을 보고 마우스·키보드를 조작하는* 새로운 패러다임의 가능성과 한계를 모두 이해하게 됩니다.

## 목적

Claude Computer Use·OpenAI Operator·browser-use 같은 도구는 LLM 에 *스크린샷을 보여 주고 액션을 선택하게 함* 으로써 *어떤 소프트웨어든* 에이전트가 사용할 수 있게 만듭니다. API 가 없는 서비스도 자동화할 수 있는 잠재력이 있지만, 안전·속도·신뢰성의 거대한 도전을 동반합니다.

## 선수 지식

- Volume 48, 72, 76 완료

## 학습 결과

1. Computer Use 패턴의 입출력 (스크린샷·액션) 을 설명할 수 있습니다.
2. Browser Agent 가 일반 Computer Use 보다 *제약된 환경* 인 이유를 알 수 있습니다.
3. 액션 공간 (클릭·타이핑·스크롤·키보드 단축키) 의 표준 형태를 알 수 있습니다.
4. 안전 정책 (승인 필수 액션·격리된 환경) 을 설계할 수 있습니다.

---

## 1. Computer Use 패러다임

### 1.1 입출력

```
입력: [현재 스크린샷]
출력: [다음 액션] (클릭, 타이핑, 스크롤, 키 입력 등)
```

LLM 이 *컴퓨터를 사용자처럼 사용*. API 가 없는 *어떤 GUI 소프트웨어든* 자동화 가능.

### 1.2 등장 배경

대부분의 소프트웨어는 *공식 API 가 없거나 제한적*. 사람이 GUI 로 하는 일을 자동화하려면 *Selenium·UI Automation* 같은 깨지기 쉬운 도구 사용. Computer Use 는 *시각 + 의도 이해* 로 더 일반화 가능한 자동화를 약속.

---

## 2. 액션 공간

### 2.1 기본 액션

```python
actions = {
    "click": {"x": int, "y": int},
    "type": {"text": str},
    "scroll": {"direction": "up|down", "amount": int},
    "key": {"keys": "ctrl+s"},
    "wait": {"seconds": float},
    "screenshot": {},
}
```

### 2.2 좌표의 어려움

LLM 이 *정확한 픽셀 좌표* 를 출력해야 함. 현재 모델은 *작은 UI 요소* 에 부정확한 경향.

---

## 3. Claude Computer Use

Anthropic 의 첫 공개 (2024).

```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[{"type": "computer_20241022", "name": "computer", ...}],
    messages=[{"role": "user", "content": "Open Chrome and search for AI news"}],
)
```

`[VERIFY: API 호출 정확한 형식]`

---

## 4. Browser Agent

### 4.1 일반 Computer Use 와의 차이

브라우저는 *DOM 접근* 가능 → 더 구조화된 환경. 픽셀 좌표 대신 *CSS 셀렉터·XPath* 사용 가능.

### 4.2 도구

- **Playwright + LLM** — 직접 통합
- **browser-use** — 오픈소스
- **OpenAI Operator** — Operator API

```python
from browser_use import Agent
agent = Agent(task="Find the cheapest flight to Tokyo")
result = agent.run()
```

---

## 5. 안전 정책

### 5.1 격리

Computer Use 는 *호스트 시스템에 직접 접근 위험*. 표준 권장:

- *VM 또는 컨테이너* 에서 실행
- *읽기 전용 파일 시스템* (필요 시)
- *네트워크 격리* (특정 도메인만 허용)

### 5.2 승인 필수 액션

다음은 *항상 사람 승인* 후 실행:

- 결제·금융 거래
- 파일 삭제
- 이메일·메시지 전송
- 시스템 설정 변경
- 외부 API 키 사용

### 5.3 롤백 가능성

가능하면 *되돌릴 수 있는 액션만*. 영구적 변경은 추가 확인.

---

## 6. 현재 한계

- *속도* — 한 액션당 *수 초* (스크린샷 → LLM → 액션). 수십 단계 작업에 분 단위 소요
- *정확도* — 작은 UI 요소·복잡한 레이아웃에서 실패
- *신뢰성* — 같은 작업도 시행마다 다른 결과
- *비용* — 매 단계 LLM 호출 + 비전 토큰
- *민감 작업* — 실수의 영향이 큼

---

## 7. 합법성·약관

웹사이트의 *robots.txt·약관* 을 위반할 수 있음. 자동화는 *허용된 범위에서만* 사용해야.

기업용 자동화는 *내부 시스템* 또는 *명시적 허락이 있는 외부 시스템* 에 한정 권장.

---

## 권 정리

- Computer Use = 스크린샷 → 액션 (마우스·키보드)
- Browser Agent = DOM 접근 가능, 더 구조화
- 액션 공간 = 클릭·타이핑·스크롤·키
- 안전 = 격리·승인·롤백
- 한계 = 속도·정확도·신뢰성·비용
- 합법성·약관 준수 필수

가장 기억할 한 줄: **"Computer Use 는 모든 소프트웨어를 자동화할 수 있는 잠재력을 가지지만, 속도·정확도·안전의 거대한 도전을 동반한다."**

다음 권: [Volume 78 — GAN과 적대적 학습](./volume_78_gan.md)

---

## 자가점검 키워드

`Computer Use`, `Browser Agent`, `Operator`, `browser-use`, `액션 공간`, `격리`, `승인 필수`, `약관`

## 자가점검 질문

1. Computer Use 의 입출력을 적으십시오.
2. Browser Agent 가 일반 Computer Use 보다 유리한 점을 설명하십시오.
3. *항상 사람 승인* 이 필요한 액션 종류 5 가지를 나열하십시오.
4. Computer Use 의 4 가지 현재 한계를 적으십시오.

## 다음 권

[Volume 78 — GAN과 적대적 학습](./volume_78_gan.md)
