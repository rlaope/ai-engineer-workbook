# Volume 77 — Computer Use·Browser Agent

> 이 권이 끝나면 *모델이 화면을 보고 마우스·키보드를 조작하는* 새로운 패러다임의 가능성과 한계를 모두 이해하게 됩니다.

## 목적

Claude Computer Use·OpenAI Operator·browser-use 같은 도구는 LLM 에 *스크린샷을 보여 주고 액션을 선택하게 함*으로써 *어떤 소프트웨어든* 에이전트가 사용할 수 있게 만듭니다. API 가 없는 서비스도 자동화할 수 있는 잠재력이 있지만, 안전·속도·신뢰성의 거대한 도전을 동반합니다. 이 권은 이 영역의 현재와 함정을 정리합니다.

## 선수 지식

- Volume 41, 65, 76 완료
- 외부 지식: 브라우저 자동화(Selenium·Playwright) 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Computer Use 패턴의 입출력(스크린샷·액션) 을 설명할 수 있습니다.
2. Browser Agent 가 일반 Computer Use 보다 *제약된 환경*인 이유를 알 수 있습니다.
3. 액션 공간(클릭·타이핑·스크롤·키보드 단축키) 의 표준 형태를 알 수 있습니다.
4. 안전 정책(승인 필수 액션·격리된 환경) 을 설계할 수 있습니다.
5. 현재 한계(속도·정확도·민감 작업) 를 인식합니다.

## 챕터 목차

1. **Computer Use 패러다임의 정의**
2. **입력 표현** — 스크린샷·DOM·접근성 트리
3. **액션 공간 설계**
4. **Claude Computer Use 의 동작**
5. **OpenAI Operator·browser-use 비교**
6. **격리·승인·롤백 — 안전 정책**
7. **속도·정확도·신뢰성의 현재**
8. **합법성·약관 위배의 함정**

## 자가점검 키워드

`Computer Use`, `Browser Agent`, `Operator`, `browser-use`, `액션 공간`, `격리`, `승인 필수`, `약관 위배`

## 다음 권

[Volume 78 — 생성 제어 기법](./volume_78_generation_control.md)
