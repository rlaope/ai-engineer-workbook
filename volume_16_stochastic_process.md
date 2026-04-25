# Volume 26 — 확률 과정 — Markov·SDE

> 이 권이 끝나면 *디퓨전 모델은 확률 과정이다* 라는 한 문장의 의미를 손에 잡히게 이해하게 됩니다.

## 목적

디퓨전 모델·강화학습·MCMC·Bayesian Optimization 은 모두 확률 과정의 언어 위에 서 있습니다. Markov Chain 의 *과거를 잊는* 성질, Brownian Motion 의 *연속 시간 노이즈*, SDE 의 *결정론적 + 확률적* 결합은 모두 디퓨전의 핵심 직관입니다. 이 권은 디퓨전 권(46) 보다 한 층 깊이의 수학적 기반을 만듭니다.

## 선수 지식

- Volume 11, 55 완료
- 외부 지식: 시간이 흐르는 시스템의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Markov Property 의 정의를 한 문장으로 적을 수 있습니다.
2. 정상 분포(Stationary Distribution) 의 의미를 그릴 수 있습니다.
3. Brownian Motion 의 4가지 핵심 성질을 나열할 수 있습니다.
4. SDE 와 ODE 의 차이를 설명할 수 있습니다.
5. DDPM 의 정·역방향 과정이 SDE 임을 보일 수 있습니다.

## 챕터 목차

1. **확률 과정의 정의**
2. **Markov Chain — 이산 시간**
3. **정상 분포와 수렴**
4. **MCMC 의 원리**
5. **Brownian Motion — 연속 시간 노이즈**
6. **SDE — Stochastic Differential Equation**
7. **Itô / Stratonovich 적분의 직관**
8. **디퓨전 모델과 SDE 의 연결**

## 자가점검 키워드

`Markov`, `정상 분포`, `MCMC`, `Brownian`, `SDE`, `Itô`, `Score`, `Reverse SDE`

## 다음 권

[Volume 27 — 특성 공학](./volume_27_feature_engineering.md)
