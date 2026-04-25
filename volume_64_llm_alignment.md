# Volume 47 — LLM 정렬

> 이 권이 끝나면 *왜 사전학습된 LLM 만으로는 챗봇을 만들 수 없는가* 를 설명할 수 있게 됩니다.

## 목적

다음 토큰 예측만 학습한 모델은 *유창하지만 도움이 되지 않거나, 위험한 응답*을 만들 수 있습니다. 정렬은 모델의 출력을 *인간의 의도와 가치*에 맞추는 일이며, SFT·RLHF·DPO·Constitutional AI 같은 기법으로 이루어집니다. 이 권은 정렬 파이프라인의 큰 그림을 잡고 각 단계의 동기와 한계를 정리합니다.

## 선수 지식

- Volume 63 완료
- 외부 지식: 강화학습의 직관(보상에 따라 행동 변화)

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. SFT(Supervised Fine-Tuning) 의 데이터·목표를 설명할 수 있습니다.
2. RLHF 의 3 단계 파이프라인을 그릴 수 있습니다.
3. DPO 가 RLHF 의 간소화된 대안인 이유를 설명할 수 있습니다.
4. Constitutional AI 의 자기 비판 루프를 이해합니다.
5. 정렬의 한계(Sycophancy·Reward Hacking·Jailbreak) 를 인식합니다.

## 챕터 목차

1. **사전학습 모델의 한계** — 유창성과 유용성의 차이
2. **SFT — 지도학습 미세조정**
3. **RLHF 1단계 — 선호 데이터 수집**
4. **RLHF 2단계 — 보상 모델 학습**
5. **RLHF 3단계 — PPO 로 정책 최적화**
6. **KL 페널티의 역할** — 사전학습 분포에서 멀어지지 않게
7. **DPO — 보상 모델 없는 직접 최적화**
8. **Constitutional AI / RLAIF**
9. **정렬의 한계와 함정**

## 자가점검 키워드

`SFT`, `RLHF`, `보상 모델`, `PPO`, `KL 페널티`, `DPO`, `Constitutional AI`, `Reward Hacking`

## 다음 권

[Volume 65 — 프롬프트와 In-Context Learning](./volume_65_prompting.md)
