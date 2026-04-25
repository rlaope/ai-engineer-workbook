# Volume 65 — 활성화 함수 깊이

> 이 권이 끝나면 ReLU·GELU·SiLU·Swish 중 *왜 현대 트랜스포머는 GELU/SiLU 를 쓰는가* 에 답할 수 있게 됩니다.

## 목적

활성화 함수 선택은 모델 성능과 학습 안정성에 영향을 주며, 시대마다 표준이 바뀌었습니다. 시그모이드 → ReLU → GELU/SiLU/SwiGLU 의 흐름에는 각각 *해결한 문제와 새로 만든 문제*가 있습니다. 이 권은 활성화 함수의 형태·미분·계산 비용·학습 안정성을 비교하고, 현대 LLM 이 SwiGLU 를 채택한 이유를 다집니다.

## 선수 지식

- Volume 30, 19 완료
- 외부 지식: 함수 그래프

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 시그모이드/tanh 가 *왜 깊은 망에서 사라졌는가* 를 설명할 수 있습니다.
2. ReLU 의 죽은 뉴런 문제와 그 변형(LeakyReLU·PReLU·ELU) 의 동기를 알 수 있습니다.
3. GELU·SiLU·Swish·Mish 의 형태와 미분을 그릴 수 있습니다.
4. SwiGLU 가 트랜스포머 FFN 에서 채택된 이유를 설명할 수 있습니다.
5. 활성화 함수 선택을 *경험적 비교 실험*으로 검증할 수 있습니다.

## 챕터 목차

1. **활성화 함수의 역할 복습**
2. **시그모이드와 tanh 의 한계**
3. **ReLU 와 그 변형들** — Leaky·Parametric·ELU·SELU
4. **GELU** — 가우시안 오차 선형
5. **Swish·SiLU·Mish**
6. **GLU·SwiGLU** — 게이트가 있는 활성화
7. **활성화 함수와 정규화의 상호작용**
8. **PyTorch 로 비교 실험**

## 자가점검 키워드

`시그모이드`, `ReLU`, `Leaky/PReLU`, `GELU`, `SiLU/Swish`, `Mish`, `GLU/SwiGLU`, `죽은 뉴런`

## 다음 권

[Volume 61 — 학습률 스케줄 깊이](./volume_40_lr_schedules.md)
