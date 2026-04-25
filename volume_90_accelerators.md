# Volume 96 — AI 가속기 비교 — GPU·TPU·Trainium·Cerebras·Groq

> 이 권이 끝나면 NVIDIA 외의 가속기 옵션이 어떤 워크로드에서 의미가 있는지 답할 수 있게 됩니다.

## 목적

NVIDIA GPU 가 사실상 표준이지만, TPU·Trainium·Inferentia·Cerebras·Groq·Sambanova 같은 대안 가속기들은 학습/추론의 특정 영역에서 우위를 가집니다. 산업 현장의 의사결정은 *어떤 워크로드에 어떤 가속기가 가성비 최고인가* 의 비교에 기반합니다. 이 권은 주요 가속기의 아키텍처적 차이와 실제 사용 시점을 정리합니다.

## 선수 지식

- Volume 84, 48 완료
- 외부 지식: 컴퓨팅 비용 모델

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. NVIDIA GPU 의 세대 차이(Ampere·Hopper·Blackwell) 를 알 수 있습니다.
2. Google TPU v4·v5 의 시스톨릭 어레이 설계와 적합 워크로드를 알 수 있습니다.
3. AWS Trainium·Inferentia 의 위치를 비용 관점에서 설명할 수 있습니다.
4. Cerebras·Groq·Sambanova 같은 비주류 가속기의 *틈새 강점* 을 알 수 있습니다.
5. 워크로드 → 가속기 의사결정 표를 그릴 수 있습니다.

## 챕터 목차

1. **NVIDIA — A100·H100·H200·B200**
2. **AMD MI300X**
3. **Google TPU v4·v5e·v5p**
4. **AWS Trainium·Inferentia**
5. **Cerebras Wafer-Scale Engine**
6. **Groq LPU** — 결정론적 추론 속도
7. **Sambanova·Tenstorrent**
8. **온디바이스 가속기** — Apple Neural Engine·Qualcomm
9. **워크로드 → 가속기 의사결정 표**

## 자가점검 키워드

`NVIDIA Hopper/Blackwell`, `MI300X`, `TPU v5`, `Trainium`, `Cerebras`, `Groq LPU`, `Apple ANE`, `의사결정 표`

## 다음 권

[Volume 61 — 임베딩 모델 직접 학습](./volume_61_embedding_training.md)
