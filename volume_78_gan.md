# Volume 81 — GAN 과 적대적 학습

> 이 권이 끝나면 *경찰과 위조범* 비유로 GAN 의 학습 동학을 한 문단으로 설명할 수 있게 됩니다.

## 목적

GAN 은 디퓨전 이전 시대 이미지 생성의 표준이었으며, 지금도 *고해상도·실시간*이 필요한 영역에서 살아 있습니다. 더 중요한 것은 GAN 이 *적대적 학습*이라는 사고법을 가르쳐 준다는 점입니다. 이 사고법은 Self-Supervised Learning·Robust Training·평가 모델 학습 등에 그대로 활용됩니다.

## 선수 지식

- Volume 36, 26 완료
- 외부 지식: 영합 게임의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Generator·Discriminator 의 역할을 설명할 수 있습니다.
2. GAN 손실 함수가 *영합 게임의 미니맥스*임을 보일 수 있습니다.
3. 모드 붕괴·학습 불안정의 원인을 알 수 있습니다.
4. DCGAN·StyleGAN·CycleGAN 의 차별점을 알 수 있습니다.
5. GAN 과 디퓨전의 트레이드오프를 비교할 수 있습니다.

## 챕터 목차

1. **GAN 의 직관** — 위조범과 경찰
2. **수학적 정의** — 미니맥스 게임
3. **학습의 어려움** — 모드 붕괴·진동
4. **DCGAN** — 합성곱 기반의 첫 안정화
5. **WGAN** — Wasserstein 거리로 안정화
6. **StyleGAN** — 스타일 분리와 고해상도
7. **CycleGAN / Pix2Pix** — 이미지-이미지 변환
8. **GAN vs Diffusion** — 적용 영역과 비용

## 자가점검 키워드

`Generator`, `Discriminator`, `미니맥스`, `모드 붕괴`, `DCGAN`, `WGAN`, `StyleGAN`, `CycleGAN`

## 다음 권

[Volume 79 — VAE 와 잠재 변수 모델](./volume_79_vae.md)
