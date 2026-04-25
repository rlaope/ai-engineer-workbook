# Volume 83 — Image Diffusion

> 이 권이 끝나면 Stable Diffusion·SDXL·FLUX 같은 모델이 내부에서 무엇을 하는지 한 그림으로 그릴 수 있게 됩니다.

## 목적

이미지 디퓨전은 GAN·VAE 의 한계를 넘어 현재 이미지·비디오·오디오 생성의 사실상 표준이 되었습니다. *노이즈를 점진적으로 제거하면서 데이터를 만들어 가는* 단순한 발상이 어떻게 SOTA 가 되었는가를 이해하면, 이 분야의 새 모델이 등장할 때마다 *어떤 부품이 새롭게 들어왔는가* 만 빠르게 추적할 수 있게 됩니다. 이 권은 디퓨전 모델 전체 계보를 정리합니다.

## 선수 지식

- Volume 10, 7, 8, 32, 45 완료
- 외부 지식: 노이즈와 신호의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. DDPM 의 정·역방향 과정을 그림으로 설명할 수 있습니다.
2. DDIM 이 *동일한 학습 모델로 더 적은 스텝* 추론을 가능하게 한 발상을 알 수 있습니다.
3. Latent Diffusion 이 *왜 폭발적으로 채택되었는지* 비용 관점에서 설명할 수 있습니다.
4. Rectified Flow 의 직관을 *직선화된 경로*로 설명할 수 있습니다.
5. CFG(Classifier-Free Guidance) 가 추론 시 무엇을 하는지 설명할 수 있습니다.

## 챕터 목차

1. **생성 모델 계보의 큰 그림** — GAN·VAE·Diffusion·Flow
2. **DDPM** — 노이즈 추가와 제거
3. **DDIM** — 결정론적 가속
4. **Score-Based Models** — SDE 관점
5. **Latent Diffusion (Stable Diffusion)** — VAE + 디퓨전
6. **Conditioning** — Text·Image·Pose 컨디셔닝의 일반 형태
7. **Classifier-Free Guidance (CFG)**
8. **Rectified Flow** — FLUX·SD3 의 기반
9. **디스틸레이션** — 4 step·1 step 모델의 비밀
10. **DiT·MM-DiT** — 디퓨전과 트랜스포머의 결합

## 자가점검 키워드

`DDPM`, `DDIM`, `Latent Diffusion`, `Conditioning`, `CFG`, `Rectified Flow`, `Distillation`, `DiT/MM-DiT`

## 다음 권

[Volume 47 — GPU 아키텍처와 CUDA](./volume_84_gpu_cuda.md)
