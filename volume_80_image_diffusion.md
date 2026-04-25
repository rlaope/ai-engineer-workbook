# Volume 80 — Image Diffusion

> 이 권이 끝나면 Stable Diffusion·SDXL·FLUX 같은 모델이 내부에서 무엇을 하는지 한 그림으로 그릴 수 있게 됩니다.

## 목적

Image Diffusion 은 2022 년 이후 *이미지 생성의 표준 패러다임* 입니다. *노이즈를 점진적으로 제거하며 이미지를 만드는* 단순한 발상이 GAN 을 넘어 SOTA 가 되었습니다.

## 선수 지식

- Volume 16, 79 완료

## 학습 결과

1. DDPM 의 *순방향·역방향* 과정을 그릴 수 있습니다.
2. DDIM 이 *DDPM 가속* 인 이유를 알 수 있습니다.
3. Latent Diffusion (Stable Diffusion) 의 효율성 비밀을 이해합니다.
4. Rectified Flow·Distillation 같은 가속 기법의 발상을 안다.
5. Classifier-Free Guidance (CFG) 의 역할을 설명합니다.

---

## 1. DDPM — 디퓨전의 기본

### 1.1 순방향 (Forward)

이미지에 *노이즈를 점진적으로 추가* — T 단계 (보통 1000) 후 *완전한 가우시안 노이즈*.

```
x_0 (이미지) → x_1 → x_2 → ... → x_T (가우시안 노이즈)
```

수식: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, $\epsilon \sim N(0, I)$.

### 1.2 역방향 (Reverse)

*노이즈에서 이미지로 점진적 복원*. 신경망 (U-Net) 이 *각 단계의 노이즈* 를 예측해 제거.

```
x_T (노이즈) → x_{T-1} → ... → x_1 → x_0 (이미지)
```

### 1.3 학습

신경망에 (x_t, t) 입력 → 노이즈 예측. MSE 로 학습.

```python
loss = ((noise_pred - noise_actual) ** 2).mean()
```

학습 후, 추론 시 *T 단계 반복* 으로 이미지 생성.

---

## 2. DDIM — 가속

DDPM 은 *T = 1000 단계 추론* 필요 → 매우 느림.

DDIM (2020) 은 *결정론적 샘플링* 으로 *50-100 단계* 만으로 같은 품질 달성.

---

## 3. Latent Diffusion (Stable Diffusion)

### 3.1 발상

512×512×3 = 786K 차원 이미지에서 직접 디퓨전 → 매우 비쌈.

VAE 로 *압축된 잠재 공간 (64×64×4 = 16K 차원)* 에서 디퓨전 → 50 배 효율.

### 3.2 구조

```
[학습]
이미지 → VAE Encoder → 잠재 → 노이즈 추가 → U-Net (노이즈 예측) ← (이미지 캡션 임베딩)

[추론]
가우시안 노이즈 → U-Net 반복 (조건: 텍스트) → 잠재 → VAE Decoder → 이미지
```

이 구조가 *Stable Diffusion 의 핵심*. SDXL, FLUX 도 같은 골격.

---

## 4. CFG — Classifier-Free Guidance

### 4.1 발상

조건부 (텍스트 있음) 와 무조건부 (텍스트 없음) 예측을 *동시에 학습*. 추론 시:

$$\hat{\epsilon} = \epsilon_\text{uncond} + s \cdot (\epsilon_\text{cond} - \epsilon_\text{uncond})$$

s (CFG scale) 가 클수록 *프롬프트에 더 충실*. 일반 7-15 범위.

### 4.2 결과

CFG 가 *프롬프트 일치도를 크게 향상*. 거의 모든 디퓨전 모델이 사용.

---

## 5. Rectified Flow — Flux 의 핵심

### 5.1 발상

DDPM 의 *곡선 경로* 를 *직선 경로* 로 만들어 *훨씬 적은 단계* 로 가능.

FLUX, Stable Diffusion 3 가 채택. *4-8 단계 추론* 가능.

---

## 6. Distillation — 더 빠른 추론

큰 디퓨전 모델 (50 단계) 의 *출력을 작은 모델이 1-4 단계로 모방*.

대표: SDXL Lightning (4 단계), FLUX.2 Klein (4 단계).

---

## 7. ControlNet·LoRA·IP-Adapter

생성 제어 기법:

- **ControlNet** — 외부 조건 (스케치·자세·깊이) 으로 생성 통제
- **LoRA** — 작은 어댑터로 스타일·도메인 추가
- **IP-Adapter** — 참조 이미지로 스타일 전달

자세한 내용은 Vol 81 (생성 제어).

---

## 권 정리

- DDPM = 노이즈 추가·제거 패러다임
- DDIM = 가속 (50-100 단계)
- Latent Diffusion (Stable Diffusion) = VAE 압축 + 디퓨전
- CFG = 프롬프트 충실도 향상
- Rectified Flow (FLUX) = 직선 경로, 매우 적은 단계
- Distillation = 더 빠른 추론

가장 기억할 한 줄: **"Image Diffusion 은 노이즈에서 이미지로 점진적 복원이며, Latent Diffusion + CFG 가 산업 표준 골격이다."**

다음 권: [Volume 81 — 생성 제어 기법](./volume_81_generation_control.md)

---

## 자가점검 키워드

`DDPM`, `DDIM`, `Latent Diffusion`, `Stable Diffusion`, `CFG`, `Rectified Flow`, `Distillation`

## 자가점검 질문

1. DDPM 의 순방향·역방향 과정을 그리십시오.
2. DDIM 이 DDPM 보다 빠른 이유를 설명하십시오.
3. Latent Diffusion 의 효율성 비밀을 적으십시오.
4. CFG 의 수식과 효과를 설명하십시오.
5. Rectified Flow 가 디퓨전을 가속하는 메커니즘을 적으십시오.

## 다음 권

[Volume 81 — 생성 제어 기법](./volume_81_generation_control.md)
