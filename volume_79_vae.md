# Volume 79 — VAE 와 잠재 변수 모델

> 이 권이 끝나면 *데이터의 잠재 공간* 이라는 표현이 추상적이지 않게 손에 잡히게 됩니다.

## 목적

VAE (Variational Autoencoder) 는 *데이터를 잠재 변수로 표현하고 재생성* 하는 확률적 모델입니다. 디퓨전 모델·LDM·표현 학습의 직접적 조상이며, 잠재 공간이라는 *현대 ML 의 핵심 개념* 을 제공한 모델입니다.

## 선수 지식

- Volume 11, 12, 32 완료

## 학습 결과

1. AE 와 VAE 의 차이를 알 수 있습니다.
2. ELBO 의 *재구성 + KL* 두 항의 의미를 이해합니다.
3. *재매개화 트릭 (reparameterization)* 의 발상을 안다.
4. Latent Diffusion 에서 VAE 가 하는 역할을 설명합니다.

---

## 1. AutoEncoder (AE)

### 1.1 기본 구조

```
입력 x → [Encoder] → 잠재 z → [Decoder] → 재구성 x'

손실 = ||x - x'||²
```

장점: 데이터를 *낮은 차원으로 압축*. 단점: 잠재 공간이 *연속·구조 없음* — 새 샘플 생성 어려움.

---

## 2. VAE — 확률적 AE

### 2.1 차이

VAE 는 *잠재 z 를 분포로* 다룸:

```
입력 x → [Encoder] → (μ, σ) → z ~ N(μ, σ²) → [Decoder] → 재구성 x'
```

Encoder 가 *결정론적 z* 가 아닌 *분포 (μ, σ)* 를 출력. 그 분포에서 *샘플링* 한 z 로 디코딩.

### 2.2 손실 — ELBO

$$\text{ELBO} = E[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

- 첫 항: *재구성 손실* — 디코더가 입력을 잘 복원
- 둘째 항: *KL 항* — 인코더 분포가 사전 분포 (보통 표준 정규) 와 가까워야

KL 항이 *생성 가능한 잠재 공간* 을 만드는 핵심.

---

## 3. 재매개화 트릭

샘플링은 *미분 불가능* 해 역전파 어려움. 트릭:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim N(0, 1)$$

샘플링을 *결정론적 변환 + 외부 노이즈* 로 분리. 그래디언트가 μ, σ 로 흐름.

---

## 4. PyTorch 미니 VAE

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_dim=784, latent_dim=20):
        super().__init__()
        self.enc_fc1 = nn.Linear(in_dim, 400)
        self.enc_mu = nn.Linear(400, latent_dim)
        self.enc_logvar = nn.Linear(400, latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim, 400)
        self.dec_fc2 = nn.Linear(400, in_dim)
    
    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl
```

---

## 5. Latent Diffusion 에서의 VAE

Stable Diffusion 같은 *Latent Diffusion Model (LDM)* 은 VAE 를 *압축 도구* 로 사용:

```
이미지 (512×512×3) → VAE Encoder → 잠재 (64×64×4)
                                       ↓
                              디퓨전 모델 (잠재 공간에서)
                                       ↓
잠재 → VAE Decoder → 이미지 복원
```

장점: 디퓨전이 *낮은 차원에서 동작* → 빠르고 메모리 적음.

이것이 *Stable Diffusion 의 효율성 비밀*.

---

## 6. β-VAE 와 disentanglement

KL 항에 가중치 β 를 더해:

$$L = \text{Recon} + \beta \cdot \text{KL}$$

큰 β 는 *각 잠재 차원이 다른 의미* 를 표현하게 (disentangle). 해석 가능성 향상.

---

## 권 정리

- AE = 결정론적 압축, VAE = 확률적 (생성 가능)
- ELBO = 재구성 + KL
- 재매개화 트릭이 그래디언트 흐름 가능하게
- Latent Diffusion = VAE 압축 + 디퓨전 학습 + VAE 복원
- β-VAE = disentangled 표현

가장 기억할 한 줄: **"VAE 는 잠재 공간을 확률 분포로 만들어 생성을 가능하게 하며, Stable Diffusion 의 압축 도구로 살아남았다."**

다음 권: [Volume 80 — Image Diffusion](./volume_80_image_diffusion.md)

---

## 자가점검 키워드

`AE`, `VAE`, `ELBO`, `재매개화`, `Latent Diffusion`, `β-VAE`, `disentanglement`

## 자가점검 질문

1. AE 와 VAE 의 본질적 차이를 적으십시오.
2. ELBO 의 두 항의 의미를 설명하십시오.
3. 재매개화 트릭의 발상을 적으십시오.
4. Stable Diffusion 에서 VAE 가 하는 역할을 설명하십시오.

## 다음 권

[Volume 80 — Image Diffusion](./volume_80_image_diffusion.md)
