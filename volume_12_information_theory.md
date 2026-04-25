# Volume 12 — 정보 이론

> 이 권이 끝나면 교차엔트로피 손실이 등장한 이유와 KL 발산이 모델 학습에서 어떤 역할을 하는지를 설명할 수 있게 됩니다.

## 목적

엔트로피·교차엔트로피·KL 발산은 분류·생성·강화학습·디퓨전·LLM 정렬의 손실 함수에 일관되게 등장합니다. 이 세 개념은 모두 *확률 분포 사이의 거리/혼란도* 를 측정하는 도구이며, 한 번 정확히 이해해 두면 이후 어떤 모델을 만나도 손실 함수의 기원을 추적할 수 있습니다. 이 권은 정보 이론을 *수식 암기* 가 아닌 *직관 + 코드* 로 익히는 데 집중합니다.

## 선수 지식

- Volume 11 완료
- 외부 지식: 로그 함수의 기본 성질

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 엔트로피를 *분포의 불확실성* 으로 설명할 수 있습니다.
2. 교차엔트로피가 *예측 분포로 실제 분포를 부호화할 때 평균 비트 수* 임을 설명할 수 있습니다.
3. KL 발산이 *대칭이 아닌 거리 유사 척도* 임을 보일 수 있습니다.
4. 분류 모델의 손실이 왜 교차엔트로피인지를 유도할 수 있습니다.
5. VAE·디퓨전·RLHF 에 KL 이 등장하는 이유를 한 줄로 설명할 수 있습니다.

---

## 이 권을 읽기 전에

정보 이론은 1948 년 Claude Shannon 의 한 논문에서 시작된 분야입니다. 원래 목적은 *통신 채널에서의 신호 전송* 이었지만, 그 도구들은 ML 의 모든 곳에 들어와 있습니다.

이 권은 정보 이론의 가장 기본적이고 ML 에 직접 관계되는 *세 개념 (엔트로피·교차엔트로피·KL)* 만 다룹니다. 이 셋을 정확히 이해하면 ML 의 모든 손실 함수와 정칙화 기법이 *통합된 시각* 으로 보입니다.

수식과 함께 *직관적 예시 (동전·주사위·문자)* 를 동반합니다. NumPy 로 계산해 봐서 *값의 의미* 를 손에 잡히게 만드시기 바랍니다.

---

## 1. 정보량과 엔트로피

### 1.1 정보량의 정의

확률 $p$ 인 사건이 일어났을 때의 *정보량* 은:

$$I(x) = -\log p(x)$$

직관:
- *흔한 일* (p 가 큼) → 정보량 작음 ("내일 해가 뜬다" 는 거의 정보 없음)
- *드문 일* (p 가 작음) → 정보량 큼 ("내일 일식이 일어난다" 는 큰 정보)
- *확실한 일* (p = 1) → 정보량 0

로그의 밑이 2 면 단위는 *비트 (bit)*, 자연로그면 *내트 (nat)* 입니다. ML 에서는 보통 자연로그를 사용합니다.

### 1.2 엔트로피의 정의

분포 P 의 *엔트로피 (entropy)* 는 *정보량의 기댓값* 입니다.

$$H(P) = -\sum_x p(x) \log p(x)$$

직관: *분포가 얼마나 불확실한가* 의 측정.

- *동전이 항상 앞면* (p = (1, 0)) → H = 0 (확실, 불확실성 없음)
- *공정한 동전* (p = (0.5, 0.5)) → H = log 2 ≈ 0.693 (최대 불확실성)
- *편향 동전* (p = (0.9, 0.1)) → H = 0.325 (중간)

### 1.3 NumPy 로 엔트로피

```python
import numpy as np

def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]  # 0 은 0*log0 = 0 으로 처리
    return -np.sum(p * np.log(p))

print(entropy([1.0, 0.0]))         # 0.0       (확실)
print(entropy([0.5, 0.5]))         # 0.693     (최대)
print(entropy([0.9, 0.1]))         # 0.325     (중간)
print(entropy([0.25]*4))           # 1.386     (균등 4 개)
print(entropy([1/6]*6))            # 1.792     (공정 주사위)
```

K 개의 가능한 결과에서 *균등 분포* 가 *최대 엔트로피* 를 가집니다 ($\log K$).

### 1.4 ML 에서의 엔트로피

- *분류 출력 분포의 엔트로피* — 모델이 얼마나 *확신* 하는가의 측정. 낮으면 확신, 높으면 불확실
- *입력 분포의 엔트로피* — 데이터셋의 다양성 측정
- *최대 엔트로피 정칙화* — 정책 분포가 너무 한 행동에 집중되지 않게 (RL 에서)

### 1.5 챕터 정리

엔트로피는 *분포의 불확실성* 이며, 정보량의 기댓값입니다. 균등 분포가 최대, 한 점에 집중된 분포가 최소입니다. ML 에서는 *모델 확신도·데이터 다양성·정책 다양성* 등 다양한 곳에 등장합니다.

---

## 2. 교차엔트로피

### 2.1 교차엔트로피의 정의

분포 P (실제) 와 Q (예측) 의 *교차엔트로피 (cross entropy)*:

$$H(P, Q) = -\sum_x p(x) \log q(x)$$

해석: *Q 분포로 P 분포를 부호화할 때 평균 비트 수*.

### 2.2 의미

- $H(P, Q) \geq H(P)$ 항상 성립 (Q 가 P 와 같을 때 등호)
- $H(P, Q) = H(P)$ 일 때 *Q 가 P 를 가장 잘 표현*
- $H(P, Q) - H(P) = D_{KL}(P \| Q)$ — 다음 챕터의 *KL 발산*

직관: *예측 분포 Q 가 실제 분포 P 와 다를수록 교차엔트로피가 크다*.

### 2.3 NumPy 로 교차엔트로피

```python
import numpy as np

def cross_entropy(p, q, eps=1e-12):
    p = np.asarray(p)
    q = np.asarray(q)
    return -np.sum(p * np.log(q + eps))

# 실제 정답: 클래스 0
p = np.array([1.0, 0.0, 0.0])

q1 = np.array([0.99, 0.005, 0.005])  # 거의 정답
q2 = np.array([0.5, 0.25, 0.25])     # 어중간
q3 = np.array([0.01, 0.495, 0.495])  # 거의 틀림

print(cross_entropy(p, q1))   # 0.010   (작음)
print(cross_entropy(p, q2))   # 0.693   (중간)
print(cross_entropy(p, q3))   # 4.605   (큼)
```

*예측이 정답에 가까울수록 교차엔트로피가 작습니다*. 따라서 *교차엔트로피를 최소화 = 예측을 정답에 맞추기*.

### 2.4 분류 손실 = 교차엔트로피

다중 분류에서 정답 라벨 y 가 one-hot 으로 표현되면 ($p(y) = 1$, 나머지 0):

$$\text{Loss} = H(P, Q) = -\log \hat{p}_y$$

이것이 *Categorical Cross Entropy (CCE)* 손실입니다.

이진 분류 (베르누이) 의 경우:

$$\text{Loss} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

이것이 *Binary Cross Entropy (BCE)* 손실입니다.

### 2.5 챕터 정리

교차엔트로피는 *예측 분포로 실제 분포를 부호화할 때 평균 비트 수* 이며, 예측이 정답에 가까울수록 작아집니다. 분류 손실 (BCE, CCE) 은 모두 교차엔트로피의 형태입니다.

---

## 3. KL 발산

### 3.1 KL 발산의 정의

*Kullback-Leibler 발산 (KL divergence)*:

$$D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

해석: *P 와 Q 가 얼마나 다른가* 의 측정.

### 3.2 KL 의 성질

- $D_{KL}(P \| Q) \geq 0$ 항상
- $D_{KL}(P \| Q) = 0 \Leftrightarrow P = Q$
- *비대칭*: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ 일반적으로

비대칭이라는 점이 중요합니다. KL 은 *진짜 거리* 가 아니라 *유사 거리* 입니다.

### 3.3 교차엔트로피·엔트로피·KL 의 관계

$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

이 식이 의미하는 것:

- *교차엔트로피 = 엔트로피 (피할 수 없는 불확실성) + KL (예측이 정답과 다른 정도)*
- *KL = 교차엔트로피 - 엔트로피*

따라서 학습이 진행되면서 *교차엔트로피가 줄어드는 것 = KL 이 줄어드는 것* 입니다 (P 의 엔트로피는 데이터에 의해 고정).

### 3.4 NumPy 로 KL

```python
import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(p * np.log((p + eps) / (q + eps)))

p = np.array([0.5, 0.5])
q1 = np.array([0.5, 0.5])    # 같음
q2 = np.array([0.9, 0.1])    # 다름
q3 = np.array([0.1, 0.9])    # 다름 (반대 방향)

print(kl_divergence(p, q1))   # 0.0
print(kl_divergence(p, q2))   # 0.368
print(kl_divergence(p, q3))   # 0.368

# 비대칭성
print(kl_divergence(q2, p))   # 0.510  (다른 값)
```

### 3.5 ML 에서 KL 의 등장

KL 은 ML 의 곳곳에 등장합니다.

- *VAE 의 ELBO* — 사후 분포가 사전 분포에서 멀어지지 않게 하는 KL 항
- *디퓨전 모델 학습* — 노이즈 분포와 모델 출력의 KL
- *RLHF / DPO* — 정책이 사전학습 분포에서 멀어지지 않게 하는 KL 페널티
- *Variational Inference* — 근사 분포와 진짜 사후의 KL 최소화
- *지식 증류* — Teacher 와 Student 분포의 KL

이 모든 응용에서 *KL 은 두 분포 사이의 거리 페널티* 로 작동합니다.

### 3.6 챕터 정리

KL 발산은 *두 분포가 얼마나 다른가* 를 측정하며, *비대칭* 이라 진짜 거리가 아닙니다. 교차엔트로피 = 엔트로피 + KL 의 관계를 통해 *분류 학습 = KL 최소화* 임을 알 수 있습니다. ML 의 곳곳 (VAE, 디퓨전, RLHF, 증류) 에 등장합니다.

---

## 4. JS 발산과 대칭 거리

### 4.1 JS 발산

KL 의 비대칭이 불편한 경우 *Jensen-Shannon 발산 (JSD)* 을 사용합니다.

$$D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$$

여기서 $M = (P + Q) / 2$.

JS 는 *대칭* 이고 *항상 유한* 합니다 (KL 은 P 가 0 이 아닌데 Q 가 0 이면 무한대).

### 4.2 ML 에서의 JS

- *GAN 의 원래 목적 함수* — JS 발산 최소화로 유도
- *모델 비교* — 두 모델 출력 분포의 차이
- *오토인코더 평가* — 입력과 재구성의 분포 차이

### 4.3 Wasserstein 거리

KL/JS 외에 *Wasserstein 거리* (Earth-Mover's distance) 가 있으며, *분포가 겹치지 않을 때도 의미 있는 거리* 를 줍니다. WGAN 의 핵심.

### 4.4 챕터 정리

JS 는 KL 의 *대칭화 변형*, Wasserstein 은 *분포가 겹치지 않을 때도 작동* 하는 거리입니다. 응용 영역에 따라 적절한 거리를 선택합니다.

---

## 5. 분류 손실의 유도

### 5.1 소프트맥스

다중 분류 모델의 출력은 *각 클래스의 점수* 입니다. 이를 *확률 분포로* 변환하는 것이 *소프트맥스 (softmax)*:

$$\text{softmax}(z)_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

특성:
- 모든 출력이 0 이상
- 합이 1
- 큰 점수가 큰 확률을 받음 (지수적으로)

### 5.2 소프트맥스 + 교차엔트로피

분류 모델의 표준 손실:

$$L = -\sum_k y_k \log p_k$$

여기서 $y$ 는 one-hot 정답, $p = \text{softmax}(z)$ 는 모델 출력.

### 5.3 NumPy 로 직접 구현

```python
import numpy as np

def softmax(z):
    z = z - z.max()  # 수치 안정화
    e = np.exp(z)
    return e / e.sum()

def cross_entropy_loss(z, y_true):
    p = softmax(z)
    return -np.log(p[y_true] + 1e-12)

# 예시
z = np.array([2.0, 1.0, 0.5])  # 모델 점수
y = 0                           # 정답 클래스

loss = cross_entropy_loss(z, y)
p = softmax(z)
print(f"확률: {p}")
print(f"손실: {loss:.4f}")
```

이 6 줄이 *분류 모델의 손실 계산* 의 본질입니다.

### 5.4 그래디언트의 우아함

소프트맥스 + 교차엔트로피의 그래디언트는 매우 단순합니다.

$$\frac{\partial L}{\partial z_k} = p_k - y_k$$

*예측 - 정답* 의 단순한 형태입니다. 이 우아함이 분류 모델 학습이 안정적인 한 이유입니다.

### 5.5 챕터 정리

소프트맥스 + 교차엔트로피는 *분류 모델의 표준 출력층* 입니다. 그래디언트가 *예측 - 정답* 으로 단순해 학습이 안정적입니다.

---

## 6. 레이블 스무딩과 엔트로피 정칙화

### 6.1 레이블 스무딩

표준 교차엔트로피는 정답을 *완벽한 one-hot* 으로 다룹니다. 그러나 모델이 *지나치게 확신* 하면 일반화가 나빠집니다.

*레이블 스무딩 (Label Smoothing)* 은 정답 라벨을 약간 *부드럽게* 만듭니다.

원래: $y = (0, 0, 1, 0, 0)$
스무딩: $y' = (0.025, 0.025, 0.9, 0.025, 0.025)$ (예: $\alpha = 0.1$, $K = 5$)

수식:

$$y'_k = (1 - \alpha) \, y_k + \frac{\alpha}{K}$$

이 단순한 수정으로 모델의 *과확신* 이 줄고 일반화가 좋아집니다.

### 6.2 엔트로피 정칙화

손실에 *예측 분포의 음의 엔트로피* 를 더해 *분포가 너무 한 점에 집중되지 않게* 합니다.

$$L = L_{\text{CE}} - \beta H(p)$$

엔트로피가 클수록 손실이 작아지므로, 모델이 *불확실성을 유지* 하도록 유도됩니다.

RL 의 *정책 엔트로피 보너스* 도 같은 사고입니다.

### 6.3 챕터 정리

레이블 스무딩과 엔트로피 정칙화는 모두 *모델이 과도하게 확신하지 않게* 만드는 도구이며, 일반화 성능을 끌어올립니다.

---

## 7. VAE 의 ELBO

### 7.1 VAE 의 손실

VAE (Variational Autoencoder) 는 *재구성 + KL* 두 항으로 학습됩니다.

$$\text{ELBO} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \| p(z))$$

- 첫 항: *재구성 손실* — 인코더가 만든 z 로 디코더가 x 를 얼마나 잘 복원하는가
- 둘째 항: *KL 항* — 인코더 출력 분포가 사전 분포 (보통 표준 정규) 에서 멀어지지 않게

이 두 항이 결합되어 *생성 가능한 잠재 공간* 을 만듭니다.

### 7.2 KL 항의 역할

KL 항이 없다면 인코더는 *각 입력에 대해 서로 다른 멀리 떨어진 z* 를 출력해 디코더가 외우게 만들 수 있습니다. KL 이 있으면 *모든 z 가 사전 분포 근처* 에 모여, 사전 분포에서 *샘플링한 새 z 도 의미 있는 출력을 만들* 수 있게 됩니다.

### 7.3 챕터 정리

VAE 의 ELBO 는 *재구성 + KL* 의 두 항이며, KL 이 없으면 *생성 능력* 이 사라집니다. 정보 이론의 KL 이 *생성 모델의 핵심 부품* 임을 보여 줍니다.

---

## 8. RLHF 의 KL 페널티

### 8.1 RLHF 의 보상

RLHF (Reinforcement Learning from Human Feedback) 는 인간 선호로 학습된 보상 모델 R 의 출력을 최대화합니다.

$$\max_\pi \mathbb{E}_{x \sim \pi}[R(x)]$$

그러나 보상 모델만 따라가면 *정책이 사전학습 분포에서 너무 멀어져* 이상한 응답을 만들 수 있습니다 (*reward hacking*).

### 8.2 KL 페널티

해법은 *사전학습된 정책 $\pi_{\text{ref}}$ 에서의 KL 을 페널티* 로 더하는 것입니다.

$$\max_\pi \mathbb{E}_{x \sim \pi}[R(x)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$$

이 페널티가 *정책이 사전학습 분포 근처에 머물게* 합니다.

### 8.3 DPO 도 같은 발상

DPO (Direct Preference Optimization) 는 RLHF 의 PPO 부분을 단순화한 것이지만, *사전학습 분포에서 멀어지지 않게* 하는 KL 항이 같은 역할을 합니다.

### 8.4 챕터 정리

RLHF 와 DPO 모두 *KL 페널티* 로 정책이 사전학습 분포에서 멀어지지 않게 합니다. 정보 이론의 KL 이 *LLM 정렬의 핵심 부품* 입니다.

---

## 9. NumPy 로 엔트로피 계산해 보기

### 9.1 종합 실습

```python
import numpy as np
import matplotlib.pyplot as plt

# 베르누이 분포의 엔트로피 vs p
ps = np.linspace(0.001, 0.999, 100)
hs = -ps * np.log(ps) - (1-ps) * np.log(1-ps)

plt.figure(figsize=(8, 5))
plt.plot(ps, hs)
plt.axvline(0.5, color='gray', linestyle='--', label='최대 엔트로피 (p=0.5)')
plt.xlabel('p'); plt.ylabel('Entropy (nats)')
plt.title('베르누이 엔트로피')
plt.legend(); plt.grid(True)
plt.show()
```

### 9.2 KL 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# Q = N(μ, 1), P = N(0, 1) 의 KL
mus = np.linspace(-3, 3, 100)

# 가우시안의 KL: D_KL(P||Q) = log(σ_Q/σ_P) + (σ_P^2 + (μ_P - μ_Q)^2) / (2 σ_Q^2) - 0.5
kls_pq = mus**2 / 2  # σ=1 가정 시
kls_qp = mus**2 / 2

plt.figure(figsize=(8, 5))
plt.plot(mus, kls_pq, label='D_KL(P||Q)')
plt.plot(mus, kls_qp, '--', label='D_KL(Q||P)')
plt.xlabel('μ_Q'); plt.ylabel('KL divergence')
plt.title('가우시안 분포 사이의 KL')
plt.legend(); plt.grid(True)
plt.show()
```

### 9.3 분류 학습 시뮬레이션

```python
import numpy as np

np.random.seed(0)

# 3 클래스, 100 샘플
n_classes = 3
n_samples = 100
features = 5

W = np.random.randn(features, n_classes) * 0.1
b = np.zeros(n_classes)

X = np.random.randn(n_samples, features)
y_true = np.random.randint(0, n_classes, n_samples)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

# 학습 루프
lr = 0.1
for step in range(200):
    z = X @ W + b
    p = softmax(z)
    
    # 손실
    loss = -np.log(p[np.arange(n_samples), y_true] + 1e-12).mean()
    
    # 그래디언트 (소프트맥스 + CE 의 우아한 형태)
    p[np.arange(n_samples), y_true] -= 1.0  # p - y
    grad_z = p / n_samples
    grad_W = X.T @ grad_z
    grad_b = grad_z.sum(axis=0)
    
    W -= lr * grad_W
    b -= lr * grad_b
    
    if step % 20 == 0:
        accuracy = (softmax(X @ W + b).argmax(axis=1) == y_true).mean()
        print(f"step {step}: loss={loss:.4f}, accuracy={accuracy:.2%}")
```

NumPy 만으로 *분류 모델의 학습 루프 전체* 가 구현됩니다.

### 9.4 챕터 정리

NumPy 만으로 *엔트로피·KL·분류 학습* 모두를 구현할 수 있습니다. 직접 손으로 코드를 짠 경험이 PyTorch 의 *고수준 API* 이해의 기반이 됩니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **정보량과 엔트로피** — 분포의 불확실성 측정.
- **교차엔트로피** — 예측으로 정답을 부호화할 때 평균 비트 수. 분류 손실의 본질.
- **KL 발산** — 두 분포의 차이. 비대칭. 교차엔트로피 - 엔트로피 = KL.
- **JS 와 Wasserstein** — KL 의 대칭/일반 변형.
- **소프트맥스 + 교차엔트로피** — 분류의 표준 출력층, 그래디언트가 *예측 - 정답* 으로 단순.
- **레이블 스무딩과 엔트로피 정칙화** — 모델의 과확신 방지.
- **VAE 의 ELBO** — 재구성 + KL. KL 이 *생성 가능한 잠재 공간* 의 핵심.
- **RLHF / DPO 의 KL 페널티** — 정책이 사전학습 분포에서 멀어지지 않게.

가장 기억에 남겨야 할 한 줄은 **"분류 학습 = 두 분포의 KL 최소화이며, KL 은 생성·정렬의 핵심 부품이다."** 입니다.

다음 권은 [Volume 13 — NumPy 로 다시 푸는 수학](./volume_13_numpy.md) 입니다. 거기서는 지금까지의 모든 수학 개념을 NumPy 코드로 종합합니다.

---

## 자가점검 키워드

`엔트로피`, `교차엔트로피`, `KL 발산`, `JS 발산`, `소프트맥스`, `레이블 스무딩`, `ELBO`, `KL 페널티`

## 자가점검 질문

1. 엔트로피의 정의를 적고, 베르누이 분포에서 엔트로피가 최대가 되는 p 를 손으로 구하십시오.
2. 교차엔트로피·엔트로피·KL 의 관계식을 적고 의미를 설명하십시오.
3. KL 발산이 *대칭이 아니라는 사실* 의 ML 함의는 무엇입니까?
4. 다중 분류의 손실 함수를 *소프트맥스 + 교차엔트로피* 로 NumPy 5 줄로 구현하십시오.
5. 레이블 스무딩이 일반화를 개선하는 메커니즘을 한 문단으로 설명하십시오.
6. VAE 의 ELBO 두 항 (재구성, KL) 의 역할을 각각 한 문장으로 적으십시오.
7. RLHF 에서 KL 페널티가 없다면 어떤 문제가 발생합니까?

## 다음 권

[Volume 13 — NumPy 로 다시 푸는 수학](./volume_13_numpy.md)
