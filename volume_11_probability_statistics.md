# Volume 11 — 확률과 통계

> 이 권이 끝나면 모든 ML 모델이 본질적으로 *확률 분포를 추정하는 일* 임을 이해하게 됩니다.

## 목적

ML·DL 의 모든 손실 함수는 *확률적 가정* 에서 유도됩니다. 회귀의 MSE 는 가우시안 잡음 가정에서 나오고, 분류의 교차엔트로피는 다항 분포 가정에서 나오며, 생성 모델은 데이터의 확률 분포 자체를 학습합니다. 확률과 통계의 기초를 다지지 않으면 *왜 이 손실 함수를 써야 하는가* 에 답할 수 없습니다. 이 권은 그 기초를 직관 위주로 다집니다.

## 선수 지식

- Volume 8, 6 완료
- 외부 지식: 확률의 기본 개념(주사위·동전), 평균·분산

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 확률 변수와 확률 분포의 차이를 구분할 수 있습니다.
2. 베르누이·이항·정규·범주 분포의 PDF/PMF 를 적을 수 있습니다.
3. 베이즈 정리를 사후 확률 추론에 적용할 수 있습니다.
4. MLE 와 MAP 의 차이를 한 문단으로 설명할 수 있습니다.
5. 로그가능도와 손실 함수의 관계를 보일 수 있습니다.

---

## 이 권을 읽기 전에

확률은 *불확실성을 다루는 언어* 입니다. ML 의 모든 결정에는 불확실성이 따라오므로, 확률은 *선택* 이 아니라 *기본 어휘* 입니다.

학교에서 배운 확률·통계가 흐릿하게 기억나는 분이 많을 것입니다. 이 권은 그 기억을 *ML 에 필요한 부분만 골라* 다시 다집니다. 학교 시험을 위한 공식 암기가 아니라, *분류 손실이 왜 교차엔트로피인가*·*VAE 의 ELBO 가 왜 두 항으로 나뉘는가* 같은 질문에 답하기 위한 도구를 만드는 것이 목표입니다.

NumPy 와 Scipy 로 모든 분포·통계량을 직접 계산해 봅니다. 식을 외우지 마시고 코드로 결과를 확인하시기 바랍니다.

---

## 1. 확률 변수와 확률 분포

### 1.1 확률 변수

**확률 변수 (random variable)** 는 *불확실한 결과를 숫자로 표현하는 함수* 입니다. 동전 던지기의 결과를 *앞=1, 뒤=0* 으로 표현하면, 그 *0 또는 1* 이 확률 변수입니다.

확률 변수는 두 종류:
- **이산 확률 변수** — 가능한 값이 *셀 수 있는* 경우 (동전, 주사위, 클래스 라벨)
- **연속 확률 변수** — 가능한 값이 *실수 구간 위* 인 경우 (키, 몸무게, 온도)

ML 에서 자주 등장하는 확률 변수:
- 분류 모델의 *예측 클래스* (이산)
- 회귀 모델의 *예측 값* (연속)
- 생성 모델의 *생성된 토큰/픽셀* (이산/연속)

### 1.2 확률 분포

**확률 분포 (probability distribution)** 는 *확률 변수가 각 값을 가질 확률* 을 모두 모아 둔 함수입니다.

이산 확률 변수에 대해서는 *확률 질량 함수 (PMF: Probability Mass Function)*:

$$P(X = x_i) = p_i, \quad \sum_i p_i = 1$$

연속 확률 변수에 대해서는 *확률 밀도 함수 (PDF: Probability Density Function)*:

$$\int_{-\infty}^{\infty} p(x) \, dx = 1$$

연속 분포에서 *한 점의 확률* 은 0 이며 (확률은 *구간* 의 적분으로만 정의), *밀도* 가 의미를 가집니다.

### 1.3 NumPy 로 분포 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# 정규 분포 샘플
samples = np.random.normal(loc=0, scale=1, size=10000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(samples, bins=50, density=True)
axes[0].set_title('정규 분포 N(0, 1) 샘플 히스토그램')

# 이론적 PDF
x = np.linspace(-4, 4, 200)
pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
axes[0].plot(x, pdf, 'r-', linewidth=2, label='이론 PDF')
axes[0].legend(); axes[0].grid(True)

# 베르누이 분포 (p=0.3) 샘플
bern = np.random.binomial(n=1, p=0.3, size=1000)
axes[1].bar([0, 1], [np.mean(bern == 0), np.mean(bern == 1)])
axes[1].set_title('베르누이 분포 Bern(0.3)')
axes[1].set_xticks([0, 1]); axes[1].grid(True)
plt.show()
```

### 1.4 챕터 정리

확률 변수는 *불확실한 결과를 숫자로* 표현하는 함수, 확률 분포는 *각 값의 확률* 을 모은 함수입니다. 이산은 PMF, 연속은 PDF 로 표현됩니다. 다음 챕터에서는 ML 에서 자주 등장하는 *주요 분포* 들을 봅니다.

---

## 2. 주요 분포

### 2.1 베르누이 분포

가장 단순한 분포. *0 또는 1* 두 값만 가집니다.

$$P(X=1) = p, \quad P(X=0) = 1-p$$

ML 응용: *이진 분류* 의 정답 라벨, 한 픽셀의 *흑/백* 결과.

```python
import numpy as np

# 동전이 앞면 나올 확률 0.6 인 베르누이
samples = np.random.binomial(n=1, p=0.6, size=1000)
print(f"평균: {samples.mean():.3f}")  # 0.6 에 가까움
```

### 2.2 이항 분포

*베르누이 시행을 n 번 반복* 했을 때 *성공 횟수* 의 분포.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

ML 응용: 100 번의 분류 중 정답 횟수, A/B 테스트의 변환 횟수.

### 2.3 정규 분포 (가우시안)

가장 흔한 연속 분포. *종 모양* 의 곡선.

$$p(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $\mu$ — 평균 (분포의 중심)
- $\sigma$ — 표준편차 (분포의 폭)

ML 응용:
- *회귀의 노이즈 가정*
- *가중치 초기화* (He, Xavier)
- *VAE 의 사전 분포*
- *디퓨전 모델의 노이즈*

### 2.4 범주 분포 (Categorical)

K 개의 클래스 중 *하나가 선택* 되는 분포.

$$P(X = k) = p_k, \quad \sum_k p_k = 1$$

ML 응용: *다중 분류* 의 정답, *생성 모델의 다음 토큰* 분포 (소프트맥스 출력).

### 2.5 디리클레 분포

*범주 분포의 파라미터 자체* 를 분포로 표현하는 *2차 분포*.

ML 응용: *LDA 토픽 모델*, *베이지안 분류*.

### 2.6 NumPy 로 분포 비교

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 베르누이
ax = axes[0, 0]
samples = np.random.binomial(1, 0.7, 10000)
ax.bar([0, 1], [np.mean(samples==0), np.mean(samples==1)])
ax.set_title('Bernoulli(p=0.7)')

# 이항
ax = axes[0, 1]
samples = np.random.binomial(20, 0.3, 10000)
ax.hist(samples, bins=range(22), density=True)
ax.set_title('Binomial(n=20, p=0.3)')

# 정규
ax = axes[1, 0]
samples = np.random.normal(0, 1, 10000)
ax.hist(samples, bins=50, density=True)
ax.set_title('Normal(0, 1)')

# 다항
ax = axes[1, 1]
counts = np.random.multinomial(100, [0.2, 0.3, 0.5], 1)[0]
ax.bar([0, 1, 2], counts)
ax.set_title('Multinomial(n=100, p=[0.2, 0.3, 0.5])')

plt.tight_layout()
plt.show()
```

### 2.7 챕터 정리

베르누이·이항·정규·범주·디리클레가 ML 에서 가장 자주 등장하는 분포입니다. 각 분포의 *파라미터와 모양* 을 익혀 두면 손실 함수의 유도가 자연스럽게 따라옵니다. 다음 챕터에서는 분포의 *요약 통계* 인 기댓값·분산을 봅니다.

---

## 3. 기댓값·분산·공분산

### 3.1 기댓값

확률 변수 X 의 *기댓값 (expected value)* 은 *값들의 확률 가중 평균* 입니다.

$$E[X] = \sum_i x_i p_i \quad \text{(이산)}$$

$$E[X] = \int x \, p(x) \, dx \quad \text{(연속)}$$

직관: *수많은 시행을 했을 때 평균적으로 나오는 값*.

### 3.2 분산과 표준편차

*분산 (variance)* 은 *값들이 평균에서 얼마나 떨어져 있는가* 의 평균.

$$\text{Var}[X] = E[(X - \mu)^2] = E[X^2] - \mu^2$$

*표준편차* 는 분산의 제곱근. 단위가 X 와 같아 더 직관적.

### 3.3 공분산과 상관

*공분산 (covariance)* 은 *두 변수가 함께 변하는 정도*.

$$\text{Cov}[X, Y] = E[(X-\mu_X)(Y-\mu_Y)]$$

*상관 계수 (correlation)* 는 공분산을 *각자의 표준편차로 정규화*.

$$\rho = \frac{\text{Cov}[X,Y]}{\sigma_X \sigma_Y} \in [-1, 1]$$

### 3.4 NumPy 로 통계량 계산

```python
import numpy as np

samples = np.random.normal(loc=5, scale=2, size=10000)

print(f"평균: {samples.mean():.4f}")           # 5 에 가까움
print(f"분산: {samples.var():.4f}")            # 4 에 가까움
print(f"표준편차: {samples.std():.4f}")        # 2 에 가까움

# 두 변수의 공분산·상관
x = np.random.normal(0, 1, 1000)
y = 2 * x + np.random.normal(0, 0.5, 1000)

print(f"공분산: {np.cov(x, y)[0, 1]:.4f}")     # 약 2 (= 2 * Var[x])
print(f"상관: {np.corrcoef(x, y)[0, 1]:.4f}")  # 0.97 정도 (강한 양의 상관)
```

### 3.5 챕터 정리

기댓값·분산·공분산은 *분포를 한두 숫자로 요약* 하는 도구입니다. ML 에서는 *데이터·모델 출력·그래디언트의 분포 특성* 을 측정하는 데 끊임없이 등장합니다. 다음 챕터에서는 *여러 변수의 결합 분포* 와 *조건부 확률* 을 봅니다.

---

## 4. 결합·주변·조건부 확률

### 4.1 결합 확률

두 확률 변수 X, Y 의 *동시 분포*: $P(X=x, Y=y)$.

예: *날씨* (X = 맑음/흐림) 와 *우산 들기* (Y = 예/아니오) 의 동시 분포.

### 4.2 주변 확률

결합 분포에서 *한 변수만* 의 분포로 환원:

$$P(X=x) = \sum_y P(X=x, Y=y)$$

### 4.3 조건부 확률

*Y 가 주어졌을 때 X 의 확률*:

$$P(X|Y) = \frac{P(X, Y)}{P(Y)}$$

ML 의 핵심 사고: *분류 모델* 은 *입력 X 가 주어졌을 때 클래스 Y 의 조건부 확률* $P(Y|X)$ 를 학습.

### 4.4 독립

X 와 Y 가 *독립* 이면:

$$P(X, Y) = P(X) P(Y)$$

조건부 확률은 *주변 확률과 같음*: $P(X|Y) = P(X)$.

ML 에서 *Naive Bayes* 가 모든 특성이 *조건부 독립* 이라 가정해 학습을 단순화합니다.

### 4.5 챕터 정리

여러 변수의 확률은 *결합·주변·조건부* 의 세 형태로 다뤄집니다. ML 모델은 본질적으로 *조건부 확률 분포* 를 학습하는 일이며, 다음 챕터의 *베이즈 정리* 가 이 사고의 핵심 도구입니다.

---

## 5. 베이즈 정리

### 5.1 베이즈 정리

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

이 식의 각 항:

- $P(A|B)$: *사후 확률 (posterior)* — B 를 관찰한 후의 A 의 확률
- $P(B|A)$: *우도 (likelihood)* — A 가 사실일 때 B 가 일어날 확률
- $P(A)$: *사전 확률 (prior)* — B 를 보기 전 A 에 대한 신념
- $P(B)$: *증거 (evidence)* — B 가 일어날 전체 확률 (정규화 상수)

직관: *사전 신념을 새 증거로 갱신* 하는 수학적 도구.

### 5.2 예시 — 의료 진단

질병 D 의 사전 확률 1%, 검사의 *민감도* (D 이면 양성) 95%, *특이도* (D 가 아니면 음성) 95%.

검사가 *양성* 일 때 실제 D 일 확률은?

$$P(D|+) = \frac{P(+|D) P(D)}{P(+)}$$

$$P(+) = P(+|D)P(D) + P(+|\neg D)P(\neg D)$$
$$= 0.95 \times 0.01 + 0.05 \times 0.99 = 0.059$$

$$P(D|+) = \frac{0.95 \times 0.01}{0.059} \approx 0.16$$

검사 양성이라도 *실제 질병일 확률은 16%* 에 불과합니다. 직관과 다른 이 결과가 베이즈 사고의 힘을 보여 줍니다.

### 5.3 ML 에서의 베이즈

*베이지안 신경망*·*VAE*·*디퓨전 모델*·*RAG 의 사후 추론* 등 베이즈 사고가 곳곳에 등장합니다. 더 자세한 응용은 Vol 28 (베이지안 ML) 에서 다룹니다.

### 5.4 챕터 정리

베이즈 정리는 *사전 신념을 새 증거로 갱신* 하는 식입니다. 직관과 다른 결론을 종종 만들어 내며, ML 의 많은 모델이 베이즈 사고를 기반으로 합니다.

---

## 6. MLE — 최대가능도 추정

### 6.1 MLE 의 정의

**최대가능도 추정 (Maximum Likelihood Estimation, MLE)** 은 *데이터를 가장 잘 설명하는 파라미터* 를 찾는 방법입니다.

데이터 $D = \{x_1, \ldots, x_n\}$ 가 파라미터 $\theta$ 의 분포에서 *독립적으로 샘플링* 되었다면:

$$L(\theta; D) = \prod_{i=1}^n p(x_i | \theta)$$

이 *가능도* 를 최대화하는 $\theta$ 를 찾습니다.

실무에서는 *로그 가능도* 를 사용해 곱을 합으로 바꿉니다 (수치 안정성·미분 편의):

$$\log L(\theta; D) = \sum_{i=1}^n \log p(x_i | \theta)$$

*로그를 씌워도 최댓값의 위치는 변하지 않습니다.*

### 6.2 정규 분포의 MLE

데이터가 $N(\mu, \sigma^2)$ 에서 왔다고 가정. 로그 가능도를 최대화하면:

- $\hat{\mu} = \frac{1}{n} \sum x_i$ (표본 평균)
- $\hat{\sigma}^2 = \frac{1}{n} \sum (x_i - \hat{\mu})^2$ (표본 분산)

NumPy:

```python
import numpy as np

data = np.random.normal(loc=3, scale=2, size=1000)

mle_mu = data.mean()
mle_var = data.var()
print(f"MLE μ: {mle_mu:.3f}, σ: {np.sqrt(mle_var):.3f}")
# MLE μ: ~3.0, σ: ~2.0
```

### 6.3 ML 에서의 MLE

*거의 모든 ML 모델 학습은 MLE 또는 그 변형* 입니다.

- *선형 회귀의 MSE 손실* — 데이터가 가우시안 노이즈를 가진다는 가정 하의 MLE
- *로지스틱 회귀의 BCE 손실* — 베르누이 분포 가정 하의 MLE
- *분류의 교차엔트로피 손실* — 범주 분포 가정 하의 MLE
- *언어 모델 학습* — 다음 토큰 분포의 MLE

이 사실이 다음 챕터에서 명시적으로 유도됩니다.

### 6.4 챕터 정리

MLE 는 *데이터를 가장 잘 설명하는 파라미터* 를 찾는 방법이며, 거의 모든 ML 학습의 본질입니다. 다음 챕터에서는 MLE 에 *사전 신념* 을 더한 MAP 를 봅니다.

---

## 7. MAP — 사전 신념을 더한 추정

### 7.1 MAP 의 정의

**최대 사후 확률 추정 (Maximum A Posteriori, MAP)** 은 *베이즈 사후 확률을 최대화* 하는 파라미터 추정입니다.

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D|\theta) P(\theta)$$

MLE 와의 차이는 *사전 분포 $P(\theta)$* 가 더해지는 점입니다.

### 7.2 MAP = MLE + 정칙화

가중치에 대한 *가우시안 사전* 을 가정하면:

$$P(W) \propto \exp\left(-\frac{\|W\|^2}{2\sigma^2}\right)$$

로그를 취하면:

$$\log P(W) = -\frac{\|W\|^2}{2\sigma^2} + \text{const}$$

이를 MAP 식에 대입:

$$\hat{W}_{\text{MAP}} = \arg\min_W \left[ -\log P(D|W) + \frac{\|W\|^2}{2\sigma^2} \right]$$

오른쪽 항이 정확히 *L2 정칙화* 입니다.

이 사실은 흥미롭습니다. *L2 정칙화 (가중치 감쇠) 는 가우시안 사전에서 유도되는 MAP 추정* 입니다. 마찬가지로 *L1 정칙화 (Lasso) 는 라플라스 사전에서 유도* 됩니다.

### 7.3 챕터 정리

MAP 은 MLE 에 *사전 분포* 를 더한 추정이며, ML 에서 가장 흔한 *정칙화* 가 본질적으로 *베이즈 사전의 일종* 임을 보여 줍니다.

---

## 8. 로그가능도와 손실 함수

### 8.1 손실 함수 = 음의 로그가능도

ML 학습은 *손실을 최소화* 하는 일이고, MLE 는 *로그가능도를 최대화* 하는 일입니다. 둘은 부호만 다릅니다.

$$\text{Loss}(\theta) = -\log L(\theta; D) = -\sum_i \log p(x_i | \theta)$$

따라서 *손실 함수의 형태* 는 *우리가 가정한 분포* 에서 결정됩니다.

### 8.2 회귀의 MSE = 가우시안 가정

데이터에 *가우시안 노이즈* 를 가정:

$$y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)$$

이때 가능도:

$$p(y_i | x_i; \theta) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - f(x_i))^2}{2\sigma^2}\right)$$

음의 로그가능도:

$$-\log L = \frac{1}{2\sigma^2} \sum (y_i - f(x_i))^2 + \text{const}$$

상수와 스케일을 무시하면 *MSE 손실* 이 됩니다.

### 8.3 이진 분류의 BCE = 베르누이 가정

데이터가 베르누이 분포에서 왔다고 가정:

$$p(y | x; \theta) = \hat{y}^y (1-\hat{y})^{1-y}$$

음의 로그가능도:

$$-\log L = -\sum [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

이것이 정확히 *이진 교차엔트로피 (BCE) 손실* 입니다.

### 8.4 다중 분류의 CCE = 범주 가정

데이터가 범주 분포에서 왔다고 가정:

$$p(y | x; \theta) = \prod_k \hat{p}_k^{[y=k]}$$

음의 로그가능도:

$$-\log L = -\sum_i \log \hat{p}_{y_i}$$

이것이 *범주 교차엔트로피 (CCE) 손실* 입니다.

### 8.5 통합적 시각

| 작업 | 분포 가정 | 손실 함수 |
|------|----------|----------|
| 회귀 | 가우시안 | MSE |
| 이진 분류 | 베르누이 | BCE |
| 다중 분류 | 범주 | CCE |
| 카운트 예측 | 푸아송 | Poisson NLL |
| 시퀀스 생성 | 자기회귀 범주 | Token-level CE |

*손실 함수는 임의가 아니라 분포 가정에서 유도됩니다*. 이 사실을 알면 새 작업에 *어떤 손실 함수를 써야 하는가* 를 *분포 가정* 만으로 도출할 수 있습니다.

### 8.6 챕터 정리

ML 의 손실 함수는 *우리가 가정한 분포의 음의 로그가능도* 입니다. MSE·BCE·CCE 는 모두 다른 분포 가정의 결과이며, 이 사실은 ML 손실 함수의 *임의성을 제거* 합니다.

---

## 9. NumPy/Scipy 로 분포 다루기

### 9.1 종합 실습

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 정규 분포의 PDF·CDF·샘플링
mu, sigma = 0, 1
dist = stats.norm(mu, sigma)

x = np.linspace(-4, 4, 100)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, dist.pdf(x))
axes[0].set_title('PDF')

axes[1].plot(x, dist.cdf(x))
axes[1].set_title('CDF')

samples = dist.rvs(10000)
axes[2].hist(samples, bins=50, density=True)
axes[2].plot(x, dist.pdf(x), 'r-')
axes[2].set_title('샘플 + 이론 PDF')

plt.tight_layout()
plt.show()
```

### 9.2 베이즈 갱신 시뮬레이션

```python
# 동전이 앞면 나올 확률 p 의 사후 분포 갱신
import numpy as np
import matplotlib.pyplot as plt

p_true = 0.7
ps = np.linspace(0, 1, 100)

# 사전: 균등 (Beta(1, 1))
prior = np.ones_like(ps)
prior /= prior.sum()

posteriors = [prior]
for n_obs in [1, 5, 20, 100]:
    np.random.seed(0)
    obs = np.random.binomial(1, p_true, n_obs)
    n_heads = obs.sum()
    
    # 우도 = p^heads * (1-p)^tails
    likelihood = ps**n_heads * (1-ps)**(n_obs - n_heads)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    posteriors.append(posterior)

fig, ax = plt.subplots(figsize=(10, 6))
labels = ['prior', '1 obs', '5 obs', '20 obs', '100 obs']
for post, label in zip(posteriors, labels):
    ax.plot(ps, post, label=label)
ax.axvline(p_true, color='gray', linestyle='--', label=f'true p={p_true}')
ax.legend(); ax.set_xlabel('p'); ax.set_ylabel('posterior')
ax.set_title('베이즈 사후 분포의 갱신')
plt.show()
```

관찰이 늘수록 사후 분포가 *진짜 값에 점점 집중* 됩니다.

### 9.3 MLE 직접 구현

```python
import numpy as np

# 데이터: 정규 분포에서 샘플
np.random.seed(0)
data = np.random.normal(loc=3, scale=2, size=500)

# 닫힌 형태 MLE
mu_mle = data.mean()
sigma_mle = data.std()
print(f"MLE μ={mu_mle:.4f}, σ={sigma_mle:.4f}")

# 그래디언트 디센트로 MLE
from scipy.stats import norm

mu = 0.0
sigma = 1.0
lr = 0.01

for step in range(1000):
    # 음의 로그가능도와 그 미분
    # NLL = (1/2σ²) Σ(x-μ)² + n log σ + const
    # d/dμ NLL = -(1/σ²) Σ(x-μ)
    # d/dσ NLL = -1/σ³ Σ(x-μ)² + n/σ
    n = len(data)
    diff = data - mu
    dmu = -(1 / sigma**2) * diff.sum()
    dsigma = -(1 / sigma**3) * (diff**2).sum() + n / sigma
    mu -= lr * dmu / n
    sigma -= lr * dsigma / n

print(f"GD μ={mu:.4f}, σ={sigma:.4f}")
```

두 방법 모두 같은 답에 도달합니다.

### 9.4 챕터 정리

NumPy 와 Scipy 로 *모든 확률·통계 계산* 을 직접 할 수 있습니다. 베이즈 갱신·MLE·분포 시각화는 코드 수십 줄이면 충분하며, 직접 돌려 본 경험이 *공식 암기* 보다 훨씬 깊은 이해를 만듭니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **확률 변수와 분포** — 불확실성을 숫자와 함수로 표현하는 도구.
- **주요 분포** — 베르누이·이항·정규·범주·디리클레가 ML 에서 가장 자주 등장.
- **기댓값·분산·공분산** — 분포를 한두 숫자로 요약.
- **결합·주변·조건부 확률** — 여러 변수의 관계를 다루는 세 형태.
- **베이즈 정리** — 사전 신념을 새 증거로 갱신.
- **MLE** — 데이터를 가장 잘 설명하는 파라미터를 찾는 방법.
- **MAP** — MLE 에 사전 신념을 더한 추정. *정칙화는 베이즈 사전의 일종*.
- **손실 함수의 유도** — MSE·BCE·CCE 는 모두 *다른 분포 가정의 음의 로그가능도*.

가장 기억에 남겨야 할 한 줄은 **"손실 함수는 임의가 아니라 분포 가정에서 유도되며, 학습은 그 가정 하의 MLE 다."** 입니다.

다음 권은 [Volume 8 — 정보 이론](./volume_12_information_theory.md) 입니다. 거기서는 *교차엔트로피와 KL 발산이 왜 그렇게 자주 등장하는가* 를 정보 이론으로 답합니다.

---

## 자가점검 키워드

`확률 변수`, `확률 분포`, `베이즈 정리`, `MLE`, `MAP`, `기댓값`, `로그가능도`, `손실 함수`

## 자가점검 질문

1. 이산과 연속 확률 변수의 차이를 한 문단으로 설명하십시오.
2. 베르누이·이항·정규·범주 분포의 PDF/PMF 와 ML 응용을 표로 정리하십시오.
3. 베이즈 정리의 *의료 진단* 예시 (질병 사전 1%, 민감도 95%, 특이도 95%) 에서 양성 결과 시 실제 질병 확률을 손으로 계산하십시오.
4. MLE 와 MAP 의 차이를 한 문단으로 설명하십시오.
5. *L2 정칙화 = 가우시안 사전 하의 MAP* 임을 유도 흐름으로 설명하십시오.
6. 회귀의 MSE 손실이 *가우시안 노이즈 가정의 MLE* 임을 유도하십시오.
7. 새 작업에 어떤 손실 함수를 쓸지 결정하는 *3 단계 사고 절차* 를 적으십시오.

## 다음 권

[Volume 8 — 정보 이론](./volume_12_information_theory.md)
