# Volume 40 — 학습률 스케줄 깊이

> 이 권이 끝나면 *학습률 스케줄을 바꾸는 것만으로 같은 모델의 성능을 수 % 끌어올릴 수 있다* 는 사실을 코드로 증명할 수 있게 됩니다.

## 목적

옵티마이저 자체보다 *학습률을 시간에 따라 어떻게 바꿀지* 가 결과에 더 큰 영향을 줄 수 있습니다. Warmup·Cosine·OneCycle·Polynomial·ReduceOnPlateau 같은 스케줄은 각자 다른 학습 동학에 적합합니다.

## 선수 지식

- Volume 33 완료

## 학습 결과

1. Warmup 의 동기를 그래디언트 분산 관점에서 설명할 수 있습니다.
2. Cosine·Polynomial·Linear Decay 의 차이를 그래프로 그릴 수 있습니다.
3. OneCycle 정책의 발상을 이해합니다.
4. PyTorch `lr_scheduler` 를 적용할 수 있습니다.
5. LR Range Test 로 적정 학습률을 빠르게 찾습니다.

---

## 1. Step Decay·Exponential

가장 단순. 일정 에폭마다 학습률을 *배수로 감소*:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 30 에폭마다 lr × 0.1
```

---

## 2. Cosine Annealing

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

학습률이 *코사인 함수처럼* 부드럽게 감소. 트랜스포머 학습의 표준.

### 2.1 Cosine Restarts

여러 *주기적 재시작* 으로 *지역 최소를 빠져나가는* 효과:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

---

## 3. Linear Warmup + Cosine Decay

트랜스포머 학습의 *현재 표준*:

```python
def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

LLM 사전학습 거의 모두 이 패턴.

---

## 4. OneCycle

학습률을 *작게 → 크게 → 작게* 한 사이클로 변동. fast.ai 인기.

```python
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=1000)
```

---

## 5. ReduceOnPlateau

검증 손실이 *N 에폭 동안 개선 없으면* 학습률 감소:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

for epoch in range(100):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)
```

---

## 6. LR Range Test

학습률을 *지수적으로 증가* 시키며 손실 관찰. *손실이 가장 빠르게 감소* 하는 구간이 적정.

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

곡선의 *최저 손실 직전 학습률* 이 좋은 시작점.

---

## 7. 권장 조합

```
+--------------------+--------------------------+
| 모델               | 권장 스케줄             |
+--------------------+--------------------------+
| 작은 CNN/MLP       | StepLR or ReduceOnPlateau|
| ResNet 류          | Cosine 또는 OneCycle    |
| Transformer 사전학습| Linear Warmup + Cosine  |
| 미세조정           | Cosine + Warmup (짧게)  |
+--------------------+--------------------------+
```

---

## 권 정리

- Step·Exponential = 단순
- Cosine = 트랜스포머 표준
- Linear Warmup + Cosine = LLM 사전학습 표준
- OneCycle = fast.ai 인기
- ReduceOnPlateau = 적응적
- LR Range Test = 빠른 학습률 탐색

가장 기억할 한 줄: **"트랜스포머 학습은 Linear Warmup + Cosine Decay 부터 시작 — 거의 항상 잘 동작한다."**

다음 권: [Volume 41 — 딥러닝 디버깅](./volume_41_dl_debugging.md)

---

## 자가점검 키워드

`Step Decay`, `Cosine`, `Warmup`, `OneCycle`, `ReduceOnPlateau`, `LR Range Test`

## 자가점검 질문

1. Warmup 이 큰 모델 학습에 필요한 이유를 설명하십시오.
2. Cosine Annealing 곡선을 그리십시오.
3. ReduceOnPlateau 의 적용 시점을 적으십시오.
4. LR Range Test 의 동작 원리를 설명하십시오.

## 다음 권

[Volume 41 — 딥러닝 디버깅](./volume_41_dl_debugging.md)
