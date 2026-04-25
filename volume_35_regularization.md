# Volume 35 — 정칙화 (Regularization)

> 이 권이 끝나면 *학습은 잘되는데 검증 성능이 나쁘다* 는 상황에서 무엇을 시도할지 표가 머릿속에 그려지게 됩니다.

## 목적

정칙화는 *과적합을 막아 일반화를 개선* 하는 모든 기법의 총칭입니다. L1/L2 정칙화·Dropout·Early Stopping·Data Augmentation·Label Smoothing 같은 다양한 도구가 있으며, 각자 다른 메커니즘으로 같은 목표 (일반화) 를 달성합니다.

## 선수 지식

- Volume 21, 32 완료

## 학습 결과

1. 5 가지 정칙화 기법의 차이를 알 수 있습니다.
2. 각 기법의 적용 시점과 함정을 안다.
3. *과적합 진단 → 정칙화 선택* 의 의사결정 흐름을 가집니다.

---

## 1. L1/L2 정칙화

가중치 노름 페널티를 손실에 추가:

$$L_{\text{total}} = L + \lambda \|W\|^2$$

L1 은 *희소성*, L2 는 *작은 가중치*. AdamW 는 L2 의 분리된 형태.

---

## 2. Dropout

학습 시 각 뉴런을 확률 $p$ 로 *비활성*:

```python
import torch.nn as nn
dropout = nn.Dropout(p=0.5)
```

장점: *암묵적 앙상블 효과*. *Bayesian 근사* 의 한 형태.
단점: *학습 시간 증가*. *추론 시는 비활성*.

---

## 3. Early Stopping

검증 손실이 *N 에폭 동안 개선 없으면 학습 중단*:

```python
patience = 5
best_loss = float('inf')
counter = 0
for epoch in range(100):
    val_loss = validate()
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint()
        counter = 0
    else:
        counter += 1
        if counter >= patience: break
```

가장 *비용 효율적* 인 정칙화. *추가 계산 없음*.

---

## 4. Data Augmentation

원본 데이터에 *변형 적용* 해 *데이터 양 증가* 효과:

- 비전: Crop, Flip, Rotate, ColorJitter, MixUp, CutMix
- NLP: Back-Translation, Synonym Replacement, EDA
- 오디오: Time Shift, Pitch Shift, SpecAugment

```python
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
])
```

---

## 5. Label Smoothing

정답 라벨을 약간 *부드럽게*:

원래: $y = (0, 0, 1, 0, 0)$
스무딩: $y' = (0.025, 0.025, 0.9, 0.025, 0.025)$ ($\alpha = 0.1$)

모델의 *과확신* 방지. 분류·번역 모델의 표준.

---

## 6. 추가 기법

- **Weight Decay** — L2 정칙화의 옵티마이저 통합 형태
- **Mixup / CutMix** — 두 샘플을 섞어 학습
- **Stochastic Depth** — 잔차 연결의 일부 층 무작위 건너뛰기
- **DropPath** — 비전 트랜스포머에서 사용

---

## 7. 의사결정 흐름

과적합 신호 감지 → 다음 순서로 시도:

1. *데이터 증강* (가장 비용 효율적)
2. *Early Stopping* (추가 비용 0)
3. *Weight Decay* 강도 증가
4. *Dropout* 추가
5. *모델 사이즈 줄이기*
6. *더 많은 학습 데이터* (가장 강력)

---

## 권 정리

- 정칙화 = 일반화 개선 도구의 총칭
- 5 표준 기법: L1/L2·Dropout·Early Stopping·Augmentation·Label Smoothing
- 의사결정: Augmentation → Early Stopping → Weight Decay → Dropout → 모델 축소 → 데이터 증가

가장 기억할 한 줄: **"가장 강력한 정칙화는 더 많은 데이터이며, 그게 불가능할 때 다른 기법들이 대안이 된다."**

다음 권: [Volume 36 — PyTorch 실전](./volume_36_pytorch_practice.md)

---

## 자가점검 키워드

`L1/L2`, `Dropout`, `Early Stopping`, `Augmentation`, `Label Smoothing`, `MixUp`

## 자가점검 질문

1. 5 가지 정칙화 기법의 차이를 적으십시오.
2. Dropout 이 *암묵적 앙상블* 효과를 만드는 메커니즘을 설명하십시오.
3. 과적합 감지 시 시도할 6 단계 순서를 나열하십시오.
4. Label Smoothing 이 일반화에 도움 되는 이유를 적으십시오.

## 다음 권

[Volume 36 — PyTorch 실전](./volume_36_pytorch_practice.md)
