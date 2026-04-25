# Volume 37 — 데이터 파이프라인과 실험 관리

> 이 권이 끝나면 *재현 가능한 실험* 을 설계할 수 있게 됩니다.

## 목적

ML 실험은 *코드 + 데이터 + 하이퍼파라미터 + 시드* 의 조합으로 정의됩니다. 어느 한 가지라도 추적되지 않으면 *결과를 재현할 수 없는* 시스템이 됩니다. 이 권은 PyTorch DataLoader 의 깊이와 실험 추적 도구 (W&B·MLflow) 통합을 다집니다.

## 선수 지식

- Volume 3, 36 완료

## 학습 결과

1. PyTorch Dataset·DataLoader 를 직접 만들 수 있습니다.
2. 데이터 변환 파이프라인 (`transforms`) 을 적용할 수 있습니다.
3. W&B / MLflow 로 실험을 추적합니다.
4. *코드·데이터·하이퍼파라미터·시드* 를 모두 버저닝할 수 있습니다.

---

## 1. 커스텀 Dataset

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

`__len__` 과 `__getitem__` 만 구현하면 어떤 데이터든 PyTorch 와 통합.

---

## 2. 변환 파이프라인

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])
```

학습 시 매번 *다른 변환* 적용 → 데이터 증강 효과.

---

## 3. W&B 통합

```python
import wandb

wandb.init(project='my-experiment', config={
    'lr': 1e-3,
    'batch_size': 32,
    'model': 'resnet50',
})

for epoch in range(10):
    train_loss = train(...)
    val_loss = validate(...)
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch,
    })

# 모델 가중치 아티팩트로 저장
wandb.save('model.pt')
```

---

## 4. 재현성 확보

```python
import torch, numpy as np, random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

이 함수를 *학습 시작 시 호출* 해야 결과 재현 가능.

추가 권장:
- *Git commit hash* 를 W&B config 에 기록
- *데이터 버전* (DVC commit) 기록
- *환경 의존성* `requirements-lock.txt` 함께 보존

---

## 5. 좋은 실험 추적 습관

- *모든 실험에 의미 있는 이름* — `lr_1e3_bs64_warmup100`
- *가설을 실험 시작 전에 적기*
- *결과 요약을 매 실험 후 한 줄*
- *주간 단위로 결과 팀 공유*
- *오래된 실험 정기 정리*

---

## 권 정리

- Dataset = `__len__` + `__getitem__`
- transforms = 학습 시 데이터 증강
- W&B / MLflow = 실험 추적 표준
- 시드 + git hash + 데이터 버전 = 재현성

가장 기억할 한 줄: **"재현 불가능한 실험은 결과 없는 실험이다."**

다음 권: [Volume 38 — 모델 평가와 메트릭](./volume_38_evaluation_metrics.md)

---

## 자가점검 키워드

`Dataset`, `DataLoader`, `transforms`, `W&B`, `시드 고정`, `재현성`

## 자가점검 질문

1. 커스텀 Dataset 의 두 필수 메서드를 적으십시오.
2. 데이터 증강이 일반화에 도움 되는 이유를 설명하십시오.
3. PyTorch 의 시드 고정 코드를 적으십시오.
4. 재현성을 위한 4 가지 추적 항목을 나열하십시오.

## 다음 권

[Volume 38 — 모델 평가와 메트릭](./volume_38_evaluation_metrics.md)
