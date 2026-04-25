# Volume 36 — PyTorch 실전

> 이 권이 끝나면 빈 노트북에서 시작해 *데이터 로드 → 모델 정의 → 학습 → 평가 → 체크포인트 저장* 까지를 30 분 안에 작성할 수 있게 됩니다.

## 목적

PyTorch 의 모든 기능을 깊이 알 필요는 없지만, *표준 학습 루프 골격* 은 손에 익혀야 합니다. 이 권은 PyTorch 의 7 가지 핵심 빌딩블록 (텐서·`nn.Module`·DataLoader·옵티마이저·손실·체크포인트·디바이스 관리) 을 통합한 *완전한 학습 코드* 를 다집니다.

## 선수 지식

- Volume 3, 32, 33 완료

## 학습 결과

1. PyTorch 의 7 가지 핵심 빌딩블록을 모두 다룰 수 있습니다.
2. 표준 학습 루프를 30 줄 안에 작성할 수 있습니다.
3. 체크포인트 저장·복원을 구현할 수 있습니다.
4. CPU/GPU/MPS 디바이스 추상화를 적용할 수 있습니다.
5. Mixed Precision 학습을 활성화할 수 있습니다.

---

## 1. 7 가지 핵심 빌딩블록

```
1. Tensor          - 데이터·가중치·그래디언트
2. nn.Module       - 모델 정의
3. Dataset/DataLoader - 데이터 공급
4. Optimizer       - 그래디언트 → 가중치 갱신
5. Loss Function   - 학습 신호
6. Checkpoint      - 모델 상태 저장·복원
7. Device          - CPU/GPU/MPS
```

각각이 *학습 루프 한 줄* 에 대응합니다.

---

## 2. 완전한 학습 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 데이터
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# 2. 모델
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)

# 3. 옵티마이저·손실
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 학습 루프
for epoch in range(10):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# 5. 체크포인트 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')
```

이 30 줄이 *PyTorch 학습 코드의 표준 골격* 입니다.

---

## 3. 체크포인트 복원

```python
ckpt = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

학습 중단 후 *이어서 학습* 가능.

---

## 4. Mixed Precision

큰 모델 학습 시 *FP16 사용* 으로 메모리·속도 개선:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch_x, batch_y in loader:
    optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

H100·A100 에서 약 2-3 배 가속.

---

## 5. 표준 사용 패턴

### 5.1 학습/평가 모드

```python
model.train()    # 학습 모드 (Dropout, BN 활성)
model.eval()     # 평가 모드 (Dropout 비활성, BN 이동평균)

with torch.no_grad():    # 그래디언트 추적 비활성 (메모리 절감)
    pred = model(x)
```

### 5.2 GPU 메모리 관리

```python
torch.cuda.empty_cache()           # 캐시 해제
print(torch.cuda.memory_allocated() / 1e9, 'GB')   # 사용량
```

### 5.3 데이터 로딩 최적화

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,           # 병렬 로딩
    pin_memory=True,         # GPU 전송 가속
    persistent_workers=True, # 워커 재사용
)
```

---

## 권 정리

- 7 빌딩블록 = Tensor·Module·DataLoader·Optimizer·Loss·Checkpoint·Device
- 표준 학습 루프 = 30 줄
- Mixed Precision = 2-3 배 가속
- DataLoader 최적화 = num_workers + pin_memory

가장 기억할 한 줄: **"PyTorch 학습 코드의 표준 골격 30 줄을 외우면, 모든 새 모델 학습이 그 변형이다."**

다음 권: [Volume 37 — 데이터 파이프라인과 실험 관리](./volume_37_data_pipeline.md)

---

## 자가점검 키워드

`Tensor`, `nn.Module`, `DataLoader`, `state_dict`, `autocast`, `train/eval`, `num_workers`

## 자가점검 질문

1. PyTorch 7 빌딩블록을 적으십시오.
2. 표준 학습 루프를 30 줄 안에 작성하십시오.
3. 체크포인트 저장·복원 코드를 적으십시오.
4. `model.train()` 과 `model.eval()` 의 차이를 설명하십시오.
5. DataLoader 최적화 4 옵션을 적으십시오.

## 다음 권

[Volume 37 — 데이터 파이프라인과 실험 관리](./volume_37_data_pipeline.md)
