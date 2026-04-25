# Volume 42 — 합성곱 신경망(CNN)

> 이 권이 끝나면 *왜 이미지에는 MLP 가 아니라 CNN 을 쓰는가* 에 답할 수 있게 됩니다.

## 목적

합성곱 신경망 (CNN) 은 *이미지 데이터의 공간적 구조* 를 활용하는 신경망입니다. *지역성·평행 이동 불변성* 같은 공간 구조의 사전 가정이 모델에 내장되어, 같은 정확도를 *훨씬 적은 파라미터* 로 달성합니다.

## 선수 지식

- Volume 32, 36 완료

## 학습 결과

1. 합성곱 연산을 NumPy 로 구현할 수 있습니다.
2. 풀링·스트라이드·패딩의 역할을 이해합니다.
3. 수용 영역 (Receptive Field) 의 의미를 알 수 있습니다.
4. CNN 이 MLP 보다 이미지에 적합한 이유를 설명합니다.
5. PyTorch 로 LeNet 급 CNN 을 학습시킬 수 있습니다.

---

## 1. 합성곱 연산

### 1.1 정의

작은 *커널 (필터)* 을 입력 위에서 *슬라이드* 하며 *지역 가중합* 계산:

```
입력  3 × 3       커널 2 × 2       출력 2 × 2
1 2 3            1 0              5 6
4 5 6      *      0 1       =     8 9
7 8 9
```

### 1.2 수식

$$Y[i, j] = \sum_{m, n} K[m, n] \cdot X[i+m, j+n]$$

### 1.3 NumPy 구현

```python
import numpy as np

def conv2d(X, K):
    H, W = X.shape
    kH, kW = K.shape
    Y = np.zeros((H - kH + 1, W - kW + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+kH, j:j+kW] * K).sum()
    return Y

X = np.arange(9).reshape(3, 3).astype(float)
K = np.array([[1, 0], [0, 1]])
print(conv2d(X, K))
```

---

## 2. 합성곱의 핵심 사고

### 2.1 지역성 (Locality)

이미지의 *가까운 픽셀* 은 의미적으로 관련. 합성곱은 *지역 패치* 만 처리.

### 2.2 평행 이동 불변성

같은 패턴이 이미지의 *어느 위치에 있어도* 같은 필터로 감지. *고양이 얼굴이 좌상단에 있어도 우하단에 있어도* 같은 필터.

### 2.3 가중치 공유

한 필터가 *모든 위치* 에 같은 가중치로 적용. *MLP 보다 훨씬 적은 파라미터* 로 같은 표현력.

---

## 3. 풀링·스트라이드·패딩

### 3.1 풀링

지역 영역의 *최댓값 (Max) 또는 평균 (Avg)* 만 남김. *공간 차원 축소 + 작은 변화에 강건*.

### 3.2 스트라이드

커널을 *N 칸씩 이동*. 큰 스트라이드는 출력 공간을 빠르게 줄임.

### 3.3 패딩

입력 가장자리에 *0 추가*. 출력 크기를 입력과 같게 유지.

---

## 4. 수용 영역

층이 깊어질수록 *한 출력 픽셀이 보는 입력 영역* 이 커집니다. 이를 *수용 영역 (Receptive Field)* 이라 합니다.

3×3 커널 두 층은 *5×5 영역* 의 정보를 봅니다. 깊이가 곧 *전체적 시각* 을 가능하게 만듭니다.

---

## 5. 미니 CNN — PyTorch

```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

MNIST 학습 시 99% 정확도 가능 (1989 LeCun 의 LeNet 의 PyTorch 버전).

---

## 권 정리

- 합성곱 = 지역 가중합
- 핵심 사고 = 지역성 + 평행 이동 불변성 + 가중치 공유
- 풀링·스트라이드·패딩 = 공간 제어
- 수용 영역 = 깊이 = 전체 시각
- LeNet 5 줄이 CNN 의 표준 골격

가장 기억할 한 줄: **"CNN 은 이미지의 공간 구조를 모델에 내장해, 같은 정확도를 훨씬 적은 파라미터로 달성한다."**

다음 권: [Volume 43 — CNN 모델 계보](./volume_43_cnn_lineage.md)

---

## 자가점검 키워드

`합성곱`, `커널/필터`, `지역성`, `평행 이동 불변성`, `가중치 공유`, `풀링`, `스트라이드`, `패딩`, `수용 영역`

## 자가점검 질문

1. 합성곱 연산을 NumPy 로 5 줄 안에 구현하십시오.
2. CNN 의 3 가지 핵심 사고를 적으십시오.
3. 수용 영역의 의미를 설명하십시오.
4. CNN 이 MLP 보다 이미지에 적합한 이유 3 가지를 적으십시오.

## 다음 권

[Volume 43 — CNN 모델 계보](./volume_43_cnn_lineage.md)
