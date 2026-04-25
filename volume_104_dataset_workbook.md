# Volume 104 — 데이터셋 종합 실습 워크북

> 이 권이 끝나면 *MNIST·CelebA·Tiny ImageNet·SST-2·SQuAD* 같은 표준 데이터셋을 *처음부터 끝까지* 다뤄 본 경험을 갖게 됩니다.

## 목적

이 워크북의 마지막 권은 *통합 실습* 입니다. 지금까지 배운 모든 도구를 *5 가지 표준 데이터셋* 에 적용하면서 *데이터 로드·전처리·학습·평가·시각화* 의 전 과정을 손으로 익힙니다.

## 선수 지식

- Volume 1-103 모두 완료
- PyTorch + Hugging Face 환경 구축

## 학습 결과

1. 5 가지 표준 데이터셋을 다룰 수 있습니다.
2. 각 데이터셋의 *표준 모델 골격* 을 적용할 수 있습니다.
3. 데이터셋별 *흔한 함정과 해결* 을 안다.
4. *처음부터 끝까지 한 사이클* 완성 경험을 갖습니다.

---

## 1. MNIST — 손글씨 숫자 분류

### 1.1 작업

28×28 회색조 손글씨 이미지 → 0-9 분류.

### 1.2 코드

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 모델 (LeNet 스타일)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 학습
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()
```

5 epoch 만 학습해도 *99% 정확도* 가능. CNN 입문의 표준.

---

## 2. CelebA — 얼굴 속성

### 2.1 작업

200K 얼굴 이미지에 *40 가지 속성 라벨* (성별·머리색·웃음 등) 다중 라벨 분류.

### 2.2 핵심

- *멀티 라벨* — 한 이미지에 여러 속성 동시
- *Sigmoid + BCE* (CCE 가 아님)
- *데이터 증강 중요* (Crop, Flip, ColorJitter)

```python
loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
```

### 2.3 응용

- 얼굴 생성 (StyleGAN) 학습
- 속성 편집

---

## 3. Tiny ImageNet — 일반 분류

### 3.1 작업

200 클래스, 클래스당 500 학습 이미지. ImageNet 의 *축소판*. 64×64 해상도.

### 3.2 표준 모델

ResNet-18 fine-tuning:

```python
from torchvision.models import resnet18

model = resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 200)   # 200 클래스
```

이 패턴이 *전이 학습 (Transfer Learning) 의 표준*.

---

## 4. SST-2 — 영화 리뷰 감정

### 4.1 작업

영화 리뷰 → 긍정·부정 이진 분류. 약 67K 학습 샘플.

### 4.2 코드

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ds = load_dataset('sst2')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(b): return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=128)
ds_tok = ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./out', num_train_epochs=3, per_device_train_batch_size=16),
    train_dataset=ds_tok['train'],
    eval_dataset=ds_tok['validation'],
)
trainer.train()
```

3 epoch 학습 → 약 91% 정확도.

---

## 5. SQuAD — 질의응답

### 5.1 작업

문맥 + 질문 → 답변 (문맥에서 추출).

### 5.2 표준 모델

BERT 기반 QA:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

qa = pipeline('question-answering', model='deepset/roberta-base-squad2')
result = qa(
    question="What is AI?",
    context="AI is the simulation of human intelligence by computers."
)
print(result['answer'])
```

직접 학습:

```python
ds = load_dataset('squad')
# 토큰화 시 question + context 결합, 답변 시작·끝 토큰 위치 라벨
# AutoModelForQuestionAnswering 학습
```

---

## 6. 워크북 활용

각 데이터셋에 대해 다음 *4 단계 사이클* 완수:

```
1. 베이스라인 — 가장 단순한 모델로 학습
2. 분석 — 오류 사례 살펴보기
3. 개선 — 데이터 증강·정칙화·모델 변경
4. 평가 — 메트릭 비교, 결론
```

이 사이클을 *5 데이터셋 모두 거치면* AI 엔지니어로서 *기본 실력 완성*.

---

## 7. 다음 단계

이 워크북을 마쳤다면, 자기 *실제 도메인 데이터* 로 같은 사이클 적용. 자기 회사의 데이터·자기 사이드 프로젝트의 데이터.

*책으로 배우는 시간 < 자기 데이터로 만들기 시간* 이어야 진짜 학습.

---

## 권 정리

- MNIST = CNN 입문
- CelebA = 멀티 라벨 + 얼굴
- Tiny ImageNet = 전이 학습
- SST-2 = 텍스트 분류
- SQuAD = 질의응답

각 데이터셋에서 *4 단계 사이클* (베이스라인·분석·개선·평가) 완수.

가장 기억할 한 줄: **"5 가지 데이터셋을 처음부터 끝까지 다뤄 보는 경험이 AI 엔지니어 기본 실력의 완성이며, 그다음은 자기 도메인 데이터로 옮겨간다."**

---

## 마치며

이 워크북은 104 권의 긴 여정이었습니다. AI 엔지니어로서 갖춰야 할 *직무 지식·수학·ML·DL·CV·시퀀스·임베딩·LLM·에이전트·생성 모델·GPU·운영·미세조정·데이터·NLP* 의 전 영역을 다뤘습니다.

이 책을 다 읽었더라도 *진짜 학습은 자기 손으로 만들 때 시작* 됩니다. 작은 RAG 챗봇·자기 도메인 분류기·간단한 에이전트 — 무엇이든 *직접 만들고 운영* 해 보시기 바랍니다.

좋은 AI 엔지니어가 되는 길에 이 책이 도움이 되었기를 바랍니다.

---

## 자가점검 키워드

`MNIST`, `CelebA`, `Tiny ImageNet`, `SST-2`, `SQuAD`, `4 단계 사이클`

## 자가점검 질문

1. 5 가지 데이터셋의 작업 종류를 적으십시오.
2. 각 데이터셋의 *표준 모델* 을 적으십시오.
3. 4 단계 사이클을 자기 도메인 데이터에 적용해 보십시오.

---

## 워크북의 끝
