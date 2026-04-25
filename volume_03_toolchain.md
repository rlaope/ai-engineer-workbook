# Volume 3 — 도구 체인과 작업 환경

> 이 권이 끝나면 AI 엔지니어가 매일 사용하는 도구의 전체 지도를 머릿속에 그릴 수 있게 됩니다.

## 목적

새로운 분야를 시작할 때 가장 큰 진입장벽은 *어떤 도구를 어떤 순서로 익혀야 하는가* 입니다. 이 권은 Python 생태계·딥러닝 프레임워크·하드웨어 가속 라이브러리·실험 추적 도구·모델 허브를 한 장의 지도 위에 펼쳐 보여 줍니다. 각 도구가 *언제 등장하고 어떤 문제를 푸는지* 를 미리 파악해 두면, 이후의 권에서 새로운 도구가 등장할 때 위치를 잡기 쉽습니다.

## 선수 지식

- Volume 1, 2 완료
- 외부 지식: 가상환경·패키지 관리·터미널 기본 사용

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 자신의 노트북에 PyTorch + CUDA 환경을 직접 구축할 수 있습니다.
2. Hugging Face Hub 에서 모델을 검색하고 로컬에 내려받을 수 있습니다.
3. Jupyter, VS Code, `python -i` 의 사용 시점을 구분할 수 있습니다.
4. 실험 추적 도구(W&B, MLflow)의 역할을 한 문장으로 설명할 수 있습니다.
5. CUDA, cuDNN, NCCL, Triton 의 관계를 도식화할 수 있습니다.

---

## 이 권을 읽기 전에

도구는 *수단* 입니다. 도구 자체를 익히는 것보다 *각 도구가 푸는 문제* 를 이해하는 것이 더 중요합니다. 도구는 빠르게 변하지만, 도구가 푸는 *문제의 구조* 는 거의 변하지 않습니다.

이 권은 도구의 *카탈로그* 가 아닙니다. 카탈로그는 인터넷에 무수히 많습니다. 이 권의 목표는 *각 도구가 어떤 문제 영역에 속하는지를 분류해 둠* 으로써, 나중에 새 도구를 만났을 때 *기존 어느 영역의 변형인지* 를 즉시 식별할 수 있게 만드는 것입니다.

코드와 설치 명령이 본문에 등장합니다. 이 권의 코드는 *직접 자기 환경에서 실행* 하는 것이 학습 효과를 가장 크게 만듭니다. 단순히 읽고 넘어가지 마시고, 한 줄씩 따라 치면서 *자신의 시스템에 환경을 구축* 하시기 바랍니다. 이 권을 마치는 시점에 자기 노트북에 PyTorch + Hugging Face 환경이 구축되어 있어야 다음 권부터의 학습이 매끄럽게 이어집니다.

---

## 1. Python 생태계의 지도

### 1.1 왜 Python 인가

AI 엔지니어링에서 Python 은 사실상 표준 언어입니다. 이는 *언어 자체가 가장 우수해서* 가 아니라 *생태계가 가장 두텁기 때문* 입니다.

다음 사실들이 Python 의 위치를 만듭니다.

- 가장 인기 있는 딥러닝 프레임워크(PyTorch, TensorFlow, JAX) 모두 *Python API 가 1차 인터페이스* 입니다.
- Hugging Face·LangChain·LlamaIndex 같은 모든 주요 AI 라이브러리가 Python 으로 작성되었습니다.
- 데이터 과학 도구(NumPy, pandas, matplotlib, scikit-learn) 가 Python 위에 자리잡았습니다.
- 학계의 거의 모든 논문 코드가 Python 으로 공개됩니다.

Python 은 *느린 언어* 라는 평가가 있지만, AI 엔지니어링에서는 *연산이 GPU 상의 컴파일된 커널* 에서 일어나므로 Python 의 느림이 전체 성능에 거의 영향을 주지 않습니다. Python 은 *지휘자* 의 역할을 하고, 무거운 일은 C++/CUDA 가 합니다.

### 1.2 데이터 과학 핵심 라이브러리

다음 다섯 라이브러리는 거의 모든 AI 작업에서 등장합니다.

**NumPy** — *N차원 배열 연산*. 모든 ML 라이브러리의 기반.

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a @ b)  # 행렬 곱
# [[19 22]
#  [43 50]]
```

**pandas** — *표 형식 데이터 처리*. CSV·Excel·SQL 결과를 DataFrame 으로 다룸.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())  # 기초 통계
df_filtered = df[df['score'] > 80]
```

**matplotlib / seaborn** — *시각화*. 그래프·차트·산점도.

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('x'); plt.ylabel('y')
plt.show()
```

**scikit-learn** — *고전 ML 알고리즘*. 선형 회귀·로지스틱 회귀·랜덤 포레스트·SVM·K-Means 등.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

**SciPy** — *과학 계산*. 최적화·통계·신호 처리·희소 행렬.

이 다섯은 *서로 호환* 됩니다. NumPy 배열을 pandas 가 받고, pandas DataFrame 을 matplotlib 가 그리고, scikit-learn 이 NumPy 배열을 받습니다. 이 호환성이 Python 데이터 과학 생태계의 가장 큰 자산입니다.

### 1.3 가상환경 관리

Python 프로젝트는 *프로젝트마다 의존성이 달라지므로* 가상환경 분리가 필수입니다. 도구 옵션:

**venv** (표준 라이브러리, 가장 단순)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install numpy pandas
```

**conda** (Anaconda/Miniconda — 과학 컴퓨팅 패키지에 강점)

```bash
conda create -n ai-env python=3.11
conda activate ai-env
conda install numpy pandas
```

**uv** (최근 등장한 빠른 대안)

```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas
```

**poetry / pdm** (의존성 잠금 + 패키지 빌드)

이 책의 코드는 venv + pip 또는 uv 를 가정합니다. conda 는 일부 GPU 라이브러리(예: CUDA 자체) 를 함께 관리할 때 편리하지만, 패키지 해상도가 느린 단점이 있습니다.

### 1.4 패키지 관리 베스트 프랙티스

다음 세 가지를 습관화하면 환경 문제로 시간을 낭비하지 않습니다.

1. **`requirements.txt` 또는 `pyproject.toml` 으로 의존성을 명시.** 다음 사람·내일의 자기 자신이 같은 환경을 재현할 수 있어야 합니다.

```
# requirements.txt
torch>=2.4.0
transformers>=4.40.0
accelerate>=0.30.0
```

2. **버전 잠금.** `pip freeze > requirements-lock.txt` 또는 poetry/uv 의 lock 파일로 정확한 버전 조합을 보존합니다.

3. **시스템 Python 을 직접 쓰지 않기.** macOS·Linux 의 시스템 Python 은 OS 가 의존하는 경우가 많아, `pip install` 로 망가뜨리면 OS 가 영향을 받습니다. 항상 가상환경.

### 1.5 챕터 정리

Python 은 *두꺼운 생태계* 때문에 AI 엔지니어링의 표준 언어가 되었으며, NumPy·pandas·matplotlib·scikit-learn·SciPy 의 다섯이 데이터 과학의 핵심 빌딩블록입니다. 가상환경(venv, conda, uv) 분리와 의존성 명시는 환경 재현성의 기초입니다. 다음 챕터에서는 그 위에 올라가는 *딥러닝 프레임워크* 를 봅니다.

---

## 2. 딥러닝 프레임워크 — PyTorch·JAX·TensorFlow

### 2.1 세 프레임워크의 위치

현재 딥러닝 프레임워크는 세 개로 압축됩니다.

- **PyTorch** — Meta(구 Facebook) 가 개발. 학계·산업의 사실상 표준. 동적 그래프·Pythonic API.
- **TensorFlow** — Google 이 개발. 한때 표준이었으나 PyTorch 에 점진적으로 자리를 내줌. 모바일·임베디드 배포(TensorFlow Lite) 에 강점.
- **JAX** — Google Research 가 개발. 함수형 스타일. 학계 연구·대규모 학습에 강세 (특히 TPU).

이 세 프레임워크의 *현재 위상* 은 다음과 같습니다.

```
PyTorch:    학계 논문·산업 응용·Hugging Face 통합 — 가장 두꺼움
JAX:        Google·DeepMind·일부 SOTA 모델 학습 — 좁고 강함
TensorFlow: 레거시 시스템·모바일 배포 — 유지 보수 단계
```

신규 학습자는 *PyTorch 부터* 시작하는 것이 가장 합리적입니다. 이 책의 코드도 모두 PyTorch 입니다.

### 2.2 PyTorch 설치 — CUDA 버전 선택

PyTorch 설치 시 가장 중요한 결정은 *CUDA 버전 선택* 입니다. 시스템 CUDA 와 PyTorch 가 함께 빌드된 CUDA 가 호환되어야 GPU 가속이 작동합니다.

[pytorch.org](https://pytorch.org) 의 *Install* 페이지에서 자기 환경(OS·CUDA 버전)에 맞는 명령을 받습니다.

CUDA 12.4 + Linux 예시:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

CPU 만 사용 (GPU 가 없는 노트북):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

설치 후 검증:

```python
import torch
print(torch.__version__)         # 예: 2.4.0+cu124
print(torch.cuda.is_available()) # GPU 사용 가능 여부
print(torch.cuda.device_count()) # GPU 개수
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

`torch.cuda.is_available()` 가 `False` 이면 GPU 가속이 작동하지 않으며, 다음 항목들을 점검해야 합니다.

- NVIDIA 드라이버가 설치되어 있는가? (`nvidia-smi` 로 확인)
- CUDA 버전이 PyTorch 빌드와 호환되는가?
- WSL/Docker 환경이라면 GPU passthrough 가 활성화되어 있는가?

### 2.3 PyTorch 첫 코드

가장 단순한 PyTorch 코드:

```python
import torch
import torch.nn as nn

# 텐서 생성
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)  # torch.Size([2, 2])

# GPU 로 이동 (GPU 가 있는 경우)
if torch.cuda.is_available():
    x = x.to('cuda')
    print(x.device)  # cuda:0

# 단순 신경망
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

y = model(x)
print(y.shape)  # torch.Size([2, 1])
```

이 코드를 자기 환경에서 실행해 결과를 확인하시기 바랍니다. PyTorch 가 정상 동작한다면 다음 모든 권의 코드가 작동합니다.

### 2.4 PyTorch vs JAX vs TensorFlow 비교 (한 줄씩)

PyTorch:
- *동적 그래프* — Python 의 일반 흐름 제어로 모델 구조를 표현 가능
- *디버깅이 쉬움* — `print` 와 `pdb` 가 그대로 작동
- *Hugging Face 통합* — `transformers`·`diffusers` 가 1순위로 PyTorch 지원

JAX:
- *함수형 스타일* — 부작용 없는 변환 (`jit`, `grad`, `vmap`, `pmap`)
- *XLA 컴파일* — TPU·GPU 모두에서 빠른 실행
- *학습 곡선이 가파름* — Python 의 일반 사고와 다른 *함수형 사고* 필요

TensorFlow:
- *모바일·임베디드 배포* (TensorFlow Lite) 에 강점
- *Keras 고수준 API* 로 빠른 프로토타이핑
- *학계의 새 코드는 거의 PyTorch 로 이동* — TF 는 유지 보수 모드

### 2.5 챕터 정리

딥러닝 프레임워크는 *PyTorch·JAX·TensorFlow* 세 개로 정리되며, 신규 학습자는 PyTorch 로 시작하는 것이 가장 합리적입니다. 설치 시 *CUDA 버전 호환* 이 가장 흔한 함정이며, `torch.cuda.is_available()` 로 빠르게 검증합니다. 다음 챕터에서는 PyTorch 가 그 위에 올라가는 *GPU 스택* 을 봅니다.

---

## 3. GPU 스택 — CUDA·cuDNN·NCCL·CUTLASS·Triton

### 3.1 NVIDIA GPU 스택의 계층

GPU 가속은 단일 라이브러리가 아니라 *여러 계층의 라이브러리* 가 쌓인 결과입니다. 각 계층이 무엇을 담당하는지를 알아 두면, 성능 문제·설치 오류·버전 충돌이 발생했을 때 *어느 계층을 봐야 하는지* 가 명확해집니다.

```
+--------------------------------------------+
|  사용자 코드 (PyTorch / JAX / TensorFlow)   |
+--------------------------------------------+
|  고수준 라이브러리                          |
|  - cuDNN  (딥러닝 기본 연산)                |
|  - NCCL   (멀티 GPU 통신)                   |
|  - CUTLASS / cuBLAS (행렬 곱)               |
|  - Triton (커널 작성 언어)                  |
+--------------------------------------------+
|  CUDA Runtime / Driver API                  |
+--------------------------------------------+
|  CUDA Driver (커널 모드)                    |
+--------------------------------------------+
|  NVIDIA GPU 하드웨어                        |
+--------------------------------------------+
```

각 계층을 한 문단씩 설명합니다.

### 3.2 CUDA — 가장 아래 계층

**CUDA (Compute Unified Device Architecture)** 는 NVIDIA 가 만든 *GPU 위에서 일반 계산* 을 할 수 있게 해 주는 플랫폼입니다. CUDA 는 두 부분으로 구성됩니다.

- **드라이버 (Driver)** — OS 와 GPU 사이의 통신을 담당. NVIDIA 드라이버 설치 시 함께 설치.
- **툴킷 (Toolkit)** — 컴파일러(`nvcc`), 헤더, 라이브러리. 개발 시 필요.

`nvidia-smi` 명령으로 드라이버 버전과 GPU 상태를 확인합니다.

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10  Driver Version: 535.86.10  CUDA Version: 12.2         |
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC|
| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M.|
+===============================+======================+======================+
|   0  NVIDIA H100 80GB PCIe ON | 00000000:01:00.0 Off |                    0|
| N/A   35C    P0    52W / 350W |    0MiB / 81559MiB   |      0%      Default|
+-----------------------------------------------------------------------------+
```

이 출력에서 가장 자주 보는 정보:

- *Driver Version*: 시스템에 설치된 NVIDIA 드라이버
- *CUDA Version*: 드라이버가 *지원하는 최대* CUDA 버전 (실제 사용 버전과 다를 수 있음)
- *GPU Name*: GPU 모델명
- *Memory-Usage*: 현재 GPU 메모리 사용량
- *GPU-Util*: 현재 GPU 활용률 (%)

### 3.3 cuDNN — 딥러닝 기본 연산

**cuDNN (CUDA Deep Neural Network library)** 은 합성곱·풀링·정규화·활성화 같은 *딥러닝 기본 연산을 GPU 에서 빠르게 실행* 하는 라이브러리입니다. PyTorch·TensorFlow·JAX 가 모두 내부적으로 사용합니다.

설치는 보통 *PyTorch 와 함께 자동* 으로 됩니다. 별도 설치를 거의 신경 쓸 필요가 없지만, *버전이 모델 호환성에 영향* 을 줄 수 있어 인지하고 있는 것이 좋습니다.

### 3.4 NCCL — 멀티 GPU 통신

**NCCL (NVIDIA Collective Communications Library)** 는 *여러 GPU 사이에서 데이터를 빠르게 주고받는* 라이브러리입니다. 분산 학습·텐서 병렬·파이프라인 병렬에서 사용됩니다.

핵심 연산:

- **AllReduce** — 모든 GPU 가 자기 값을 합쳐 결과를 모두에게 다시 분배
- **AllGather** — 모든 GPU 가 자기 데이터를 모아 모두에게 분배
- **Broadcast** — 한 GPU 가 모든 GPU 에게 값 전송
- **ReduceScatter** — 합친 결과를 GPU 들에 분할

단일 GPU 환경에서는 NCCL 을 신경 쓸 일이 거의 없지만, 멀티 GPU 학습·추론으로 갈 때 등장합니다.

### 3.5 CUTLASS / cuBLAS — 행렬 곱

**cuBLAS** 는 행렬 곱·벡터 연산의 표준 라이브러리이며, **CUTLASS** 는 더 유연한 *템플릿 기반* 행렬 곱 라이브러리입니다. FlashAttention 같은 고성능 커널이 CUTLASS 를 기반으로 만들어집니다.

PyTorch 의 `torch.matmul` 은 내부적으로 cuBLAS 를 호출합니다. 사용자가 직접 호출할 일은 거의 없지만, *어디서 행렬 곱이 가속되는지* 의 그림을 잡아 두는 데 의미가 있습니다.

### 3.6 Triton — Python 으로 GPU 커널 작성

**Triton** 은 OpenAI 가 개발한 *Python 으로 GPU 커널을 작성* 할 수 있게 해 주는 언어입니다. C++ CUDA 보다 훨씬 낮은 진입장벽으로 *맞춤 커널* 을 만들 수 있어, 최근 빠르게 채택이 늘고 있습니다.

PyTorch 2.0 의 `torch.compile` 도 내부적으로 Triton 을 사용합니다.

Triton 커널의 모습:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

이 권에서는 Triton 의 자세한 사용법은 다루지 않으며, Vol 91 (CUDA Python 입문) 에서 본격적으로 다룹니다.

### 3.7 챕터 정리

GPU 스택은 *NVIDIA Driver → CUDA → cuDNN/NCCL/cuBLAS → Triton → 사용자 프레임워크* 의 계층 구조입니다. 일반 AI 엔지니어는 *드라이버와 CUDA 버전 호환* 만 신경 쓰면 되고, 나머지는 PyTorch 가 추상화합니다. 그러나 성능 문제 진단 시 *어느 계층의 문제인지* 를 식별할 수 있어야 합니다. 다음 챕터에서는 모델을 *어디서 가져오는지* 를 다룹니다.

---

## 4. 모델 허브 — Hugging Face Hub·transformers·diffusers

### 4.1 Hugging Face Hub

**Hugging Face Hub** 는 *모델·데이터셋·평가 셋의 GitHub* 같은 역할을 하는 저장소입니다. 50 만 개 이상의 모델, 25 만 개 이상의 데이터셋이 무료로 공개되어 있습니다. AI 엔지니어가 매일 방문하는 사이트 중 하나입니다.

주요 기능:

- *모델 검색·다운로드* — 작업·언어·라이선스로 필터링
- *모델 카드* — 모델의 학습 데이터·평가 결과·사용법·한계 정보
- *Spaces* — 모델 데모를 누구나 띄울 수 있는 호스팅 환경
- *Inference API* — 작은 모델은 무료 API 로 즉시 사용 가능
- *Datasets* — 표준 평가 셋·학습 데이터의 저장소

### 4.2 `transformers` 라이브러리

`transformers` 는 Hugging Face 가 만든 *Transformer 계열 모델 사용을 통합한* Python 라이브러리입니다. BERT·GPT·LLaMA·Mistral·Whisper·Stable Diffusion 등 거의 모든 주요 모델이 같은 인터페이스로 사용됩니다.

설치:

```bash
pip install transformers torch accelerate
```

가장 단순한 사용 예 — 사전학습된 BERT 로 문장 임베딩:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 모델 로드 (첫 실행 시 자동 다운로드)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# 입력 토큰화
text = "Hello, AI engineer."
inputs = tokenizer(text, return_tensors='pt')

# 모델 실행
with torch.no_grad():
    outputs = model(**inputs)

# 첫 토큰 ([CLS]) 의 임베딩이 문장 임베딩의 단순 형태
sentence_emb = outputs.last_hidden_state[0, 0, :]
print(sentence_emb.shape)  # torch.Size([768])
```

이 코드는 첫 실행 시 약 440MB 의 모델 가중치를 다운받습니다. 이후 실행은 캐시된 파일을 사용합니다 (`~/.cache/huggingface/`).

### 4.3 `diffusers` — 디퓨전 모델용

`diffusers` 는 Stable Diffusion 같은 *디퓨전 모델* 을 위한 라이브러리입니다.

```bash
pip install diffusers transformers accelerate
```

Stable Diffusion 으로 이미지 생성:

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
).to('cuda')

prompt = "a serene mountain landscape at sunset, high quality"
image = pipe(prompt).images[0]
image.save('output.png')
```

이 코드를 실행하려면 CUDA GPU 가 필요하며, 첫 실행 시 약 4GB 의 모델 가중치를 다운받습니다.

### 4.4 `accelerate` — 디바이스/분산 추상화

`accelerate` 는 *같은 코드를 CPU·GPU·멀티 GPU·TPU 에서 모두 돌게* 하는 추상화 라이브러리입니다. 디바이스 이동·혼합 정밀도·분산 학습을 *명시적 코드 없이* 처리합니다.

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

for batch in dataloader:
    outputs = model(batch)
    loss = compute_loss(outputs)
    accelerator.backward(loss)
    optimizer.step()
```

이 코드는 환경에 따라 자동으로 CPU·단일 GPU·DDP·FSDP 로 동작합니다.

### 4.5 모델 검색의 사고법

Hugging Face 에서 모델을 고를 때 다음 정보를 확인합니다.

- **다운로드 수** — 인기 = 안정성 (대부분의 경우)
- **마지막 갱신일** — 1 년 이상 갱신 안 된 모델은 의심
- **라이선스** — Apache 2.0, MIT, LLaMA license, Gemma, 상업 사용 가능 여부
- **모델 카드의 평가 결과** — 학습 데이터·평가 메트릭·한계
- **모델 사이즈** — 자기 GPU 메모리에 들어가는가
- **언어 지원** — 한국어·다국어 모델 여부

좋은 모델 카드 예시(Llama 3, Mistral 등) 를 몇 개 읽어 두면 *나쁜 모델 카드* 를 빠르게 식별할 수 있게 됩니다.

### 4.6 챕터 정리

Hugging Face Hub 는 *모델·데이터셋의 GitHub* 이며, `transformers`·`diffusers`·`accelerate` 가 모델 사용·생성·디바이스 추상화의 표준 라이브러리입니다. 모델 선택은 *다운로드 수·갱신일·라이선스·사이즈·언어 지원* 의 다섯 축으로 합니다. 다음 챕터에서는 *데이터* 를 다루는 도구를 봅니다.

---

## 5. 데이터 도구 — `datasets`·DVC·LakeFS

### 5.1 `datasets` 라이브러리

Hugging Face 의 `datasets` 라이브러리는 *데이터셋 다운로드·변환·스트리밍* 의 표준 도구입니다. 50 만 건 이상의 데이터셋이 동일한 인터페이스로 다뤄집니다.

설치 및 사용:

```bash
pip install datasets
```

```python
from datasets import load_dataset

# 데이터셋 다운로드 (첫 실행 시)
ds = load_dataset('imdb')
print(ds)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 첫 샘플 보기
print(ds['train'][0])
# {'text': '...', 'label': 1}

# 변환·필터링
filtered = ds['train'].filter(lambda x: len(x['text']) < 500)
mapped = filtered.map(lambda x: {'text_length': len(x['text'])})
```

대용량 데이터셋의 경우 *스트리밍 모드* 를 사용해 *전체를 다운받지 않고* 사용할 수 있습니다.

```python
ds = load_dataset('c4', 'en', streaming=True)
for sample in ds['train']:
    print(sample['text'][:100])
    break
```

### 5.2 데이터 버저닝의 필요

코드는 git 으로 버저닝하지만, *데이터* 는 git 에 직접 넣기엔 너무 큽니다. 그러나 *어떤 데이터로 학습한 모델이 어떤 결과를 만들었는가* 를 추적하지 못하면 실험의 재현성이 무너집니다.

데이터 버저닝 도구가 이 문제를 해결합니다.

### 5.3 DVC (Data Version Control)

**DVC** 는 *git 과 유사한 인터페이스로 데이터·모델 파일을 버저닝* 하는 도구입니다. 큰 파일은 외부 스토리지(S3, GCS, Azure Blob) 에 저장되고, *포인터(메타데이터) 만 git 에 들어갑니다*.

```bash
pip install dvc dvc-s3
dvc init
dvc remote add -d s3-storage s3://my-bucket/dvc
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "add training data"
dvc push
```

다른 사람은 같은 git 저장소를 클론한 뒤:

```bash
dvc pull  # 데이터를 외부 스토리지에서 받음
```

### 5.4 LakeFS

**LakeFS** 는 *데이터 레이크 자체를 git 처럼 다루는* 도구입니다. 객체 스토리지(S3) 위에 *브랜치·커밋·머지* 같은 git 추상화를 제공합니다.

DVC 가 *git 의 포인터로 데이터를 추적* 한다면, LakeFS 는 *데이터 자체에 git 추상화를 입힙니다*. 데이터 양이 매우 많고 변경이 잦은 환경에서 유리합니다.

### 5.5 어떤 도구를 언제 쓰는가

다음 가이드라인을 권장합니다.

- *데이터 < 100GB, 변경 빈도 낮음* — 그냥 git-lfs 로 충분
- *데이터 100GB-수 TB, 코드와 함께 버저닝* — DVC
- *데이터 수십 TB+, 데이터 자체가 자산* — LakeFS, Iceberg, Delta Lake
- *공개 데이터셋 활용* — Hugging Face `datasets` 로 충분

### 5.6 챕터 정리

데이터 도구는 *데이터 사용 (`datasets`)* 과 *데이터 버저닝 (DVC, LakeFS)* 의 두 영역으로 나뉩니다. 데이터 양과 변경 빈도에 따라 도구를 선택하며, 초기에는 `datasets` 만으로도 대부분의 학습이 가능합니다. 다음 챕터에서는 실험 결과를 추적하는 도구를 봅니다.

---

## 6. 실험 추적 — Weights & Biases·MLflow·TensorBoard

### 6.1 왜 실험 추적인가

ML/AI 개발은 *수많은 실험* 의 반복입니다. 학습률·배치 크기·아키텍처를 바꿔 가며 어떤 조합이 가장 좋은지를 찾습니다. 이 실험들의 결과를 *기억하고 비교* 하는 것이 실험 추적 도구의 역할입니다.

실험 추적 없이 일하면:

- *지난주에 어떤 학습률로 결과가 좋았는지* 가 기억나지 않음
- 같은 실험을 두 번 반복함
- 동료가 한 실험 결과를 알지 못해 다시 함
- *왜 이 모델이 좋은가* 를 설명하지 못함

### 6.2 Weights & Biases (W&B)

**Weights & Biases (wandb)** 는 가장 널리 쓰이는 실험 추적 SaaS 입니다. 무료 플랜으로 개인 프로젝트는 충분히 사용 가능합니다.

```bash
pip install wandb
wandb login  # 첫 사용 시 API 키 입력
```

코드 통합:

```python
import wandb

wandb.init(project='my-experiment', config={
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 10,
})

for epoch in range(10):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch,
    })

wandb.finish()
```

이 코드를 실행하면 W&B 웹 대시보드에서 실시간으로 그래프를 확인할 수 있고, 여러 실험을 *나란히 비교* 할 수 있습니다.

### 6.3 MLflow

**MLflow** 는 오픈소스 실험 추적·모델 레지스트리 도구입니다. 자체 호스팅이 가능하다는 장점이 있어, *데이터를 외부로 보낼 수 없는* 환경에서 선호됩니다.

```bash
pip install mlflow
```

```python
import mlflow

mlflow.set_experiment('my-experiment')

with mlflow.start_run():
    mlflow.log_param('learning_rate', 1e-3)
    mlflow.log_param('batch_size', 32)
    
    for epoch in range(10):
        train_loss = train_one_epoch(...)
        mlflow.log_metric('train_loss', train_loss, step=epoch)
    
    mlflow.pytorch.log_model(model, 'model')
```

```bash
mlflow ui  # 로컬 웹 UI 띄우기
```

### 6.4 TensorBoard

**TensorBoard** 는 TensorFlow 와 함께 등장한 시각화 도구이지만, PyTorch 도 `torch.utils.tensorboard.SummaryWriter` 로 지원합니다.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/my-experiment')

for epoch in range(10):
    train_loss = train_one_epoch(...)
    writer.add_scalar('train/loss', train_loss, epoch)

writer.close()
```

```bash
tensorboard --logdir=runs
```

W&B 와 MLflow 보다 *기능이 단순* 하지만 *완전 로컬·무료* 라는 장점이 있습니다.

### 6.5 어떤 도구를 언제 쓰는가

- *개인 학습·작은 팀* — W&B (UI 좋음, 무료 플랜 충분)
- *기업·자체 호스팅 필요* — MLflow
- *완전 로컬·간단함* — TensorBoard
- *LLM 추론 추적* — Helicone, Langfuse, LangSmith (Vol 96 에서 다룸)

이 책의 학습 권장 도구는 W&B 입니다. 무료 플랜이 충분하고 UI 가 직관적입니다.

### 6.6 챕터 정리

실험 추적 도구는 *수많은 실험의 결과를 기록·비교·재현* 하는 핵심 도구입니다. W&B (SaaS)·MLflow (오픈소스 자체 호스팅)·TensorBoard (단순) 가 표준 선택지이며, 환경에 따라 선택합니다. 다음 챕터에서는 *코드를 작성하는 환경* 자체를 봅니다.

---

## 7. 개발 환경 — Jupyter·VS Code·`python -i`·DevContainer

### 7.1 세 환경의 사용 시점

AI 엔지니어가 코드를 쓰는 환경은 크게 세 가지로 나뉩니다.

**Jupyter Notebook / JupyterLab.** 셀 단위로 코드를 실행하며 결과를 즉시 확인. *탐색적 분석·실험·시각화* 에 최적.

**VS Code (또는 PyCharm).** 일반 IDE. *프로덕션 코드 작성·리팩토링·디버깅·git 통합* 에 최적.

**`python -i` (인터랙티브 셸).** 가장 가벼운 REPL. *작은 실험·디버깅·라이브러리 시도* 에 최적.

세 환경의 분류 기준은 다음과 같습니다.

- *결과를 자주 보고 싶다* → Jupyter
- *큰 코드 베이스를 다룬다* → VS Code
- *한 번 빠르게 시도한다* → `python -i`

### 7.2 Jupyter 설치와 사용

```bash
pip install jupyter
jupyter notebook  # 또는 jupyter lab
```

브라우저가 열리며 노트북 환경이 시작됩니다. 셀 단위로 코드를 실행할 수 있고, 변수·시각화·중간 결과를 그대로 보존합니다.

학습 권장: *모든 본문 챕터의 코드는 Jupyter 노트북에 옮겨 한 셀씩 실행* 하시기 바랍니다. 이 습관이 학습 효과를 가장 크게 만듭니다.

### 7.3 VS Code + Python Extension

VS Code 는 *AI 엔지니어가 가장 많이 사용하는* 에디터입니다. Python Extension 을 설치하면 자동완성·디버깅·노트북 지원·git 통합·터미널이 모두 한 자리에 들어옵니다.

설치:
1. [code.visualstudio.com](https://code.visualstudio.com) 에서 VS Code 다운로드
2. 확장 마켓플레이스에서 *Python* 검색해 설치
3. `Python: Select Interpreter` 명령으로 가상환경 선택

VS Code 에서 노트북도 직접 열 수 있어, *Jupyter 와 IDE 의 장점을 결합* 할 수 있습니다.

### 7.4 `python -i` 인터랙티브 모드

가장 가벼운 옵션입니다.

```bash
python -i my_script.py
```

`my_script.py` 가 실행된 뒤 인터프리터가 *그대로 살아 있어*, 변수 상태를 들여다보거나 추가 코드를 실행할 수 있습니다. 작은 디버깅·실험에 매우 유용합니다.

`ipython` 을 설치하면 더 풍부한 REPL 을 사용할 수 있습니다.

```bash
pip install ipython
ipython
```

자동완성·구문 강조·매직 명령 (`%timeit`, `%load`, `%history`) 같은 기능이 추가됩니다.

### 7.5 DevContainer — 컨테이너 기반 개발

**DevContainer** 는 *개발 환경 자체를 Docker 컨테이너로 정의* 하는 방식입니다. 팀원 모두가 *완전히 동일한 환경* 에서 작업하게 만들어 *내 환경에서는 되는데...* 문제를 없앱니다.

`.devcontainer/devcontainer.json`:

```json
{
  "name": "AI Workbook",
  "image": "nvcr.io/nvidia/pytorch:24.04-py3",
  "extensions": ["ms-python.python"],
  "runArgs": ["--gpus=all"]
}
```

VS Code 가 이 정의를 읽어 *컨테이너 안에서 개발 환경을 자동 구성* 합니다. 신규 팀원이 오면 *git clone + DevContainer 열기* 만으로 개발 환경이 완성됩니다.

대규모 팀·복잡한 GPU 환경·온보딩이 잦은 환경에서 큰 가치가 있습니다.

### 7.6 챕터 정리

개발 환경은 *Jupyter (탐색)·VS Code (개발)·`python -i` (가벼운 실험)* 의 세 갈래로 나뉘며, 작업의 성격에 따라 선택합니다. DevContainer 는 *팀 전체의 환경 일관성* 을 확보하는 도구입니다. 다음 챕터에서는 GPU 가 없는 환경에서 *클라우드 GPU* 를 쓰는 옵션을 봅니다.

---

## 8. 클라우드 GPU

### 8.1 자체 GPU vs 클라우드 GPU

GPU 사용 옵션은 크게 두 가지입니다.

- **자체 GPU** — 노트북·데스크톱·자체 서버
- **클라우드 GPU** — Colab·Lambda Labs·RunPod·AWS·GCP

자체 GPU 가 유리한 경우:
- *지속적이고 빈번한 사용* — 클라우드 비용이 누적되면 자체 장비가 더 쌈
- *데이터 보안* — 외부에 데이터 송출 불가
- *낮은 지연* — 인터넷 속도에 영향받지 않음

클라우드 GPU 가 유리한 경우:
- *간헐적 사용* — 한 달에 며칠만 사용
- *최신 GPU 필요* — H100/H200 같은 최신 장비
- *대량 병렬* — 일시적으로 수십-수백 GPU 필요

### 8.2 Google Colab

**Google Colab** 은 무료로 시작할 수 있는 가장 쉬운 클라우드 GPU 옵션입니다.

- 무료 플랜: T4 GPU, 사용 시간 제한
- Colab Pro: A100 까지 사용 가능, 월 구독
- Jupyter 노트북 환경 그대로

```bash
# Colab 셀에서
!nvidia-smi
!pip install transformers
```

학습 단계에서 *작은 모델 실험* 에 매우 유용합니다. 단, *세션이 끊어지면 메모리·파일이 사라지므로* Google Drive 마운트나 외부 저장소를 사용해야 합니다.

### 8.3 Lambda Labs / RunPod

전문 GPU 클라우드. 시간당 과금이며, 가격이 AWS/GCP 보다 저렴한 경우가 많습니다.

- **Lambda Labs** — H100, A100, RTX A6000 등. SSH 로 접속.
- **RunPod** — 다양한 GPU. Jupyter 또는 SSH.
- **Vast.ai** — 사용자 GPU 를 빌리는 P2P 형태. 가장 저렴할 수 있음.
- **Modal** — 함수 단위로 GPU 호출. 서버리스.

선택 기준:
- *지속적 학습* — Lambda Labs (안정성)
- *일회성 실험* — RunPod, Vast.ai (가성비)
- *서버리스 추론* — Modal

### 8.4 AWS / GCP / Azure

대형 클라우드는 *EC2 / Compute Engine / VM* 인스턴스 형태로 GPU 를 제공합니다.

- AWS: `p4d.24xlarge` (8x A100), `p5.48xlarge` (8x H100)
- GCP: `a2-highgpu-1g` (A100), `a3-highgpu-8g` (H100)
- Azure: `Standard_ND96amsr_A100_v4` (A100)

전문 GPU 클라우드보다 *비싸지만* 다음 장점이 있습니다.

- *기존 인프라 통합* — 다른 서비스와 같은 VPC, IAM, 모니터링
- *예약 인스턴스* — 1-3 년 약정 시 큰 할인
- *Spot 인스턴스* — 일시적 사용 시 매우 저렴 (단, 중단 가능)

대규모 프로덕션 환경은 보통 AWS/GCP 를 선택합니다.

### 8.5 학습용 권장

이 책의 학습 단계에서는 다음 흐름을 권장합니다.

1. *없는 경우* — Google Colab 무료 플랜으로 시작
2. *학습이 진지해지면* — Colab Pro (월 약 10 달러) 또는 Lambda Labs (시간당 약 1 달러)
3. *프로덕션 흉내* — RunPod 또는 Modal 로 짧은 실험

자체 GPU 가 있다면 RTX 4090 (24GB) 한 장으로도 이 책의 거의 모든 권을 학습할 수 있습니다. 70B+ 모델 본격 학습은 H100 다수가 필요해 클라우드가 거의 필수입니다.

### 8.6 챕터 정리

GPU 옵션은 *자체 vs 클라우드* 로 나뉘며, 클라우드는 *Colab (입문)·Lambda Labs/RunPod (전문)·AWS/GCP (대규모)* 의 단계로 분류됩니다. 학습 단계에서는 Colab 또는 Lambda Labs 가 비용 효율이 높습니다. 다음 챕터에서는 *코드와 모델의 협업* 에 필요한 도구를 봅니다.

---

## 9. 버전 관리와 협업

### 9.1 git — 코드 버전 관리

git 은 모든 소프트웨어 엔지니어의 표준 도구이며, AI 엔지니어도 예외가 아닙니다. 이 책의 독자는 이미 git 을 알고 있다고 가정합니다.

AI 프로젝트에서 git 으로 버저닝하는 것:

- 학습 코드·추론 코드
- 설정 파일·하이퍼파라미터
- 데이터 처리 코드
- 평가 셋·평가 코드
- 모델 카드·문서

git 으로 버저닝하지 *않는* 것:

- 큰 데이터 파일 (DVC 사용)
- 모델 가중치 (Hugging Face Hub 또는 모델 레지스트리)
- 실험 결과 (W&B/MLflow)
- 학습 중간 산출물 (체크포인트)

### 9.2 Git LFS — 큰 파일

git 은 *작은 텍스트 파일* 을 위해 설계되었습니다. 큰 바이너리 파일(이미지·모델·데이터셋) 을 git 에 직접 넣으면 저장소가 빠르게 비대해집니다.

**Git LFS (Large File Storage)** 는 큰 파일을 *별도 스토리지* 에 저장하고 *git 에는 포인터만* 두는 확장입니다.

```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.bin"
git add .gitattributes
git add model.pkl
git commit -m "add model"
git push
```

DVC 가 더 강력하지만, *간단한 경우* 에는 Git LFS 만으로 충분합니다.

### 9.3 Hugging Face Hub 통합

Hugging Face Hub 자체가 git 저장소입니다. 모델·데이터셋을 다음과 같이 푸시할 수 있습니다.

```bash
huggingface-cli login
git lfs install
git clone https://huggingface.co/your-username/your-model
cd your-model
# 파일 추가
git add .
git commit -m "update model"
git push
```

또는 Python API:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./my-model',
    repo_id='your-username/your-model',
    repo_type='model',
)
```

Hub 에 올린 모델은 *전 세계 누구나* `from_pretrained('your-username/your-model')` 로 사용할 수 있게 됩니다.

### 9.4 협업 워크플로

AI 팀의 표준 협업 흐름:

1. **이슈 작성** — Linear, Jira, GitHub Issues 에 작업 정의
2. **브랜치 생성** — `git checkout -b feature/new-prompt-system`
3. **로컬 개발 + 실험** — Jupyter / VS Code, W&B 로 실험 결과 추적
4. **PR 생성** — 코드 + 실험 결과 링크 + 모델 카드 + 평가 결과
5. **리뷰** — 동료가 코드와 실험 결과를 함께 검토
6. **머지 + 배포** — CI/CD 파이프라인으로 자동 배포

이 흐름은 일반 백엔드 개발과 거의 같지만, *PR 에 실험 결과·모델 평가가 포함된다* 는 점이 추가됩니다. PR 본문에 *어떤 실험을 했고 어떤 결과가 나왔는지* 를 명시하는 습관이 팀 협업 품질을 크게 끌어올립니다.

### 9.5 RFC 와 의사결정 문서

큰 변경 (모델 교체·새 시스템 도입) 은 *RFC (Request for Comments)* 문서를 통해 합의를 거치는 것이 좋습니다.

RFC 표준 양식:

```markdown
# RFC: 추론 모델을 GPT-4o 에서 Claude 3.7 로 교체

## 배경
현재 GPT-4o 사용 중이며, 비용·지연·환각율 측면에서 ...

## 제안
Claude 3.7 Sonnet 으로 교체. 이유는 ...

## 대안
- Gemini 2.0 Pro: ...
- 자체 호스팅 LLaMA 3 70B: ...

## 위험
- 라이선스 변경
- 응답 스타일 변화
- 비용 변동성

## 측정 계획
- 100 건 샘플로 정확도 비교
- 1000 건 샘플로 지연·비용 비교
- 1주일 섀도우 트래픽 운영
```

RFC 문화는 *결정 근거를 문서화* 함으로써 팀의 *제도적 기억* 을 만듭니다.

### 9.6 챕터 정리

git 은 코드·설정·평가 셋·문서를 버저닝하고, Git LFS·DVC·Hugging Face Hub 가 큰 파일을 보완합니다. AI 협업은 *PR 에 실험 결과를 포함* 하고 *큰 변경은 RFC 로 합의* 하는 워크플로 위에서 이루어집니다.

---

## 권 정리

이 권에서 우리는 다음을 배웠습니다.

- **Python** 은 두꺼운 생태계 때문에 표준이며, NumPy·pandas·matplotlib·scikit-learn 의 기반 위에서 작업합니다.
- **딥러닝 프레임워크** 는 PyTorch 가 사실상 표준이며, CUDA 버전 호환이 가장 흔한 함정입니다.
- **GPU 스택** 은 Driver → CUDA → cuDNN/NCCL/cuBLAS → 사용자 프레임워크의 계층 구조이며, 일반 사용자는 드라이버와 CUDA 버전만 신경 쓰면 됩니다.
- **모델 허브** 는 Hugging Face Hub 가 표준이며, `transformers`·`diffusers`·`accelerate` 가 모델 사용의 핵심 라이브러리입니다.
- **데이터 도구** 는 `datasets` 로 사용·DVC/LakeFS 로 버저닝합니다.
- **실험 추적** 은 W&B (SaaS), MLflow (오픈소스), TensorBoard (단순) 가 표준이며 *수많은 실험을 기억·비교* 하는 데 필수입니다.
- **개발 환경** 은 Jupyter (탐색)·VS Code (개발)·`python -i` (가벼운 실험) 로 구분되며, DevContainer 가 팀 환경 일관성을 만듭니다.
- **클라우드 GPU** 는 Colab (입문)·Lambda Labs/RunPod (전문)·AWS/GCP (대규모) 의 단계로 선택합니다.
- **버전 관리·협업** 은 git + Git LFS/DVC + Hugging Face Hub + W&B + RFC 의 결합으로 이뤄집니다.

가장 기억에 남겨야 할 한 줄은 **"도구는 수단이며, 각 도구가 푸는 문제 영역을 식별할 수 있으면 새 도구도 빠르게 익힐 수 있다."** 입니다.

이 권을 마치는 시점에 자기 노트북에 다음 환경이 구축되어 있어야 합니다.

```bash
python --version  # 3.11+
pip list | grep torch          # torch 2.4+ 설치
pip list | grep transformers   # transformers 4.40+ 설치
python -c "import torch; print(torch.cuda.is_available())"  # True (GPU 있는 경우)
```

다음 권은 [Volume 4 — 선형대수 1 — 벡터·행렬·내적·노름](./volume_08_linear_algebra_1.md) 입니다. 거기서는 모든 데이터를 *벡터·행렬·텐서* 로 추상화해 사고하는 언어를 다룹니다.

---

## 자가점검 키워드

`PyTorch`, `CUDA`, `cuDNN`, `Hugging Face`, `transformers`, `W&B`, `MLflow`, `Jupyter`

## 자가점검 질문

다음 질문에 막힘없이 답할 수 있을 때 다음 권으로 넘어가십시오.

1. PyTorch·JAX·TensorFlow 세 프레임워크의 현재 위상을 비교하고, 신규 학습자에게 어느 것을 권장하는지 그 이유를 적으십시오.
2. NVIDIA GPU 스택을 *Driver·CUDA·cuDNN·NCCL·CUTLASS·Triton* 의 계층으로 그리고, 각 계층이 무엇을 담당하는지 한 줄씩 설명하십시오.
3. Hugging Face `transformers` 의 `AutoModel.from_pretrained('bert-base-uncased')` 한 줄이 내부적으로 어떤 일을 하는지 단계별로 적으십시오.
4. 데이터 버저닝이 *git 만으로 충분하지 않은* 이유와 DVC·LakeFS·Git LFS 의 차이를 설명하십시오.
5. W&B·MLflow·TensorBoard 의 적용 시점을 구분하고, 각 도구를 추천할 환경을 한 가지씩 적으십시오.
6. Jupyter·VS Code·`python -i` 의 사용 시점을 구분하고, 자신이 가장 자주 쓸 환경을 정하십시오.
7. 자체 GPU 가 없는 학습자에게 *4 단계 클라우드 GPU 로드맵* (입문 → 진지 → 프로덕션 흉내) 을 추천하고, 각 단계의 비용 추정을 적으십시오.

## 다음 권

[Volume 4 — 선형대수 1 — 벡터·행렬·내적·노름](./volume_08_linear_algebra_1.md)
