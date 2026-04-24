# AI Engineer Workbook

이 책은 AI에 대한 사전 지식이 거의 없는 일반 개발자가 AI 엔지니어로 전직하기 위해 필요한 모든 개념을 처음부터 끝까지 단권으로 학습할 수 있도록 만든 워크북입니다. 특정 모델에 종속되지 않으며, 어디에서든 AI 추론·서빙 업무를 시작할 수 있는 수준의 기초 체력을 만드는 것을 목표로 합니다.

이 워크북은 *읽기만 하는 책*이 아니라 *직접 손으로 따라 하면서 익히는 책*입니다. 각 권은 개념의 의미·필요성·내부 원리·코드 구현·점검 질문을 포함하며, 모든 코드는 실제로 실행 가능한 형태로 제공됩니다.

---

## 구성 원칙

이 워크북은 다음 네 가지 원칙 아래 작성되었습니다.

### 1. 백엔드·웹 개발 경험을 자산으로 활용합니다

이 책의 독자는 한 가지 이상의 프로그래밍 언어를 다룰 줄 알고 REST API·데이터베이스·메시지 큐·캐시 같은 백엔드 시스템에 익숙한 개발자입니다. 새로운 개념을 도입할 때마다 가능한 한 이러한 기존 지식과 연결합니다. *모델 가중치는 서비스 시작 시 로드되는 거대한 설정 객체*에 비유하고, *토큰 생성 루프는 응답 스트리밍*에 비유하는 식입니다. 이러한 연결을 통해 새로운 용어를 *완전히 새로운 것*이 아니라 *이미 알고 있는 무엇과 비슷한 것*으로 받아들이도록 돕습니다.

### 2. 본문은 표·코드가 아니라 문장이 중심입니다

기술 문서에서 표와 코드가 많아지면 한 번에 훑어보기는 좋지만 *왜 그런가*를 이해하기는 어렵습니다. 이 책의 모든 챕터는 충분히 긴 본문 설명으로 시작하여 개념의 배경·필요성·역사적 맥락·실제 적용 사례를 서술합니다. 표는 비교가 본질적으로 필요한 경우에만 등장하며, 코드는 *말로 설명한 내용을 손으로 확인*하기 위한 수단으로 등장합니다.

### 3. 모호한 표현을 배제합니다

"보통은", "일반적으로", "대체로" 같은 표현은 독자에게 정확한 그림을 그리지 못하게 합니다. 이 책에서는 가능한 한 구체적인 수치, 명시적인 조건, 관찰 가능한 행동으로 표현합니다. *모델이 빠르다*가 아니라 *H100 한 장에서 4 step 추론으로 약 1.6 초가 소요됩니다*와 같이 적습니다.

### 4. 한 번에 한 가지만 가르칩니다

새로운 개념을 도입할 때 동시에 너무 많은 신개념을 끌어오지 않습니다. 권 사이의 의존성을 명시적으로 표현하고, 어떤 개념이 다른 권에서 먼저 다뤄졌는지 항상 표시합니다.

---

## 인덱스 (전 50권)

### Track A — 입문과 직무 지형

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 1 | [AI 엔지니어라는 직업의 본질](./volume_01_what_is_ai_engineer.md) | AI 엔지니어가 무엇을 하는 사람이고, ML 엔지니어·AI 리서처·MLOps 와 어떻게 다른가 |
| 2 | [AI·ML·DL의 관계와 현대 AI의 큰 그림](./volume_02_ai_ml_dl_landscape.md) | 용어 정리, 패러다임 변천, 현대 AI 시스템의 구성요소 |
| 3 | [도구 체인과 작업 환경](./volume_03_toolchain.md) | Python·PyTorch·CUDA·HuggingFace·실험 추적 도구의 전체 지도 |

### Track B — 수학과 과학 컴퓨팅

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 4 | [선형대수 1 — 벡터·행렬·내적·노름](./volume_04_linear_algebra_1.md) | 데이터를 다루는 가장 기본 언어 |
| 5 | [선형대수 2 — 행렬 분해·고유값·SVD](./volume_05_linear_algebra_2.md) | 차원축소·임베딩의 수학적 근거 |
| 6 | [미적분과 그래디언트](./volume_06_calculus_gradient.md) | 학습이 *기울기를 따라 내려가는 일*임을 이해 |
| 7 | [확률과 통계](./volume_07_probability_statistics.md) | 분포·기댓값·베이즈 정리·MLE·MAP |
| 8 | [정보 이론](./volume_08_information_theory.md) | 엔트로피·교차엔트로피·KL — 손실 함수의 본질 |
| 9 | [NumPy로 다시 푸는 수학](./volume_09_numpy.md) | 모든 수학 개념을 코드로 검증하는 사고법 |

### Track C — 머신러닝

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 10 | [머신러닝의 본질과 학습 패러다임](./volume_10_machine_learning_essence.md) | 지도·비지도·자가지도·강화학습의 차이 |
| 11 | [회귀](./volume_11_regression.md) | 선형·다항·릿지·라쏘·엘라스틱넷 |
| 12 | [분류](./volume_12_classification.md) | 로지스틱 회귀·의사결정나무·k-NN |
| 13 | [앙상블](./volume_13_ensemble.md) | 배깅·부스팅·랜덤포레스트·XGBoost·LightGBM |
| 14 | [SVM과 커널 방법](./volume_14_svm_kernel.md) | 마진의 직관·커널 트릭 |
| 15 | [비지도학습](./volume_15_unsupervised.md) | K-Means·GMM·DBSCAN·계층적 군집 |
| 16 | [차원 축소와 시각화](./volume_16_dimensionality_reduction.md) | PCA·t-SNE·UMAP |

### Track D — 딥러닝 핵심

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 17 | [퍼셉트론과 신경망의 기원](./volume_17_perceptron.md) | 생물 뉴런부터 XOR 문제까지 |
| 18 | [MLP와 유니버설 근사](./volume_18_mlp_universal_approx.md) | 신경망이 *어떤 함수든 근사할 수 있다*는 의미 |
| 19 | [역전파와 자동미분](./volume_19_backprop_autograd.md) | 그래디언트가 흐르는 길 |
| 20 | [옵티마이저](./volume_20_optimizers.md) | SGD·모멘텀·Adam·AdamW·LR 스케줄 |
| 21 | [정규화 (Normalization)](./volume_21_normalization.md) | BatchNorm·LayerNorm·GroupNorm·RMSNorm |
| 22 | [정칙화 (Regularization)](./volume_22_regularization.md) | Dropout·Weight Decay·Early Stopping·Augmentation |
| 23 | [PyTorch 실전](./volume_23_pytorch_practice.md) | 텐서·모듈·학습 루프·체크포인트 |
| 24 | [데이터 파이프라인과 실험 관리](./volume_24_data_pipeline.md) | Dataset·DataLoader·W&B·MLflow |
| 25 | [모델 평가와 메트릭](./volume_25_evaluation_metrics.md) | 분류·회귀·생성 모델 평가 지표 |

### Track E — 컴퓨터 비전

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 26 | [합성곱 신경망(CNN)](./volume_26_cnn.md) | 컨볼루션·풀링·수용 영역 |
| 27 | [CNN 모델 계보](./volume_27_cnn_lineage.md) | AlexNet → VGG → ResNet → EfficientNet |
| 28 | [객체 탐지와 세그멘테이션](./volume_28_detection_segmentation.md) | YOLO·Faster R-CNN·Mask R-CNN·SAM |
| 29 | [Vision Transformer (ViT)](./volume_29_vit.md) | 이미지를 패치로 다루는 사고 전환 |

### Track F — 시퀀스와 트랜스포머

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 30 | [RNN·LSTM·GRU와 그 한계](./volume_30_rnn_lstm.md) | 시퀀스 모델의 진화사 |
| 31 | [Attention 메커니즘](./volume_31_attention.md) | *관련도를 가중평균*하는 단순한 아이디어 |
| 32 | [Transformer 완전 정복](./volume_32_transformer.md) | Self-Attention·Multi-Head·FFN·Position Encoding |

### Track G — 임베딩과 벡터 검색

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 33 | [단어 임베딩](./volume_33_word_embedding.md) | Word2Vec·GloVe·FastText |
| 34 | [문장·문서 임베딩](./volume_34_sentence_embedding.md) | Sentence-BERT·E5·OpenAI Embedding |
| 35 | [벡터 검색과 ANN](./volume_35_vector_search.md) | FAISS·HNSW·IVF·Hybrid Search |

### Track H — 대규모 언어 모델 (LLM)

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 36 | [토크나이저](./volume_36_tokenizer.md) | BPE·WordPiece·SentencePiece·Tiktoken |
| 37 | [LLM 사전학습과 스케일링 법칙](./volume_37_llm_pretraining.md) | GPT 계보·LLaMA 계보·Chinchilla 법칙 |
| 38 | [LLM 정렬](./volume_38_llm_alignment.md) | SFT·RLHF·DPO·Constitutional AI |
| 39 | [프롬프트와 In-Context Learning](./volume_39_prompting.md) | Zero/Few-shot·CoT·ToT |
| 40 | [RAG — 검색 증강 생성](./volume_40_rag.md) | 인덱싱·검색·재순위·답변 합성 |

### Track I — AI Agent

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 41 | [에이전트의 본질](./volume_41_agent_essence.md) | Tool Use·Function Calling·ReAct |
| 42 | [에이전트 프레임워크](./volume_42_agent_frameworks.md) | LangChain·LlamaIndex·MCP·Claude Agent SDK |
| 43 | [멀티에이전트와 워크플로](./volume_43_multi_agent.md) | Orchestration·Planner-Executor·Memory·Self-Reflection |

### Track J — 생성 모델

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 44 | [GAN과 적대적 학습](./volume_44_gan.md) | Generator vs Discriminator·DCGAN·StyleGAN |
| 45 | [VAE와 잠재 변수 모델](./volume_45_vae.md) | 인코더·디코더·KL 항·재매개화 |
| 46 | [Image Diffusion](./volume_46_image_diffusion.md) | DDPM·DDIM·Latent Diffusion·Rectified Flow·Distillation·CFG |

### Track K — GPU와 추론 가속

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 47 | [GPU 아키텍처와 CUDA](./volume_47_gpu_cuda.md) | SM·워프·메모리 계층·Tensor Core·자료형 |
| 48 | [추론 최적화](./volume_48_inference_optimization.md) | 양자화·`torch.compile`·TensorRT·KV 캐시·FlashAttention |
| 49 | [추론 서빙 시스템](./volume_49_serving_systems.md) | vLLM·Triton·TGI·동적·연속 배칭·오토스케일 |

### Track L — 운영과 책임

| # | 제목 | 한 줄 요약 |
|---|------|-----------|
| 50 | [AI 시스템의 운영·관측·비용·거버넌스](./volume_50_ops_governance.md) | 모니터링·A/B 테스트·비용 모델·안전·평가 파이프라인 |

---

## 학습 경로

전직 시점과 가용 시간에 따라 세 가지 경로를 제안합니다. 어느 경로든 Volume 1과 마지막 Track L 의 운영 권은 반드시 포함됩니다.

### 빠른 진입 (8 주, 추론 서빙 트랙)

서빙·추론 운영부터 진입하려는 백엔드 엔지니어를 위한 경로입니다. 학습 알고리즘은 의도적으로 건너뛰고, *이미 학습된 모델을 가져다 운영하는* 관점에 집중합니다.

`1 → 2 → 3 → 23 → 31 → 32 → 47 → 48 → 49 → 50`

### 표준 (6 개월)

모델 자체를 이해하면서 추론까지 다루는 경로입니다.

`1 → 2 → 3 → 6 → 7 → 10 → 17 → 18 → 19 → 20 → 23 → 26 → 30 → 31 → 32 → 36 → 37 → 39 → 40 → 41 → 47 → 48 → 49 → 50`

### 마스터 (12 개월)

모든 50권을 순서대로 학습하고, 각 권의 자가점검 키워드와 코드를 모두 손으로 작성합니다.

`Volume 1 → 2 → 3 → ... → 50`

---

## 의존성 그래프 (트랙 단위)

```
Track A (직무)
    │
    ├─→ Track B (수학) ──→ Track C (ML) ──→ Track D (DL 핵심)
    │                                                │
    │                                                ├─→ Track E (CV)
    │                                                ├─→ Track F (시퀀스/Transformer)
    │                                                │       │
    │                                                │       ├─→ Track G (임베딩)
    │                                                │       └─→ Track H (LLM) ──→ Track I (Agent)
    │                                                │
    │                                                ├─→ Track J (생성 모델)
    │                                                │
    │                                                └─→ Track K (GPU/추론) ──→ Track L (운영)
```

화살표가 없는 권도 독립적으로 읽을 수 있지만, 화살표를 따라가면 개념 누락 없이 학습할 수 있습니다.

---

## 사용 방법

### 1. 한 권을 끝까지 읽습니다

각 권은 *목적 → 선수 지식 → 학습 결과 → 챕터 본문 → 자가점검*의 흐름으로 구성됩니다. 자가점검 키워드를 자기 말로 설명할 수 있을 때 다음 권으로 넘어갑니다.

### 2. 모든 코드를 직접 실행합니다

코드 블록은 복사해서 그대로 실행할 수 있도록 작성됩니다. `python -i` 로 한 줄씩 실행하면서 변수의 모양과 값을 직접 확인하시기 바랍니다.

### 3. 자기만의 메모를 남깁니다

각 권의 핵심 통찰은 자신의 표현으로 다시 적어보면 더 오래 기억에 남습니다.

---

## 환경 준비

이 워크북의 코드는 다음 환경을 가정합니다.

- Python 3.11 이상
- PyTorch 2.4 이상
- CUDA 사용 가능한 NVIDIA GPU 1 장 (없으면 CPU 로도 학습 권까지는 진행 가능)
- (선택) Hugging Face `transformers` 4.40 이상

설치 예시는 다음과 같습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
```

GPU가 없는 환경에서는 Track K, L 의 일부 실습이 제약될 수 있으나, 개념 학습에는 지장이 없습니다.

---

## 라이선스 및 사용

이 워크북은 학습 목적으로 자유롭게 활용하실 수 있습니다. 책 안의 코드 또한 별도의 표시가 없는 한 학습·변형·재배포가 가능합니다.

다음 권으로 진행하시려면 [Volume 1 — AI 엔지니어라는 직업의 본질](./volume_01_what_is_ai_engineer.md) 을 펼쳐 주십시오.
