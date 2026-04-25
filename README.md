# AI Engineer Workbook

> 소프트웨어 엔지니어인데 AI 엔지니어로 전직하고 싶고, 막막하고, 뭐부터 해야 할지 모르겠다면 — **그냥 이것만 보십시오.**

이 워크북은 AI 사전 지식이 거의 없는 개발자가 처음부터 끝까지 단권으로 따라갈 수 있도록 86권으로 구성한 학습서입니다. 특정 모델에 종속되지 않으며, 어디에서든 AI 추론·서빙 업무를 시작할 수 있는 수준의 기초 체력을 만드는 것을 목표로 합니다.

각 권은 *목적 → 선수 지식 → 학습 결과 → 챕터 본문 → 자가점검* 의 일관된 구조를 따르며, 코드는 모두 실제로 실행 가능한 형태로 제공됩니다.

---

## 인덱스 (전 86권)

### Track A — 입문과 직무 지형

- [01](./volume_01_what_is_ai_engineer.md) — AI 엔지니어라는 직업의 본질
- [02](./volume_02_ai_ml_dl_landscape.md) — AI·ML·DL의 관계와 현대 AI의 큰 그림
- [03](./volume_03_toolchain.md) — 도구 체인과 작업 환경
- [51](./volume_51_collaboration.md) — 협업과 의사소통
- [52](./volume_52_ethics.md) — AI 엔지니어의 윤리와 책임
- [53](./volume_53_info_tracking.md) — 정보 추적 시스템

### Track B — 수학과 과학 컴퓨팅

- [04](./volume_04_linear_algebra_1.md) — 선형대수 1 — 벡터·행렬·내적·노름
- [05](./volume_05_linear_algebra_2.md) — 선형대수 2 — 행렬 분해·고유값·SVD
- [06](./volume_06_calculus_gradient.md) — 미적분과 그래디언트
- [07](./volume_07_probability_statistics.md) — 확률과 통계
- [08](./volume_08_information_theory.md) — 정보 이론
- [09](./volume_09_numpy.md) — NumPy로 다시 푸는 수학
- [54](./volume_54_numerical.md) — 수치 해석과 부동소수점
- [55](./volume_55_optimization.md) — 최적화 이론
- [56](./volume_56_stochastic_process.md) — 확률 과정 — Markov·SDE

### Track C — 머신러닝

- [10](./volume_10_machine_learning_essence.md) — 머신러닝의 본질과 학습 패러다임
- [11](./volume_11_regression.md) — 회귀
- [12](./volume_12_classification.md) — 분류
- [13](./volume_13_ensemble.md) — 앙상블
- [14](./volume_14_svm_kernel.md) — SVM과 커널 방법
- [15](./volume_15_unsupervised.md) — 비지도학습
- [16](./volume_16_dimensionality_reduction.md) — 차원 축소와 시각화
- [57](./volume_57_feature_engineering.md) — 특성 공학
- [58](./volume_58_bayesian_ml.md) — 베이지안 머신러닝
- [59](./volume_59_hpo.md) — 하이퍼파라미터 탐색

### Track D — 딥러닝 핵심

- [17](./volume_17_perceptron.md) — 퍼셉트론과 신경망의 기원
- [18](./volume_18_mlp_universal_approx.md) — MLP와 유니버설 근사
- [19](./volume_19_backprop_autograd.md) — 역전파와 자동미분
- [20](./volume_20_optimizers.md) — 옵티마이저
- [21](./volume_21_normalization.md) — 정규화 (Normalization)
- [22](./volume_22_regularization.md) — 정칙화 (Regularization)
- [23](./volume_23_pytorch_practice.md) — PyTorch 실전
- [24](./volume_24_data_pipeline.md) — 데이터 파이프라인과 실험 관리
- [25](./volume_25_evaluation_metrics.md) — 모델 평가와 메트릭
- [60](./volume_60_activations.md) — 활성화 함수 깊이
- [61](./volume_61_lr_schedules.md) — 학습률 스케줄 깊이
- [62](./volume_62_dl_debugging.md) — 딥러닝 디버깅

### Track E — 컴퓨터 비전

- [26](./volume_26_cnn.md) — 합성곱 신경망(CNN)
- [27](./volume_27_cnn_lineage.md) — CNN 모델 계보
- [28](./volume_28_detection_segmentation.md) — 객체 탐지와 세그멘테이션
- [29](./volume_29_vit.md) — Vision Transformer (ViT)
- [63](./volume_63_augmentation.md) — 데이터 증강 깊이
- [64](./volume_64_ssl_vision.md) — 자가지도 비전
- [65](./volume_65_multimodal_vl.md) — 멀티모달 비전-언어

### Track F — 시퀀스와 트랜스포머

- [30](./volume_30_rnn_lstm.md) — RNN·LSTM·GRU와 그 한계
- [31](./volume_31_attention.md) — Attention 메커니즘
- [32](./volume_32_transformer.md) — Transformer 완전 정복
- [66](./volume_66_long_context.md) — Long Context 기법
- [67](./volume_67_moe.md) — Mixture of Experts (MoE)
- [68](./volume_68_efficient_attention.md) — 효율적 어텐션 변형

### Track G — 임베딩과 벡터 검색

- [33](./volume_33_word_embedding.md) — 단어 임베딩
- [34](./volume_34_sentence_embedding.md) — 문장·문서 임베딩
- [35](./volume_35_vector_search.md) — 벡터 검색과 ANN
- [69](./volume_69_multimodal_embedding.md) — 멀티모달 임베딩
- [70](./volume_70_embedding_eval.md) — 임베딩 평가와 벤치마크
- [71](./volume_71_reranker.md) — Reranker 깊이

### Track H — 대규모 언어 모델 (LLM)

- [36](./volume_36_tokenizer.md) — 토크나이저
- [37](./volume_37_llm_pretraining.md) — LLM 사전학습과 스케일링 법칙
- [38](./volume_38_llm_alignment.md) — LLM 정렬
- [39](./volume_39_prompting.md) — 프롬프트와 In-Context Learning
- [40](./volume_40_rag.md) — RAG — 검색 증강 생성
- [72](./volume_72_decoding.md) — LLM 디코딩 알고리즘
- [73](./volume_73_model_sizing.md) — 모델 사이즈 의사결정
- [74](./volume_74_constrained_gen.md) — Constrained Generation

### Track I — AI Agent

- [41](./volume_41_agent_essence.md) — 에이전트의 본질
- [42](./volume_42_agent_frameworks.md) — 에이전트 프레임워크
- [43](./volume_43_multi_agent.md) — 멀티에이전트와 워크플로
- [75](./volume_75_agent_memory.md) — Agent Memory 시스템
- [76](./volume_76_agent_eval.md) — Agent 평가와 벤치마크
- [77](./volume_77_computer_use.md) — Computer Use·Browser Agent

### Track J — 생성 모델

- [44](./volume_44_gan.md) — GAN과 적대적 학습
- [45](./volume_45_vae.md) — VAE와 잠재 변수 모델
- [46](./volume_46_image_diffusion.md) — Image Diffusion
- [78](./volume_78_generation_control.md) — 생성 제어 기법
- [79](./volume_79_img2img.md) — Image-to-Image·Inpainting·Outpainting
- [80](./volume_80_video_3d.md) — 비디오·3D 생성

### Track K — GPU와 추론 가속

- [47](./volume_47_gpu_cuda.md) — GPU 아키텍처와 CUDA
- [48](./volume_48_inference_optimization.md) — 추론 최적화
- [49](./volume_49_serving_systems.md) — 추론 서빙 시스템
- [81](./volume_81_kv_cache.md) — KV 캐시 깊이
- [82](./volume_82_speculative.md) — Speculative Decoding
- [83](./volume_83_distributed_training.md) — 분산 학습

### Track L — 운영과 책임

- [50](./volume_50_ops_governance.md) — AI 시스템의 운영·관측·비용·거버넌스
- [84](./volume_84_monitoring_deep.md) — 모델 모니터링 깊이
- [85](./volume_85_postmortem.md) — AI 인시던트 사후 분석
- [86](./volume_86_data_governance.md) — 데이터 거버넌스·라벨링

---

## 학습 경로

목표와 가용 시간에 따라 세 가지 경로를 제안합니다. 어느 경로든 Volume 1과 마지막 Track L 의 운영 권은 반드시 포함됩니다.

### 빠른 진입 — 추론 서빙 트랙

서빙·추론 운영부터 진입하려는 개발자를 위한 경로입니다. 학습 알고리즘은 의도적으로 건너뛰고, *이미 학습된 모델을 가져다 운영하는* 관점에 집중합니다.

`1 → 2 → 3 → 23 → 31 → 32 → 47 → 48 → 49 → 50`

### 표준

모델 자체를 이해하면서 추론까지 다루는 경로입니다.

`1 → 2 → 3 → 6 → 7 → 10 → 17 → 18 → 19 → 20 → 23 → 26 → 30 → 31 → 32 → 36 → 37 → 39 → 40 → 41 → 47 → 48 → 49 → 50`

### 마스터

모든 86권을 순서대로 학습하고, 각 권의 자가점검 키워드와 코드를 모두 손으로 작성합니다.

`Volume 1 → 2 → 3 → ... → 86`

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
