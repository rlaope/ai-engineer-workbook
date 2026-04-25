# AI Engineer Workbook

> 소프트웨어 엔지니어인데 AI 엔지니어로 전직하고 싶고, 막막하고, 뭐부터 해야 할지 모르겠다면 — **그냥 이것만 보십시오.**

이 워크북은 AI 사전 지식이 거의 없는 개발자가 처음부터 끝까지 단권으로 따라갈 수 있도록 86권으로 구성한 학습서입니다. 특정 모델에 종속되지 않으며, 어디에서든 AI 추론·서빙 업무를 시작할 수 있는 수준의 기초 체력을 만드는 것을 목표로 합니다.

각 권은 *목적 → 선수 지식 → 학습 결과 → 챕터 본문 → 자가점검* 의 일관된 구조를 따르며, 코드는 모두 실제로 실행 가능한 형태로 제공됩니다.

---

## 인덱스 (전 104권)

### Track A — 입문과 직무 지형

- [01 — AI 엔지니어라는 직업의 본질](./volume_01_what_is_ai_engineer.md)
- [02 — AI·ML·DL의 관계와 현대 AI의 큰 그림](./volume_02_ai_ml_dl_landscape.md)
- [03 — 도구 체인과 작업 환경](./volume_03_toolchain.md)
- [04 — 협업과 의사소통](./volume_04_collaboration.md)
- [05 — AI 엔지니어의 윤리와 책임](./volume_05_ethics.md)
- [06 — 정보 추적 시스템](./volume_06_info_tracking.md)
- [07 — AI 애플리케이션 기획·PRD](./volume_07_ai_app_planning.md)

### Track B — 수학과 과학 컴퓨팅

- [08 — 선형대수 1 — 벡터·행렬·내적·노름](./volume_08_linear_algebra_1.md)
- [09 — 선형대수 2 — 행렬 분해·고유값·SVD](./volume_09_linear_algebra_2.md)
- [10 — 미적분과 그래디언트](./volume_10_calculus_gradient.md)
- [11 — 확률과 통계](./volume_11_probability_statistics.md)
- [12 — 정보 이론](./volume_12_information_theory.md)
- [13 — NumPy로 다시 푸는 수학](./volume_13_numpy.md)
- [14 — 수치 해석과 부동소수점](./volume_14_numerical.md)
- [15 — 최적화 이론](./volume_15_optimization.md)
- [16 — 확률 과정 — Markov·SDE](./volume_16_stochastic_process.md)
- [17 — 통계적 학습 이론](./volume_17_learning_theory.md)
- [18 — 정보 기하와 매니폴드](./volume_18_information_geometry.md)
- [19 — 그래프 이론과 스펙트럴 방법](./volume_19_graph_spectral.md)

### Track C — 머신러닝

- [20 — 머신러닝의 본질과 학습 패러다임](./volume_20_machine_learning_essence.md)
- [21 — 회귀](./volume_21_regression.md)
- [22 — 분류](./volume_22_classification.md)
- [23 — 앙상블](./volume_23_ensemble.md)
- [24 — SVM과 커널 방법](./volume_24_svm_kernel.md)
- [25 — 비지도학습](./volume_25_unsupervised.md)
- [26 — 차원 축소와 시각화](./volume_26_dimensionality_reduction.md)
- [27 — 특성 공학](./volume_27_feature_engineering.md)
- [28 — 베이지안 머신러닝](./volume_28_bayesian_ml.md)
- [29 — 하이퍼파라미터 탐색](./volume_29_hpo.md)

### Track D — 딥러닝 핵심

- [30 — 퍼셉트론과 신경망의 기원](./volume_30_perceptron.md)
- [31 — MLP와 유니버설 근사](./volume_31_mlp_universal_approx.md)
- [32 — 역전파와 자동미분](./volume_32_backprop_autograd.md)
- [33 — 옵티마이저](./volume_33_optimizers.md)
- [34 — 정규화 (Normalization)](./volume_34_normalization.md)
- [35 — 정칙화 (Regularization)](./volume_35_regularization.md)
- [36 — PyTorch 실전](./volume_36_pytorch_practice.md)
- [37 — 데이터 파이프라인과 실험 관리](./volume_37_data_pipeline.md)
- [38 — 모델 평가와 메트릭](./volume_38_evaluation_metrics.md)
- [39 — 활성화 함수 깊이](./volume_39_activations.md)
- [40 — 학습률 스케줄 깊이](./volume_40_lr_schedules.md)
- [41 — 딥러닝 디버깅](./volume_41_dl_debugging.md)

### Track E — 컴퓨터 비전

- [42 — 합성곱 신경망(CNN)](./volume_42_cnn.md)
- [43 — CNN 모델 계보](./volume_43_cnn_lineage.md)
- [44 — 객체 탐지와 세그멘테이션](./volume_44_detection_segmentation.md)
- [45 — Vision Transformer (ViT)](./volume_45_vit.md)
- [46 — 데이터 증강 깊이](./volume_46_augmentation.md)
- [47 — 자가지도 비전](./volume_47_ssl_vision.md)
- [48 — 멀티모달 비전-언어](./volume_48_multimodal_vl.md)

### Track F — 시퀀스와 트랜스포머

- [49 — RNN·LSTM·GRU와 그 한계](./volume_49_rnn_lstm.md)
- [50 — Attention 메커니즘](./volume_50_attention.md)
- [51 — Transformer 완전 정복](./volume_51_transformer.md)
- [52 — Long Context 기법](./volume_52_long_context.md)
- [53 — Mixture of Experts (MoE)](./volume_53_moe.md)
- [54 — 효율적 어텐션 변형](./volume_54_efficient_attention.md)

### Track G — 임베딩과 벡터 검색

- [55 — 단어 임베딩](./volume_55_word_embedding.md)
- [56 — 문장·문서 임베딩](./volume_56_sentence_embedding.md)
- [57 — 벡터 검색과 ANN](./volume_57_vector_search.md)
- [58 — 멀티모달 임베딩](./volume_58_multimodal_embedding.md)
- [59 — 임베딩 평가와 벤치마크](./volume_59_embedding_eval.md)
- [60 — Reranker 깊이](./volume_60_reranker.md)
- [61 — 임베딩 모델 직접 학습](./volume_61_embedding_training.md)

### Track H — 대규모 언어 모델 (LLM)

- [62 — 토크나이저](./volume_62_tokenizer.md)
- [63 — LLM 사전학습과 스케일링 법칙](./volume_63_llm_pretraining.md)
- [64 — LLM 정렬](./volume_64_llm_alignment.md)
- [65 — 프롬프트와 In-Context Learning](./volume_65_prompting.md)
- [66 — RAG — 검색 증강 생성](./volume_66_rag.md)
- [67 — LLM 디코딩 알고리즘](./volume_67_decoding.md)
- [68 — 모델 사이즈 의사결정](./volume_68_model_sizing.md)
- [69 — Constrained Generation](./volume_69_constrained_gen.md)
- [70 — 방어적 프롬프트 엔지니어링](./volume_70_defensive_prompting.md)
- [71 — LLM-as-Judge 평가 자동화](./volume_71_llm_as_judge.md)

### Track I — AI Agent

- [72 — 에이전트의 본질](./volume_72_agent_essence.md)
- [73 — 에이전트 프레임워크](./volume_73_agent_frameworks.md)
- [74 — 멀티에이전트와 워크플로](./volume_74_multi_agent.md)
- [75 — Agent Memory 시스템](./volume_75_agent_memory.md)
- [76 — Agent 평가와 벤치마크](./volume_76_agent_eval.md)
- [77 — Computer Use·Browser Agent](./volume_77_computer_use.md)

### Track J — 생성 모델

- [78 — GAN과 적대적 학습](./volume_78_gan.md)
- [79 — VAE와 잠재 변수 모델](./volume_79_vae.md)
- [80 — Image Diffusion](./volume_80_image_diffusion.md)
- [81 — 생성 제어 기법](./volume_81_generation_control.md)
- [82 — Image-to-Image·Inpainting·Outpainting](./volume_82_img2img.md)
- [83 — 비디오·3D 생성](./volume_83_video_3d.md)

### Track K — GPU와 추론 가속

- [84 — GPU 아키텍처와 CUDA](./volume_84_gpu_cuda.md)
- [85 — 추론 최적화](./volume_85_inference_optimization.md)
- [86 — 추론 서빙 시스템](./volume_86_serving_systems.md)
- [87 — KV 캐시 깊이](./volume_87_kv_cache.md)
- [88 — Speculative Decoding](./volume_88_speculative.md)
- [89 — 분산 학습](./volume_89_distributed_training.md)
- [90 — AI 가속기 비교 — GPU·TPU·Trainium·Cerebras·Groq](./volume_90_accelerators.md)
- [91 — CUDA Python 입문 — Numba·CuPy·Triton](./volume_91_cuda_python.md)

### Track L — 운영과 책임

- [92 — AI 시스템의 운영·관측·비용·거버넌스](./volume_92_ops_governance.md)
- [93 — 모델 모니터링 깊이](./volume_93_monitoring_deep.md)
- [94 — AI 인시던트 사후 분석](./volume_94_postmortem.md)
- [95 — 데이터 거버넌스·라벨링](./volume_95_data_governance.md)
- [96 — AI 시스템 아키텍처 종합](./volume_96_system_architecture.md)
- [97 — 사용자 피드백 시스템](./volume_97_user_feedback.md)

### Track M — 미세조정과 적응

- [98 — PEFT — LoRA·QLoRA·Adapter·Prompt Tuning](./volume_98_peft.md)
- [99 — 모델 병합과 멀티태스크](./volume_99_model_merging.md)

### Track N — 데이터·NLP 응용

- [100 — 데이터셋 엔지니어링 종합](./volume_100_dataset_engineering.md)
- [101 — NER·정보 추출](./volume_101_ner.md)
- [102 — 텍스트 분류 4 접근 비교](./volume_102_text_classification_compare.md)
- [103 — 텍스트 클러스터링·토픽 모델링](./volume_103_text_clustering.md)

### Track O — 실습 워크북

- [104 — 데이터셋 종합 실습 워크북](./volume_104_dataset_workbook.md)

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

모든 104권을 순서대로 학습하고, 각 권의 자가점검 키워드와 코드를 모두 손으로 작성합니다.

`Volume 1 → 2 → 3 → ... → 104`

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
