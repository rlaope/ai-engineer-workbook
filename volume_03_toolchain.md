# Volume 3 — 도구 체인과 작업 환경

> 이 권이 끝나면 AI 엔지니어가 매일 사용하는 도구의 전체 지도를 머릿속에 그릴 수 있게 됩니다.

## 목적

새로운 분야를 시작할 때 가장 큰 진입장벽은 *어떤 도구를 어떤 순서로 익혀야 하는가*입니다. 이 권은 Python 생태계·딥러닝 프레임워크·하드웨어 가속 라이브러리·실험 추적 도구·모델 허브를 한 장의 지도 위에 펼쳐 보여 줍니다. 각 도구가 *언제 등장하고 어떤 문제를 푸는지*를 미리 파악해 두면, 이후의 권에서 새로운 도구가 등장할 때 위치를 잡기 쉽습니다.

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

## 챕터 목차

1. **Python 생태계의 지도** — NumPy·SciPy·pandas·scikit-learn
2. **딥러닝 프레임워크** — PyTorch·JAX·TensorFlow
3. **GPU 스택** — CUDA·cuDNN·NCCL·CUTLASS·Triton
4. **모델 허브** — Hugging Face Hub·`transformers`·`diffusers`
5. **데이터 도구** — `datasets`·DVC·LakeFS
6. **실험 추적** — Weights & Biases·MLflow·TensorBoard
7. **개발 환경** — Jupyter·VS Code·`python -i`·DevContainer
8. **클라우드 GPU** — Colab·Lambda Labs·RunPod·자체 GPU 서버
9. **버전 관리와 협업** — git·Git LFS·모델 버저닝

## 자가점검 키워드

`PyTorch`, `CUDA`, `cuDNN`, `Hugging Face`, `transformers`, `W&B`, `MLflow`, `Jupyter`

## 다음 권

[Volume 4 — 선형대수 1 — 벡터·행렬·내적·노름](./volume_04_linear_algebra_1.md)
