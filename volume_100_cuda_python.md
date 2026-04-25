# Volume 100 — CUDA Python 입문 — Numba·CuPy·Triton

> 이 권이 끝나면 *Python 만 쓰면서도 CUDA 커널을 직접 작성*할 수 있게 됩니다.

## 목적

대부분의 AI 엔지니어는 PyTorch 가 호출하는 CUDA 를 *블랙박스*로 두지만, 성능 최적화·새로운 연산 구현·맞춤 fused kernel 이 필요할 때는 직접 CUDA 를 작성해야 합니다. C++ CUDA 를 배우는 것은 진입장벽이 크지만, Python 기반 도구(Numba·CuPy·Triton) 는 같은 사고를 훨씬 낮은 비용으로 접할 수 있게 합니다. 이 권은 Python 으로 GPU 커널을 작성하는 첫걸음을 다집니다.

## 선수 지식

- Volume 47 완료
- 외부 지식: NumPy·기본 병렬 사고

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. CUDA 의 그리드·블록·스레드 모델을 코드 수준에서 이해합니다.
2. CuPy 로 NumPy 코드를 GPU 로 옮길 수 있습니다.
3. Numba `@cuda.jit` 로 간단한 커널을 작성할 수 있습니다.
4. Triton 의 *블록 단위 프로그래밍 모델* 을 적용할 수 있습니다.
5. PyTorch Custom Op 으로 만든 커널을 통합할 수 있습니다.

## 챕터 목차

1. **CUDA 실행 모델 복습** (Vol 47 보강)
2. **CuPy** — NumPy 호환 GPU 배열
3. **Numba `@cuda.jit`** — 데코레이터 기반 커널
4. **공유 메모리(SMEM) 활용**
5. **Triton 입문** — 블록 단위 프로그래밍
6. **Triton 으로 GEMM·소프트맥스 작성**
7. **PyTorch Custom Op 통합**
8. **프로파일링** — Nsight Systems·Nsight Compute

## 자가점검 키워드

`그리드/블록/스레드`, `CuPy`, `Numba`, `Triton`, `SMEM`, `Custom Op`, `Nsight`, `GEMM`

## 다음 권

[Volume 101 — 데이터셋 종합 실습 워크북](./volume_101_dataset_workbook.md)
