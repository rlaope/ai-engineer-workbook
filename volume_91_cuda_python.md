# Volume 91 — CUDA Python 입문 — Numba·CuPy·Triton

> 이 권이 끝나면 *Python 만 쓰면서도 CUDA 커널을 직접 작성* 할 수 있게 됩니다.

## 목적

대부분의 AI 엔지니어는 PyTorch 가 호출하는 CUDA 를 *블랙박스* 로 두지만, 성능 최적화·새로운 연산 구현·맞춤 fused kernel 이 필요할 때는 직접 CUDA 를 작성해야 합니다. C++ CUDA 를 배우는 것은 진입장벽이 크지만, Python 기반 도구 (Numba·CuPy·Triton) 는 같은 사고를 훨씬 낮은 비용으로 접할 수 있게 합니다.

## 선수 지식

- Volume 84 완료

## 학습 결과

1. CUDA 의 그리드·블록·스레드 모델을 코드 수준에서 이해합니다.
2. CuPy 로 NumPy 코드를 GPU 로 옮길 수 있습니다.
3. Numba `@cuda.jit` 로 간단한 커널을 작성할 수 있습니다.
4. Triton 의 *블록 단위 프로그래밍 모델* 을 적용할 수 있습니다.

---

## 1. CuPy — NumPy 호환 GPU 배열

```python
import cupy as cp

# NumPy 와 거의 동일
a = cp.random.randn(1000, 1000)
b = cp.random.randn(1000, 1000)
c = a @ b   # GPU 에서 행렬 곱

# NumPy 변환
import numpy as np
np_array = cp.asnumpy(c)
gpu_array = cp.asarray(np_array)
```

NumPy 사용자가 *거의 변경 없이* GPU 활용 가능. 학습용·프로토타입에 좋음.

---

## 2. Numba `@cuda.jit`

데코레이터로 *Python 함수를 CUDA 커널로 컴파일*.

```python
from numba import cuda

@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + y[idx]

import numpy as np
n = 1_000_000
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.zeros_like(x)

threads = 256
blocks = (n + threads - 1) // threads
add_kernel[blocks, threads](x, y, out)
```

C++ CUDA 보다 단순. 학습 곡선 낮음.

---

## 3. Triton

OpenAI 의 *블록 단위 GPU 프로그래밍 언어*. PyTorch 2.0 의 `torch.compile` 도 내부적으로 사용.

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

장점: *블록 단위 사고* 로 직관적. 자동 최적화. C++ CUDA 와 비슷한 성능.

FlashAttention 같은 *고성능 커널이 Triton 으로 작성* 되는 경우 많음.

---

## 4. 도구 비교

```
+--------+--------+--------+--------+
| 도구   | 학습 곡선 | 성능   | 유연성 |
+--------+--------+--------+--------+
| CuPy   | 매우 낮음 | 보통  | 낮음 |
| Numba  | 낮음   | 보통    | 보통  |
| Triton | 중간   | 높음    | 높음  |
| C++ CUDA| 높음  | 최고    | 최고  |
+--------+--------+--------+--------+
```

학습 권장 순서: CuPy → Numba → Triton.

---

## 5. PyTorch Custom Op 통합

Triton 으로 만든 커널을 PyTorch 에 통합:

```python
import torch

class MyAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
        return out
```

이 방식으로 *PyTorch 에 새 연산 추가* 가능.

---

## 6. 프로파일링

- **Nsight Systems** — 시스템 수준 (커널 실행·메모리 전송)
- **Nsight Compute** — 개별 커널 깊이 분석
- **PyTorch Profiler** — 모델 내부 시간

이 도구로 *커널이 실제로 빠른지* 확인.

---

## 권 정리

- CuPy = NumPy 의 GPU 버전, 학습 권장
- Numba `@cuda.jit` = Python 데코레이터 커널
- Triton = 블록 단위·자동 최적화·FlashAttention 의 표준
- PyTorch Custom Op = 통합 인터페이스
- Nsight 로 프로파일링

가장 기억할 한 줄: **"Python 만으로도 CUDA 커널을 작성할 수 있고, Triton 이 현대 고성능 커널의 새 표준이 되어 가고 있다."**

다음 권: [Volume 92 — AI 시스템의 운영·관측·비용·거버넌스](./volume_92_ops_governance.md)

---

## 자가점검 키워드

`CuPy`, `Numba`, `Triton`, `블록 단위`, `Custom Op`, `Nsight`

## 자가점검 질문

1. CuPy·Numba·Triton 의 학습 곡선·성능을 표로 정리하십시오.
2. Triton 의 *블록 단위 프로그래밍* 발상을 설명하십시오.
3. PyTorch Custom Op 으로 Triton 커널을 통합하는 흐름을 적으십시오.

## 다음 권

[Volume 92 — AI 시스템의 운영·관측·비용·거버넌스](./volume_92_ops_governance.md)
