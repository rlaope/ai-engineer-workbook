# Volume 90 — AI 가속기 비교 — GPU·TPU·Trainium·Cerebras·Groq

> 이 권이 끝나면 NVIDIA 외의 가속기 옵션이 어떤 워크로드에서 의미가 있는지 답할 수 있게 됩니다.

## 목적

NVIDIA GPU 가 *AI 가속기 시장의 90% 이상* 을 차지하지만, 점점 다양한 가속기가 등장합니다. TPU·AWS Trainium·Cerebras·Groq·Etched 같은 경쟁자의 강약을 다집니다.

## 선수 지식

- Volume 84 완료

## 학습 결과

1. NVIDIA·TPU·Trainium·Cerebras·Groq 의 강약을 비교할 수 있습니다.
2. 각 가속기가 *어떤 워크로드에 유리한가* 알 수 있습니다.
3. *벤더 종속* 의 위험을 인식합니다.

---

## 1. NVIDIA GPU

시장 표준. H100·H200·B100·B200·GB200.

장점: *생태계 (CUDA·cuDNN·TensorRT·vLLM 등)*, *모든 모델 호환*, *최대 시장 점유*.
단점: *비싸고 공급 부족*.

---

## 2. Google TPU

Google 의 자체 가속기. v4·v5e·v5p·v6 (Trillium).

장점: *대규모 학습에서 NVIDIA 와 경쟁력*. JAX·TensorFlow 와 통합.
단점: *Google Cloud 에서만*, *PyTorch 지원 제한적*, *생태계 좁음*.

Gemini·PaLM 같은 Google 모델이 TPU 로 학습.

---

## 3. AWS Trainium / Inferentia

AWS 의 자체 가속기.

- **Trainium** — 학습용
- **Inferentia** — 추론용

장점: *AWS 안에서 NVIDIA 보다 저렴*, AWS 인프라 통합.
단점: *AWS 종속*, *모델 호환성 제한*.

---

## 4. Cerebras WSE

전체 웨이퍼 (큰 칩) 한 장으로 구성된 *세계 최대 칩*. WSE-3 = 4 조 트랜지스터.

장점: *모델이 한 칩에* 들어가 분산 통신 불필요. 대규모 학습에 유리.
단점: *비싸고 시장 작음*, 생태계 좁음.

특수 프로젝트에서 사용.

---

## 5. Groq LPU

LLM 추론 *속도 SOTA*. NVIDIA H100 보다 *5-10 배 빠름* (특정 워크로드).

장점: *극단적으로 빠른 추론*. Llama 70B 가 *수백 토큰/초*.
단점: 학습 불가 (추론 전용), 메모리 작음.

대규모 LLM 추론·실시간 응답에 사용.

---

## 6. Etched·Tenstorrent·SambaNova

신생 가속기 회사들. 각자 다른 발상:

- **Etched** — Transformer 전용 ASIC
- **Tenstorrent** — 오픈소스 RISC-V 기반
- **SambaNova** — Reconfigurable Dataflow

대부분 *특수 워크로드 또는 연구* 단계.

---

## 7. 비교 표

```
+-----------+------+------+--------+--------+
| 가속기     | 학습 | 추론 | 생태계 | 가용성 |
+-----------+------+------+--------+--------+
| NVIDIA    | ✓✓✓ | ✓✓✓ | ✓✓✓   | ✓     |
| TPU       | ✓✓✓ | ✓✓  | ✓     | GCP만 |
| Trainium  | ✓✓  | ✓✓  | ✓     | AWS만 |
| Cerebras  | ✓✓✓ | ✓   | ✓     | 작음  |
| Groq      | ✗   | ✓✓✓ | ✓     | 작음  |
+-----------+------+------+--------+--------+
```

---

## 8. 선택 가이드

- *대부분의 시작* — NVIDIA (생태계·호환성)
- *Google Cloud 사용* — TPU (할인 가능)
- *AWS 사용 + 비용 중요* — Trainium/Inferentia
- *극단적 추론 속도 필요* — Groq
- *대형 모델 학습 + 분산 회피* — Cerebras

---

## 권 정리

- NVIDIA = 시장 표준, 생태계 압도
- TPU = Google Cloud + JAX
- Trainium/Inferentia = AWS + 비용
- Cerebras = 큰 칩, 분산 회피
- Groq = 추론 속도 SOTA
- 신생 가속기들 = 특수 워크로드

가장 기억할 한 줄: **"가속기 시장은 NVIDIA 가 압도하지만, 클라우드 종속 가속기 (TPU·Trainium) 와 특수 워크로드 가속기 (Groq·Cerebras) 가 점점 의미를 가진다."**

다음 권: [Volume 91 — CUDA Python 입문](./volume_91_cuda_python.md)

---

## 자가점검 키워드

`NVIDIA`, `TPU`, `Trainium`, `Cerebras`, `Groq`, `생태계`, `벤더 종속`

## 자가점검 질문

1. 5 가지 가속기의 강약을 표로 정리하십시오.
2. *극단적 추론 속도가 필요한* 워크로드에 적합한 가속기와 그 이유를 적으십시오.
3. 벤더 종속의 위험을 적으십시오.

## 다음 권

[Volume 91 — CUDA Python 입문](./volume_91_cuda_python.md)
