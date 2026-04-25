# Volume 86 — 추론 서빙 시스템

> 이 권이 끝나면 *내일 회사에 들어가 LLM 추론 서비스를 설계해야 한다* 는 상황에 흔들리지 않을 수 있게 됩니다.

## 목적

vLLM·Triton·TGI 같은 추론 서빙 프레임워크는 *동적 배칭·KV 캐시·오토스케일* 같은 표준 부품을 통합합니다. 이 권은 LLM 서빙의 표준 구조와 도구 비교를 다집니다.

## 선수 지식

- Volume 84, 85 완료

## 학습 결과

1. 표준 LLM 서빙 시스템의 구조를 그릴 수 있습니다.
2. vLLM·Triton·TGI 의 차이를 알 수 있습니다.
3. P95·TTFT·TPOT 같은 SLA 지표를 측정·튜닝합니다.
4. 오토스케일·라우팅·캐시의 통합 설계를 설명합니다.

---

## 1. LLM 서빙 시스템 구조

```
[클라이언트]
    ↓
[Load Balancer]
    ↓
[API Gateway / Router]
    ↓
[Cache (semantic/prompt)]   ← 캐시 적중 시 즉시 응답
    ↓ (캐시 미스)
[추론 서버 (vLLM/Triton)]
    ↓
[GPU 클러스터]
```

각 계층이 *별도 책임* 을 가지며, 독립적으로 확장 가능.

---

## 2. vLLM

### 2.1 강점

- PagedAttention (Vol 85)
- Continuous Batching
- FP8/INT8/AWQ 지원
- 다양한 모델 (LLaMA, Mistral, Qwen, etc)

### 2.2 사용

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

OpenAI 호환 API 자동 제공:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

산업 표준 LLM 서빙.

---

## 3. Triton Inference Server

NVIDIA 의 *범용 추론 서버*. LLM 외 비전·음성 등 모든 모델 지원.

장점: *다양한 백엔드* (PyTorch, TensorRT, ONNX, Python). *동적 배칭·모델 앙상블* 강력.

LLM 만 한다면 vLLM 이 더 단순. *멀티모델·다양한 워크로드* 에는 Triton.

---

## 4. TGI (Text Generation Inference)

Hugging Face 의 LLM 서빙. vLLM 과 비슷. HF 생태계 통합.

---

## 5. SLA 지표

### 5.1 핵심 지표

- **TTFT (Time To First Token)** — 첫 토큰까지 시간. 사용자 체감 응답 속도.
- **TPOT (Time Per Output Token)** — 토큰 사이 시간. 스트리밍 속도.
- **P50/P95/P99 지연** — 응답 시간 분포
- **처리량 (req/s 또는 tokens/s)** — 시스템 용량
- **오류율** — 5xx 비율

### 5.2 표준 목표

```
+----------+----------+--------+
| 지표     | 좋음     | 임계   |
+----------+----------+--------+
| TTFT     | <500ms   | <2s    |
| TPOT     | <50ms    | <100ms |
| P95      | <3s      | <10s   |
| 오류율   | <0.1%    | <1%    |
+----------+----------+--------+
```

이 목표를 *SLA* 로 합의하고 모니터링.

---

## 6. 오토스케일

### 6.1 신호

- *큐 깊이* — 대기 요청 수
- *GPU 활용률*
- *지연 (P95)*

### 6.2 설계

- *최소 인스턴스* (콜드스타트 방지)
- *최대 인스턴스* (비용 한계)
- *스케일 업/다운 임계값*
- *스케일 다운 지연* (요청이 다시 올 가능성)

콜드스타트 시 *모델 로드 시간* (수십 초 - 수 분) 이 큰 도전.

---

## 7. 캐시 통합

### 7.1 종류

- **응답 캐시** — 동일 프롬프트 → 동일 응답
- **시멘틱 캐시** — 의미적으로 비슷한 프롬프트 → 같은 응답
- **프롬프트 캐시** — 같은 시스템 프롬프트의 KV 캐시 재사용
- **임베딩 캐시** — 동일 텍스트 임베딩 재사용

OpenAI·Anthropic 의 *프롬프트 캐싱* API 가 표준화 중.

---

## 8. 멀티 모델 라우팅

같은 인프라에서 여러 모델 운영:

```python
def route(query):
    intent = classify(query)
    if intent == "code":
        return code_model.generate(query)
    elif intent == "chat":
        return chat_model.generate(query)
```

*LiteLLM·Portkey* 같은 AI Gateway 가 표준.

---

## 권 정리

- 표준 구조 = LB → Gateway → Cache → vLLM → GPU
- vLLM = LLM 서빙의 산업 표준
- Triton = 범용 추론 서버
- TGI = HF 생태계
- SLA = TTFT·TPOT·P95·처리량·오류율
- 오토스케일·캐시·라우팅의 통합 설계

가장 기억할 한 줄: **"LLM 서빙의 표준은 vLLM 위에 캐시·라우터·오토스케일을 얹은 시스템이며, 이 골격이 산업의 80% 를 커버한다."**

다음 권: [Volume 87 — KV 캐시 깊이](./volume_87_kv_cache.md)

---

## 자가점검 키워드

`vLLM`, `Triton`, `TGI`, `TTFT`, `TPOT`, `오토스케일`, `시멘틱 캐시`, `프롬프트 캐싱`

## 자가점검 질문

1. LLM 서빙 시스템의 표준 구조 5 계층을 그리십시오.
2. vLLM·Triton·TGI 의 적용 시점을 비교하십시오.
3. SLA 5 지표와 표준 목표를 적으십시오.
4. 캐시 4 종류를 나열하고 적용 시점을 적으십시오.

## 다음 권

[Volume 87 — KV 캐시 깊이](./volume_87_kv_cache.md)
