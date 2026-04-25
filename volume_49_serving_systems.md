# Volume 49 — 추론 서빙 시스템

> 이 권이 끝나면 *내일 회사에 들어가 LLM 추론 서비스를 설계해야 한다*는 상황에 흔들리지 않을 수 있게 됩니다.

## 목적

추론 서빙은 *모델 + 큐 + 배칭 + 캐시 + 오토스케일 + 관측성*의 종합 시스템입니다. 백엔드 엔지니어의 기존 시스템 사고가 가장 자연스럽게 이어지는 영역이며, 동시에 *GPU 자원의 특수성·LLM 의 가변 길이 출력·Continuous Batching* 같은 새 사고를 요구합니다. 이 권은 vLLM·Triton Inference Server·TGI 같은 표준 도구를 사용 시점 기준으로 정리합니다.

## 선수 지식

- Volume 47, 48 완료
- 외부 지식: API 서버 운영 경험·Kubernetes 의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 추론 서비스의 표준 컴포넌트(게이트웨이·큐·워커·캐시·관측) 를 그릴 수 있습니다.
2. vLLM·Triton·TGI 의 사용 시점을 구분할 수 있습니다.
3. Continuous Batching 이 *왜 LLM 처리량을 폭발적으로 올리는가*를 설명할 수 있습니다.
4. GPU 오토스케일의 신호와 함정을 알 수 있습니다.
5. 추론 SLA(P50·P95·TTFT·TPOT) 를 정의·측정할 수 있습니다.

## 챕터 목차

1. **추론 서비스의 표준 아키텍처**
2. **vLLM** — PagedAttention + Continuous Batching
3. **Triton Inference Server** — 멀티 프레임워크 서빙
4. **TGI / SGLang / TensorRT-LLM Server**
5. **요청 큐와 우선순위**
6. **GPU 오토스케일** — 콜드스타트의 함정
7. **캐싱 계층** — 프롬프트·응답·임베딩
8. **SLA 와 측정 지표** — TTFT·TPOT·P95
9. **A/B 테스트·섀도우 트래픽**
10. **운영 사고 사례** — 메모리 폭발·OOM·디스크 부족

## 자가점검 키워드

`vLLM`, `PagedAttention`, `Triton`, `TGI`, `Continuous Batching`, `오토스케일`, `TTFT/TPOT`, `섀도우 트래픽`

## 다음 권

[Volume 50 — AI 시스템의 운영·관측·비용·거버넌스](./volume_50_ops_governance.md)
