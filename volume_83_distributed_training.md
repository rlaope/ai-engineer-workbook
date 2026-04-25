# Volume 83 — 분산 학습

> 이 권이 끝나면 *왜 70B 모델 학습에 GPU 한 장이 아니라 수천 장이 필요한가* 와 *그 수천 장이 어떻게 협업하는가* 를 그림으로 그릴 수 있게 됩니다.

## 목적

대형 모델은 한 GPU 메모리에 들어가지 않습니다. 따라서 *데이터·모델 자체·옵티마이저 상태*를 여러 GPU 에 분산해야 하며, 이것이 데이터 병렬·텐서 병렬·파이프라인 병렬·시퀀스 병렬·ZeRO 의 세계입니다. 이 권은 분산 학습의 표준 패턴을 정리하고, PyTorch FSDP·DeepSpeed·Megatron-LM 의 사용 시점을 다룹니다.

## 선수 지식

- Volume 23, 47, 48 완료
- 외부 지식: 분산 시스템의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. DDP·FSDP·ZeRO 의 차이를 메모리 관점에서 설명할 수 있습니다.
2. 텐서 병렬과 파이프라인 병렬의 통신 패턴을 그릴 수 있습니다.
3. 3D 병렬(DP × TP × PP) 의 결합 발상을 이해합니다.
4. NCCL·AllReduce·AllGather 의 의미를 알 수 있습니다.
5. Activation Checkpointing 의 메모리 절감 원리를 설명할 수 있습니다.

## 챕터 목차

1. **분산 학습의 동기** — 메모리·계산·시간
2. **데이터 병렬 (DDP)**
3. **FSDP** — 모델·옵티마이저 상태 분산
4. **ZeRO Stage 1·2·3**
5. **텐서 병렬 (TP)**
6. **파이프라인 병렬 (PP)**
7. **시퀀스 병렬·Context 병렬**
8. **3D 병렬** — DP × TP × PP
9. **Activation Checkpointing 과 Offloading**
10. **NCCL 통신 원시연산** — AllReduce·AllGather·ReduceScatter

## 자가점검 키워드

`DDP`, `FSDP`, `ZeRO`, `TP`, `PP`, `3D Parallelism`, `Activation Checkpointing`, `NCCL`

## 다음 권

[Volume 84 — 모델 모니터링 깊이](./volume_84_monitoring_deep.md)
