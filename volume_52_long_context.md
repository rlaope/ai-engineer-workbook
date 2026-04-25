# Volume 5 — Long Context 기법

> 이 권이 끝나면 *왜 트랜스포머가 긴 시퀀스에서 약한가* 와 *그 한계를 어떻게 우회하는가* 를 모두 설명할 수 있게 됩니다.

## 목적

표준 어텐션은 시퀀스 길이의 제곱에 비례하는 시간·메모리를 사용합니다. 1 만 토큰 이상의 컨텍스트를 다루려면 새로운 발상이 필요하며, Sliding Window·LongFormer·BigBird 같은 희소 어텐션, Mamba·RetNet 같은 SSM/RNN-like 구조, RAG 같은 외부화 전략이 모두 그 답입니다. 이 권은 Long Context 의 도구상자를 정리합니다.

## 선수 지식

- Volume 50, 32 완료
- 외부 지식: 시간 복잡도

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 표준 어텐션의 O(n²) 복잡도를 시간·메모리 양쪽으로 설명할 수 있습니다.
2. Sliding Window·LongFormer·BigBird 의 희소 패턴을 그릴 수 있습니다.
3. Mamba/SSM 이 *RNN 의 부활*인 이유를 알 수 있습니다.
4. Position Encoding 의 외삽 한계(RoPE·ALiBi) 를 설명할 수 있습니다.
5. *Lost in the Middle* 같은 Long Context 의 함정을 인식합니다.

## 챕터 목차

1. **O(n²) 의 본질** — 시간·메모리 폭발
2. **Sliding Window Attention**
3. **LongFormer·BigBird** — Sparse + Global 패턴
4. **Hierarchical Attention**
5. **State Space Models (SSM)** — S4·Mamba
6. **RetNet·RWKV** — RNN-like 의 부활
7. **Position Encoding 외삽** — RoPE·ALiBi·NTK Scaling
8. **Long Context 의 함정** — Lost in the Middle·정보 희석
9. **외부화 전략** — RAG vs Long Context

## 자가점검 키워드

`O(n²)`, `Sliding Window`, `LongFormer`, `Mamba`, `SSM`, `RoPE Scaling`, `Lost in the Middle`, `RAG`

## 다음 권

[Volume 67 — Mixture of Experts (MoE)](./volume_53_moe.md)
