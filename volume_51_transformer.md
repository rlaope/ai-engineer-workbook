# Volume 4 — Transformer 완전 정복

> 이 권이 끝나면 빈 노트북에 100 줄 안쪽으로 미니 트랜스포머를 구현할 수 있게 됩니다.

## 목적

트랜스포머는 LLM·디퓨전·멀티모달의 공통 백본입니다. 이 한 권의 이해 정도가 이후 Volumes 33–46 의 진도를 결정합니다. 이 권은 *Self-Attention → Multi-Head → Position Encoding → Block 구조 → Encoder/Decoder*의 흐름을 하나도 빠짐없이 따라가고, 마지막에 PyTorch 로 직접 구현하면서 점검합니다.

## 선수 지식

- Volume 50 완료
- 외부 지식: 행렬 곱·소프트맥스

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Self-Attention 의 수식을 손으로 유도할 수 있습니다.
2. Multi-Head 가 *서로 다른 표현 공간을 동시에 보는 일*임을 설명할 수 있습니다.
3. Sinusoidal·Learned·RoPE 위치 인코딩의 차이를 그림으로 보일 수 있습니다.
4. Transformer Block 의 잔차·LayerNorm·FFN 의 순서를 정확히 적을 수 있습니다.
5. Encoder·Decoder·Encoder-Decoder 모델의 차이를 구분할 수 있습니다.

## 챕터 목차

1. **Self-Attention** — 같은 시퀀스 안에서의 어텐션
2. **Multi-Head Attention** — 병렬 헤드의 직관
3. **Position Encoding** — Sinusoidal·Learned
4. **RoPE** — 회전 위치 임베딩의 동기
5. **Transformer Block 구조** — Attention + FFN + 잔차 + LayerNorm
6. **Encoder vs Decoder** — BERT vs GPT 의 갈림길
7. **Encoder-Decoder** — T5·번역 모델
8. **마스크드 셀프 어텐션** — 미래를 보지 않는 메커니즘
9. **PyTorch 로 100 줄 미니 트랜스포머**
10. **트랜스포머 변형들** — Sparse·Linear·Performer

## 자가점검 키워드

`Self-Attention`, `Multi-Head`, `Sinusoidal`, `RoPE`, `Transformer Block`, `Encoder/Decoder`, `Causal Mask`, `FFN`

## 다음 권

[Volume 33 — 단어 임베딩](./volume_55_word_embedding.md)
