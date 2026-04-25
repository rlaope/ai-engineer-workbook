# Volume 84 — 자가지도 비전

> 이 권이 끝나면 *라벨이 없는 이미지로도 강력한 표현을 학습할 수 있다* 는 패러다임을 직관적으로 이해하게 됩니다.

## 목적

자가지도 학습은 *데이터 자체에서 정답을 만드는* 학습입니다. 라벨링 비용 없이 거대한 이미지 풀에서 사전학습한 표현은 다운스트림 작업에서 적은 라벨로도 좋은 성능을 냅니다. SimCLR·MoCo·DINO·MAE 의 발상은 모두 *이미지를 변형해 두 뷰가 비슷해야 한다* 는 단순한 원칙에서 출발합니다. 이 권은 그 흐름을 정리합니다.

## 선수 지식

- Volume 42, 27, 29 완료
- 외부 지식: 대조의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 자가지도 학습의 *프리텍스트 작업* 정의를 설명할 수 있습니다.
2. 대조 학습(Contrastive Learning) 의 손실(InfoNCE) 을 적을 수 있습니다.
3. SimCLR·MoCo 의 차이를 그릴 수 있습니다.
4. DINO·DINOv2 의 자기 증류(Self-Distillation) 발상을 이해합니다.
5. MAE 의 *마스킹 + 재구성* 발상을 설명할 수 있습니다.

## 챕터 목차

1. **자가지도 학습의 정의와 동기**
2. **프리텍스트 작업** — 회전 예측·직소·컬러화
3. **대조 학습 일반론** — Anchor·Positive·Negative
4. **SimCLR** — 강한 증강 + InfoNCE
5. **MoCo** — 메모리 큐와 모멘텀 인코더
6. **BYOL·SimSiam** — Negative 없는 학습
7. **DINO·DINOv2** — Vision Transformer + Self-Distillation
8. **MAE** — Masked Autoencoder

## 자가점검 키워드

`프리텍스트`, `Contrastive`, `InfoNCE`, `SimCLR`, `MoCo`, `BYOL`, `DINO`, `MAE`

## 다음 권

[Volume 65 — 멀티모달 비전-언어](./volume_48_multimodal_vl.md)
