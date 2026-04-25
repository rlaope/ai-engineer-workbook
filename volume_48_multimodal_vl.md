# Volume 85 — 멀티모달 비전-언어

> 이 권이 끝나면 CLIP·BLIP·LLaVA 같은 모델이 *이미지와 텍스트를 같은 공간으로 모으는* 발상을 그림으로 그릴 수 있게 됩니다.

## 목적

이미지와 텍스트를 함께 다루는 모델은 텍스트→이미지 검색, 이미지 캡셔닝, 시각적 질문응답(VQA), 멀티모달 챗봇의 기반입니다. 이 권은 CLIP 의 대조 학습으로 시작해, BLIP 의 생성 능력 추가, LLaVA 의 LLM 결합까지의 흐름을 다집니다. 멀티모달 임베딩 권(69) 의 직접 전제입니다.

## 선수 지식

- Volume 51, 33, 34, 64 완료
- 외부 지식: 검색·임베딩의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. CLIP 의 *이미지-텍스트 대조 학습* 손실을 적을 수 있습니다.
2. CLIP 임베딩으로 Zero-Shot 분류를 수행할 수 있습니다.
3. BLIP·BLIP-2 의 캡셔닝 능력 추가 방식을 설명할 수 있습니다.
4. LLaVA 가 비전 인코더와 LLM 을 결합하는 *Projector* 의 역할을 이해합니다.
5. 멀티모달 모델의 평가 어려움(VQA·캡셔닝 메트릭) 을 인식합니다.

## 챕터 목차

1. **멀티모달의 정의와 의미**
2. **CLIP** — Contrastive Language-Image Pretraining
3. **CLIP 의 응용** — Zero-Shot 분류·검색·생성 가이던스
4. **SigLIP** — Sigmoid 기반 변형
5. **BLIP / BLIP-2** — 캡셔닝과 생성
6. **LLaVA** — 비전 인코더 + LLM
7. **Flamingo·Qwen-VL·InternVL** — 멀티모달 LLM 계보
8. **멀티모달 평가** — VQA·CIDEr·CLIPScore

## 자가점검 키워드

`CLIP`, `Contrastive`, `Zero-Shot`, `SigLIP`, `BLIP`, `LLaVA`, `Projector`, `VQA`

## 다음 권

[Volume 66 — Long Context 기법](./volume_52_long_context.md)
