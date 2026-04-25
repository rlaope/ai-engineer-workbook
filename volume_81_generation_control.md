# Volume 87 — 생성 제어 기법

> 이 권이 끝나면 *원하는 구성·자세·스타일·아이덴티티를 가진 이미지* 를 생성하는 도구상자를 갖게 됩니다.

## 목적

텍스트 프롬프트만으로는 *정확한 자세·정확한 인물·정확한 구성*을 만들 수 없습니다. ControlNet·LoRA·Adapter·IP-Adapter·Reference Net 같은 기법은 *추가 조건*을 모델에 주입해 생성을 정밀하게 제어합니다. 이 권은 생성 제어의 일반론과 패턴을 다룹니다.

## 선수 지식

- Volume 80 완료
- 외부 지식: 이미지 편집의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. ControlNet 의 *추가 조건 인코딩 + 잔차 주입* 발상을 그릴 수 있습니다.
2. LoRA 가 생성 모델에서 *스타일·아이덴티티 미세조정*에 쓰이는 방식을 알 수 있습니다.
3. IP-Adapter 의 *디커플드 크로스-어텐션* 발상을 설명할 수 있습니다.
4. 여러 제어 신호를 *동시에* 적용하는 방법을 알 수 있습니다.
5. 제어 강도(가중치·스케일) 의 효과를 그릴 수 있습니다.

## 챕터 목차

1. **생성 제어의 일반 형태** — Conditioning
2. **ControlNet** — 잔차 주입식 조건화
3. **T2I-Adapter** — 더 가벼운 변형
4. **LoRA in Generation** — 스타일·아이덴티티
5. **IP-Adapter** — 이미지 프롬프트
6. **Reference Net·InstantID** — 인물 일관성
7. **다중 조건 결합**
8. **제어 강도와 가중치 튜닝**

## 자가점검 키워드

`Conditioning`, `ControlNet`, `T2I-Adapter`, `LoRA`, `IP-Adapter`, `InstantID`, `다중 조건`, `제어 강도`

## 다음 권

[Volume 79 — Image-to-Image·Inpainting·Outpainting](./volume_82_img2img.md)
