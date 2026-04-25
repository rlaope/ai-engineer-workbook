# Volume 80 — 데이터 증강 깊이

> 이 권이 끝나면 *데이터를 더 모을 수 없을 때 무엇을 시도할지* 의 표준 카드 묶음을 갖게 됩니다.

## 목적

데이터 증강은 *학습 데이터를 인위적으로 늘리는* 도구이며, 비전·음성·NLP 모두에서 강력한 정칙화 효과를 가집니다. 단순 뒤집기·잘라내기에서 시작해 MixUp·CutMix·RandAugment 같은 *학습 가능한 증강*까지 발전했습니다. 이 권은 도메인별 표준 증강과 *어떤 증강이 어떤 상황에 맞는가*를 다집니다.

## 선수 지식

- Volume 35, 26 완료
- 외부 지식: 이미지 처리의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 비전·NLP·오디오의 표준 증강 기법을 도메인별로 나열할 수 있습니다.
2. MixUp·CutMix·CutOut 의 차이를 그릴 수 있습니다.
3. RandAugment·AutoAugment·TrivialAugment 의 자동화 발상을 알 수 있습니다.
4. Albumentations 로 비전 증강 파이프라인을 구성할 수 있습니다.
5. *증강이 망치는 경우* 를 식별할 수 있습니다 (도메인 일치성 위반).

## 챕터 목차

1. **데이터 증강의 동기** — 일반화·정칙화
2. **비전 기본** — Flip·Crop·Rotate·ColorJitter
3. **CutOut·MixUp·CutMix·Mosaic**
4. **자동 증강** — AutoAugment·RandAugment·TrivialAugment
5. **Albumentations 파이프라인**
6. **NLP 증강** — Back-Translation·EDA·Synonym
7. **오디오 증강** — Time Shift·Pitch·SpecAugment
8. **증강이 해로운 경우** — 도메인 일치성

## 자가점검 키워드

`Flip/Crop`, `MixUp`, `CutMix`, `RandAugment`, `Albumentations`, `Back-Translation`, `SpecAugment`, `도메인 일치성`

## 다음 권

[Volume 64 — 자가지도 비전](./volume_47_ssl_vision.md)
