# Volume 94 — 모델 병합과 멀티태스크

> 이 권이 끝나면 *서로 다른 미세조정 모델을 합쳐 더 좋은 모델을 만든다* 는 발상의 원리와 한계를 알게 됩니다.

## 목적

학습은 비싸지만, 이미 학습된 모델은 많습니다. *Model Merging* 은 추가 학습 없이 가중치 평균만으로 두 모델의 능력을 결합하는 기법이며, Open LLM 리더보드 상위권의 다수가 이 기법으로 만들어집니다. SLERP·TIES·DARE·Task Arithmetic 같은 알고리즘이 표준이며, MergeKit 같은 도구로 손쉽게 적용할 수 있습니다.

## 선수 지식

- Volume 38, 88 완료
- 외부 지식: 가중평균의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. *왜 가중치 평균이 작동하는가* 를 손실 풍경(loss landscape) 관점에서 설명할 수 있습니다.
2. Model Soup·SLERP·TIES·DARE 의 차이를 알 수 있습니다.
3. Task Arithmetic 의 *능력의 산술* 발상을 그릴 수 있습니다.
4. MergeKit 으로 모델 병합 실험을 수행할 수 있습니다.
5. 멀티태스크 미세조정 vs 모델 병합의 트레이드오프를 비교할 수 있습니다.

## 챕터 목차

1. **모델 병합의 가능성과 동기**
2. **Linear Interpolation·Model Soup**
3. **SLERP — 구면 보간**
4. **TIES Merging** — 부호 정렬 + 절단
5. **DARE** — 무작위 드롭 + 스케일링
6. **Task Arithmetic** — `model + (skill_A) - (skill_B)`
7. **MergeKit 실습 패턴**
8. **멀티태스크 미세조정과의 비교**

## 자가점검 키워드

`Model Soup`, `SLERP`, `TIES`, `DARE`, `Task Arithmetic`, `MergeKit`, `손실 풍경`, `멀티태스크`

## 다음 권

[Volume 95 — 텍스트 분류 4 접근 비교](./volume_95_text_classification_compare.md)
