# Volume 88 — PEFT — LoRA·QLoRA·Adapter·Prompt Tuning

> 이 권이 끝나면 *7B 모델을 24GB GPU 하나로 미세조정* 하는 방법을 코드 수준에서 설명할 수 있게 됩니다.

## 목적

전체 파라미터 미세조정(Full Fine-Tuning) 은 큰 모델에서는 메모리·연산 비용이 비현실적입니다. PEFT(Parameter-Efficient Fine-Tuning) 는 *전체의 0.1~1%* 만 학습하면서도 비슷한 성능을 얻는 도구입니다. LoRA·QLoRA·Adapter·Prompt Tuning·IA³ 가 표준이며, 산업 현장의 거의 모든 LLM 미세조정은 PEFT 로 이루어집니다. 이 권은 PEFT 의 모든 갈래와 `peft`/`trl` 라이브러리 사용을 다룹니다.

## 선수 지식

- Volume 19, 23, 38 완료
- 외부 지식: 메모리 사용량 계산 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. LoRA 가 *낮은 랭크 분해를 학습 가능 파라미터로 두는 일* 임을 그릴 수 있습니다.
2. QLoRA 가 *4-bit 양자화 + LoRA* 의 결합인 이유와 수식을 설명할 수 있습니다.
3. Adapter·Prompt Tuning·Prefix Tuning·IA³ 의 차이를 한 표로 정리할 수 있습니다.
4. `peft` 라이브러리로 LoRA 미세조정 스크립트를 작성할 수 있습니다.
5. PEFT 어댑터를 *합치기·교체·서빙* 하는 운영 패턴을 알 수 있습니다.

## 챕터 목차

1. **Full Fine-Tuning 의 메모리 비용**
2. **LoRA — Low-Rank Adaptation**
3. **랭크 r·알파·드롭아웃 튜닝**
4. **QLoRA** — 4-bit + LoRA
5. **Adapter** — 추가 작은 모듈
6. **Prompt / Prefix Tuning**
7. **IA³** — 활성화 스케일링
8. **`peft` + `trl` 라이브러리 사용 패턴**
9. **어댑터 합치기 / 교체 / 서빙**
10. **DoRA·LoRA-FA 등 최신 변형**

## 자가점검 키워드

`LoRA`, `QLoRA`, `Adapter`, `Prompt Tuning`, `Prefix Tuning`, `IA³`, `DoRA`, `peft`

## 다음 권

[Volume 89 — 데이터셋 엔지니어링 종합](./volume_89_dataset_engineering.md)
