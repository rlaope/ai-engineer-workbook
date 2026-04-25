# Volume 104 — NER·정보 추출

> 이 권이 끝나면 *비정형 텍스트에서 사람·장소·금액·날짜 같은 구조화 정보를 추출하는* 시스템을 설계할 수 있게 됩니다.

## 목적

NER(Named Entity Recognition) 과 정보 추출은 *문서를 검색 가능한 데이터로 바꾸는* 가장 기본적인 NLP 작업입니다. 의료·법률·금융·뉴스 분야에서 LLM 시대 이전부터 가장 큰 산업 응용 중 하나였고, 지금도 *정확도·일관성·낮은 비용*이 필요한 영역에서 BERT 계열 모델이 LLM 보다 강합니다. 이 권은 BIO 태깅·CRF·BERT-NER·LLM-기반 추출의 흐름을 정리합니다.

## 선수 지식

- Volume 22, 32, 36 완료
- 외부 지식: 정규식·문자열 처리

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. BIO·BIOES 태깅 체계를 그릴 수 있습니다.
2. CRF 가 *시퀀스 라벨링*에서 하는 역할을 알 수 있습니다.
3. BERT-NER 의 토큰 분류 헤드 구조를 설명할 수 있습니다.
4. LLM 기반 정보 추출(Function Calling + JSON 강제) 을 설계할 수 있습니다.
5. NER 평가 지표(Entity-level F1) 를 손으로 계산할 수 있습니다.

## 챕터 목차

1. **NER 의 정의와 활용**
2. **BIO·BIOES 태깅 체계**
3. **HMM·CRF 시대**
4. **BERT-NER** — 토큰 분류 + 후처리
5. **Span-based NER** — SpanBERT
6. **LLM 기반 정보 추출** — Function Calling
7. **관계 추출(RE) 과 이벤트 추출**
8. **평가 — Entity F1·Span Match·Type 정확도**
9. **spaCy / Stanza / GLiNER 도구**

## 자가점검 키워드

`BIO`, `BIOES`, `CRF`, `BERT-NER`, `Span`, `Function Calling 추출`, `Entity F1`, `spaCy`

## 다음 권

[Volume 99 — 모델 병합과 멀티태스크](./volume_99_model_merging.md)
