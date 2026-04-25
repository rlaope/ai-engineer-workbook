# Volume 18 — 텍스트 클러스터링·토픽 모델링

> 이 권이 끝나면 *수만 건의 문서를 라벨 없이 의미 있는 그룹으로 묶고* 그 그룹의 주제를 자동으로 명명할 수 있게 됩니다.

## 목적

라벨이 없는 텍스트 더미에서 *어떤 주제들이 있는가* 를 자동으로 발견하는 일은 고객 피드백 분석·로그 분석·뉴스 큐레이션의 표준 작업입니다. LDA 같은 고전 기법부터, BERTopic·Top2Vec 같은 임베딩 + 클러스터링 + LLM 명명 결합까지가 이 영역의 도구입니다. 이 권은 텍스트 클러스터링 파이프라인을 정리합니다.

## 선수 지식

- Volume 25, 16, 34 완료
- 외부 지식: 군집화의 일반 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. LDA 의 *문서-주제·주제-단어* 두 분포 발상을 그릴 수 있습니다.
2. BERTopic 의 *임베딩 + UMAP + HDBSCAN + c-TF-IDF* 파이프라인을 설명할 수 있습니다.
3. 토픽 명명을 LLM 으로 자동화하는 방법을 알 수 있습니다.
4. 토픽 응집도(Coherence) 같은 평가 지표를 사용할 수 있습니다.
5. 토픽 변화 추적(Dynamic Topic Modeling) 의 기본 발상을 압니다.

## 챕터 목차

1. **텍스트 클러스터링의 두 단계** — 임베딩 + 군집화
2. **고전 — LDA·NMF**
3. **BERTopic 파이프라인** — Embed → UMAP → HDBSCAN → c-TF-IDF
4. **Top2Vec**
5. **토픽 명명 자동화 — LLM 활용**
6. **평가 — Coherence·Diversity·Outliers**
7. **Dynamic Topic Modeling** — 시간 변화 추적
8. **시각화 — Inter-topic Distance·DataMapPlot**

## 자가점검 키워드

`LDA`, `NMF`, `BERTopic`, `UMAP`, `HDBSCAN`, `c-TF-IDF`, `Coherence`, `Top2Vec`

## 다음 권

[Volume 97 — 사용자 피드백 시스템](./volume_97_user_feedback.md)
