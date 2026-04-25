# Volume 99 — 임베딩 모델 직접 학습

> 이 권이 끝나면 자기 도메인에 특화된 임베딩 모델을 처음부터 학습·평가·서빙할 수 있게 됩니다.

## 목적

상용 임베딩으로 충분하지 않은 도메인(의료·법률·코드·내부 문서) 에서는 자체 임베딩 모델 학습이 필요합니다. 대조 학습 손실(InfoNCE), 트리플렛 마이닝, 하드 네거티브 마이닝, 학습률·배치 크기 튜닝이 핵심입니다. 이 권은 Sentence-Transformers 라이브러리로 직접 학습하는 워크플로를 다집니다.

## 선수 지식

- Volume 34, 64, 70 완료
- 외부 지식: 트리플렛·앵커-양성-음성 구조

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Bi-Encoder 의 *공유 가중치 + 풀링* 구조를 그릴 수 있습니다.
2. InfoNCE·MultipleNegatives·Triplet Loss 의 차이를 알 수 있습니다.
3. 하드 네거티브 마이닝의 동기와 방법을 적용할 수 있습니다.
4. 도메인 데이터로 임베딩 미세조정 스크립트를 작성할 수 있습니다.
5. 학습된 임베딩을 평가 셋(Vol 70) 으로 검증할 수 있습니다.

## 챕터 목차

1. **임베딩 모델 학습이 필요한 시점**
2. **Bi-Encoder 구조 복습**
3. **대조 손실** — InfoNCE·MultipleNegatives·Triplet
4. **하드 네거티브 마이닝**
5. **Sentence-Transformers 학습 스크립트**
6. **학습률·배치 크기·온도 튜닝**
7. **자체 평가 셋과 비교**
8. **압축·증류·서빙 연계**

## 자가점검 키워드

`Bi-Encoder`, `InfoNCE`, `MultipleNegatives`, `Triplet`, `Hard Negative`, `Sentence-Transformers`, `온도`, `Distill`

## 다음 권

[Volume 100 — CUDA Python 입문](./volume_100_cuda_python.md)
