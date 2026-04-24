# Volume 21 — 정규화 (Normalization)

> 이 권이 끝나면 BatchNorm·LayerNorm·RMSNorm 중 어떤 것을 언제 써야 하는지 한 문장으로 답할 수 있게 됩니다.

## 목적

신경망의 학습이 안정적으로 진행되려면 *각 층 입력의 분포가 너무 크게 흔들리지 않아야* 합니다. 정규화는 이 분포를 안정화하는 도구이며, 종류에 따라 *어느 축을 따라 정규화하는가*가 다릅니다. CNN 은 BatchNorm, Transformer 는 LayerNorm, 최근 LLM 은 RMSNorm 을 쓰는 데에는 모두 이유가 있습니다. 이 권은 그 이유를 다집니다.

## 선수 지식

- Volume 19, 20 완료
- 외부 지식: 평균·분산의 정의

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. BatchNorm·LayerNorm·GroupNorm·InstanceNorm·RMSNorm 의 정규화 축을 그림으로 그릴 수 있습니다.
2. 각 정규화가 *어떤 모델 종류에 적합한지* 짝지을 수 있습니다.
3. BatchNorm 의 학습/추론 모드 차이를 설명할 수 있습니다.
4. RMSNorm 이 *왜 LLM 에서 LayerNorm 을 대체했는지*를 비용 관점에서 설명할 수 있습니다.
5. 정규화가 학습 안정성·수렴 속도에 주는 영향을 실험으로 확인할 수 있습니다.

## 챕터 목차

1. **왜 정규화가 필요한가** — 내부 공변량 변화의 직관
2. **BatchNorm** — 미니배치 차원으로 정규화
3. **BatchNorm 의 학습/추론 차이** — 이동 평균의 사용
4. **LayerNorm** — 특성 차원으로 정규화
5. **GroupNorm·InstanceNorm** — 중간 형태들
6. **RMSNorm** — 평균을 빼지 않는 단순화
7. **정규화 ≠ 정칙화** — 두 개념의 차이
8. **PyTorch 로 비교 실험**

## 자가점검 키워드

`BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `학습/추론 모드`, `이동 평균`, `정규화 vs 정칙화`, `내부 공변량 변화`

## 다음 권

[Volume 22 — 정칙화 (Regularization)](./volume_22_regularization.md)
