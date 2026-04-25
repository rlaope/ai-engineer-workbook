# Volume 43 — CNN 모델 계보

> 이 권이 끝나면 AlexNet 부터 EfficientNet 까지 CNN 의 진화 흐름을 한 장의 계보도로 그릴 수 있게 됩니다.

## 목적

ResNet 의 잔차 연결, VGG 의 단순함, Inception 의 멀티 스케일, EfficientNet 의 복합 스케일링은 모두 *현재 트랜스포머에도 그대로 살아남은 사고* 입니다. 모델 계보를 따라가면 *왜 깊어지면 학습이 어려워지는가*, *왜 잔차 연결이 폭발적으로 채택되었는가* 같은 질문에 답할 수 있게 됩니다.

## 선수 지식

- Volume 42 완료

## 학습 결과

1. AlexNet → VGG → ResNet → EfficientNet 의 핵심 차별점을 한 줄씩 설명할 수 있습니다.
2. 잔차 연결이 그래디언트 흐름에 미치는 영향을 그릴 수 있습니다.
3. Inception 의 멀티 스케일 발상의 동기를 설명할 수 있습니다.
4. 복합 스케일링 (폭·깊이·해상도) 의 의미를 설명할 수 있습니다.

---

## 1. AlexNet (2012)

ImageNet 대회 우승. CNN 시대의 시작.

특징:
- 8 층 깊이 (그 시점 매우 깊음)
- ReLU 활성화 (시그모이드 대신)
- Dropout 정칙화
- GPU 학습 (GTX 580 두 장)

`[VERIFY: Krizhevsky et al. 2012]`

---

## 2. VGG (2014)

*단순함의 미학*. 모든 합성곱이 3×3, 모든 풀링이 2×2.

장점: *구조가 단순해 따라하기 쉬움*. 다른 모델의 *백본 (backbone)* 으로 자주 사용.
단점: 파라미터 매우 많음 (140M+).

---

## 3. GoogLeNet / Inception (2014)

*멀티 스케일 모듈* — 한 층에 *1×1, 3×3, 5×5 합성곱과 풀링* 을 모두 적용 후 결합.

발상: *어떤 크기의 필터가 좋은지 모르겠으니 다 써 보고 결합*.

또한 *1×1 합성곱* 을 차원 축소에 사용해 파라미터 감소.

---

## 4. ResNet (2015)

*잔차 연결 (Residual Connection)* 의 발견.

```
입력 x → 합성곱 블록 → F(x) → + → 출력 (= F(x) + x)
              ↑              ↑
              └──── 스킵 ────┘
```

이 단순한 연결이 *수백 층 신경망의 학습을 가능* 하게 만들었습니다. 그래디언트가 *스킵 경로* 로 직접 흐를 수 있어 그래디언트 소실이 완화됩니다.

ResNet-50, ResNet-101, ResNet-152 가 여전히 *백본의 표준* 으로 사용됩니다.

이 사고는 *트랜스포머의 잔차 연결* 로도 그대로 이어집니다.

---

## 5. DenseNet (2017)

ResNet 의 확장 — 모든 이전 층의 출력을 *현재 층에 연결*.

장점: *그래디언트 흐름 더 좋음*, *파라미터 효율*.
단점: 메모리 사용량 큼.

---

## 6. MobileNet (2017)

*Depthwise Separable Convolution* 으로 모바일 기기에서 동작. 정확도 약간 손해 + 파라미터 10 배 감소.

---

## 7. EfficientNet (2019)

*복합 스케일링 (Compound Scaling)*. 모델 크기를 키울 때 *깊이·폭·해상도를 균형 있게* 동시에 증가.

EfficientNet-B0 부터 B7 까지 같은 패턴으로 스케일.

---

## 8. ConvNeXt (2022)

*트랜스포머에서 배워 다시 강해진 CNN*. ResNet 에 ViT 의 아이디어 (큰 커널·LayerNorm·GELU 등) 적용.

성능: ViT 와 비슷, *합성곱의 이점 (연산 효율)* 유지.

---

## 권 정리

- AlexNet (2012): CNN 시대 시작, ReLU + Dropout + GPU
- VGG (2014): 단순함, 백본 표준
- Inception (2014): 멀티 스케일
- ResNet (2015): 잔차 연결, 깊은 망 가능
- DenseNet (2017): 모든 층 연결
- MobileNet (2017): 모바일용 경량
- EfficientNet (2019): 복합 스케일링
- ConvNeXt (2022): Transformer 영향 받은 CNN 부활

가장 기억할 한 줄: **"잔차 연결은 깊은 신경망 학습을 가능하게 만든 가장 단순하고 가장 강력한 발견이다."**

다음 권: [Volume 44 — 객체 탐지와 세그멘테이션](./volume_44_detection_segmentation.md)

---

## 자가점검 키워드

`AlexNet`, `VGG`, `Inception`, `ResNet`, `잔차 연결`, `DenseNet`, `MobileNet`, `EfficientNet`, `ConvNeXt`

## 자가점검 질문

1. CNN 모델 8 가지의 핵심 차별점을 한 줄씩 적으십시오.
2. 잔차 연결이 깊은 망 학습을 가능하게 만든 메커니즘을 설명하십시오.
3. EfficientNet 의 복합 스케일링 발상을 적으십시오.
4. 1×1 합성곱이 Inception 에서 사용된 이유를 설명하십시오.

## 다음 권

[Volume 44 — 객체 탐지와 세그멘테이션](./volume_44_detection_segmentation.md)
