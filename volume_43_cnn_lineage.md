# Volume 74 — CNN 모델 계보

> 이 권이 끝나면 AlexNet 부터 EfficientNet 까지 CNN 의 진화 흐름을 한 장의 계보도로 그릴 수 있게 됩니다.

## 목적

ResNet 의 잔차 연결, VGG 의 단순함, Inception 의 멀티 스케일, EfficientNet 의 복합 스케일링은 모두 *현재 트랜스포머에도 그대로 살아남은 사고*입니다. 모델 계보를 따라가면 *왜 깊어지면 학습이 어려워지는가*, *왜 잔차 연결이 폭발적으로 채택되었는가* 같은 질문에 답할 수 있게 되며, 이는 비전·언어·디퓨전 모두에 적용됩니다.

## 선수 지식

- Volume 42 완료
- 외부 지식: 시간순 정리에 대한 인내심

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. AlexNet → VGG → ResNet → EfficientNet 의 핵심 차별점을 한 줄씩 설명할 수 있습니다.
2. 잔차 연결이 그래디언트 흐름에 미치는 영향을 그릴 수 있습니다.
3. Inception 의 멀티 스케일 발상의 동기를 설명할 수 있습니다.
4. 복합 스케일링(폭·깊이·해상도) 의 의미를 설명할 수 있습니다.
5. ImageNet 벤치마크의 역사를 짚을 수 있습니다.

## 챕터 목차

1. **AlexNet (2012)** — 딥러닝의 전환점
2. **VGG (2014)** — 단순한 깊은 네트워크
3. **GoogLeNet/Inception (2014)** — 멀티 스케일 모듈
4. **ResNet (2015)** — 잔차 연결의 발견
5. **DenseNet (2017)** — 모든 층을 연결
6. **MobileNet / SqueezeNet** — 경량 모델 계보
7. **EfficientNet (2019)** — 복합 스케일링
8. **ConvNeXt (2022)** — 트랜스포머에서 배워 다시 강해진 CNN

## 자가점검 키워드

`AlexNet`, `VGG`, `Inception`, `ResNet`, `잔차 연결`, `DenseNet`, `MobileNet`, `EfficientNet`

## 다음 권

[Volume 44 — 객체 탐지와 세그멘테이션](./volume_44_detection_segmentation.md)
