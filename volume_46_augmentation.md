# Volume 46 — 데이터 증강 깊이

> 이 권이 끝나면 *데이터를 더 모을 수 없을 때 무엇을 시도할지* 의 표준 카드 묶음을 갖게 됩니다.

## 목적

데이터 증강은 *학습 데이터를 인위적으로 늘리는* 도구이며, 비전·음성·NLP 모두에서 강력한 정칙화 효과를 가집니다. 단순 뒤집기·잘라내기에서 시작해 MixUp·CutMix·RandAugment 같은 *학습 가능한 증강* 까지 발전했습니다.

## 선수 지식

- Volume 35, 42 완료

## 학습 결과

1. 비전·NLP·오디오의 표준 증강 기법을 도메인별로 나열할 수 있습니다.
2. MixUp·CutMix·CutOut 의 차이를 그릴 수 있습니다.
3. RandAugment·AutoAugment·TrivialAugment 의 자동화 발상을 알 수 있습니다.
4. *증강이 망치는 경우* 를 식별할 수 있습니다.

---

## 1. 비전 기본 증강

```python
from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 2. CutOut·MixUp·CutMix

### 2.1 CutOut

이미지의 일부를 *0 으로 채움*. 단순하지만 효과적.

### 2.2 MixUp

두 샘플의 *픽셀 + 라벨 모두를 선형 결합*:

$$x' = \lambda x_1 + (1-\lambda) x_2$$
$$y' = \lambda y_1 + (1-\lambda) y_2$$

### 2.3 CutMix

이미지의 한 영역을 *다른 이미지로 교체*. 라벨도 비례로 결합.

---

## 3. 자동 증강

### 3.1 AutoAugment (2018)

*RL 로 최적 증강 정책* 학습. 큰 비용.

### 3.2 RandAugment (2019)

*N 개 변환을 무작위로 강도 M 으로 적용*. 단순한 2 파라미터.

### 3.3 TrivialAugment (2021)

*RandAugment 보다 더 단순*. 매번 1 개 변환 무작위 선택. 대부분의 경우 비슷한 성능.

```python
from torchvision.transforms import RandAugment, TrivialAugmentWide

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
])
```

---

## 4. NLP 증강

- **Back-Translation** — 다른 언어로 번역 후 다시 원본 언어로
- **EDA** — 동의어 교체·삽입·삭제·위치 교환
- **Paraphrasing** — LLM 으로 의미는 같고 표현만 다르게

---

## 5. 오디오 증강

- **Time Shift** — 시간축 이동
- **Pitch Shift** — 음높이 변경
- **SpecAugment** — 스펙트로그램에서 시간·주파수 마스킹

---

## 6. 증강이 해로운 경우

증강이 *도메인 특성을 위반* 하면 해로움:

- 의료 영상의 *수평 뒤집기* — 좌/우 장기 위치 의미 있음
- 텍스트 인식의 *수평 뒤집기* — 문자가 거꾸로
- 색깔 의미 있는 분류의 *ColorJitter* — 신호가 바뀜

도메인 지식으로 *허용 가능한 증강* 만 선택해야 합니다.

---

## 권 정리

- 비전 기본: Crop·Flip·Rotate·ColorJitter
- 결합 증강: CutOut·MixUp·CutMix
- 자동: AutoAugment → RandAugment → TrivialAugment (단순화)
- NLP: Back-Translation·EDA
- 오디오: Time/Pitch Shift·SpecAugment
- 도메인 일치성 위반 증강은 해로움

가장 기억할 한 줄: **"데이터를 더 모을 수 없을 때 가장 효과적인 정칙화는 증강이며, 도메인 일치성을 지키는 한도 안에서 적극 사용해야 한다."**

다음 권: [Volume 47 — 자가지도 비전](./volume_47_ssl_vision.md)

---

## 자가점검 키워드

`Crop/Flip/ColorJitter`, `CutOut/MixUp/CutMix`, `RandAugment`, `Back-Translation`, `SpecAugment`, `도메인 일치성`

## 자가점검 질문

1. MixUp·CutMix·CutOut 의 차이를 그림으로 설명하십시오.
2. AutoAugment 와 RandAugment 의 차이를 적으십시오.
3. 도메인 일치성을 위반하는 증강 사례 3 가지를 적으십시오.

## 다음 권

[Volume 47 — 자가지도 비전](./volume_47_ssl_vision.md)
