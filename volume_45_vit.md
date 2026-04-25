# Volume 45 — Vision Transformer (ViT)

> 이 권이 끝나면 이미지에 트랜스포머를 적용한다는 발상이 왜 자연스러우면서도 혁신적이었는지 설명할 수 있게 됩니다.

## 목적

2017 년 Transformer 가 NLP 를 정복한 뒤, *이미지에도 적용 가능한가* 라는 질문이 자연스럽게 등장했습니다. 2020 년 Vision Transformer (ViT) 가 *이미지를 패치로 자르고 트랜스포머에 입력* 하는 단순한 발상으로 CNN 을 능가했습니다. 이 권은 ViT 의 구조와 영향을 다집니다.

## 선수 지식

- Volume 42 완료, Volume 50 (Attention) 미리 보기 권장

## 학습 결과

1. ViT 의 *이미지 → 패치 → 토큰* 변환을 그릴 수 있습니다.
2. CNN 과 ViT 의 *귀납적 편향* 차이를 알 수 있습니다.
3. ViT 의 데이터 효율성 한계를 안다.
4. Swin Transformer·DeiT 같은 변형의 동기를 설명할 수 있습니다.

---

## 1. ViT 의 핵심 아이디어

이미지를 *고정 크기 패치 (예: 16×16)* 로 자르고, 각 패치를 *토큰* 으로 다룸. 트랜스포머는 토큰 시퀀스를 처리하는 일반 도구이므로 그대로 적용.

```
이미지 (224×224×3) → 14×14 = 196 개 패치 (16×16×3 = 768 차원 벡터)
                  → [CLS] + 196 토큰 시퀀스 → 트랜스포머 → 분류
```

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)
        return self.proj(x).flatten(2).transpose(1, 2)
```

이 단순한 변환이 ViT 의 출발점입니다.

---

## 2. CNN vs ViT 의 귀납적 편향

CNN 은 *지역성·평행 이동 불변성* 이 모델 구조에 내장. ViT 는 *그런 가정 없음*.

함의:
- *작은 데이터에서는 CNN 이 강함* (사전 가정이 도움)
- *큰 데이터에서는 ViT 가 더 잘 학습* (가정에 갇히지 않음)

ViT 가 *수억 장 이상의 데이터* 로 사전학습되어야 CNN 을 능가합니다.

---

## 3. DeiT — 데이터 효율 개선

ViT 는 데이터 굶주리는데, *Distillation* 으로 데이터 효율 개선. CNN 교사 모델을 사용.

---

## 4. Swin Transformer

*윈도우 단위 어텐션* + *계층적 구조* 로 ViT 의 약점 (큰 이미지의 O(n²) 어텐션) 을 보완.

---

## 5. 영향

ViT 의 등장으로:
- 비전·NLP 가 *같은 트랜스포머 구조* 사용 → 멀티모달 모델 (CLIP, LLaVA) 가능
- 비전 백본이 *CNN 에서 ViT 로* 점진 이동 (단, 여전히 CNN 이 많이 쓰임)
- ConvNeXt 같은 *CNN 의 ViT 영향* 모델 등장

---

## 권 정리

- ViT = 이미지를 패치로 자르고 트랜스포머에 입력
- CNN 의 귀납적 편향 없음 → 데이터 굶주림
- DeiT, Swin 으로 변형
- 멀티모달 시대를 가능하게 한 토대

가장 기억할 한 줄: **"ViT 는 이미지에 트랜스포머를 적용한 단순한 발상이지만, 비전과 NLP 의 통합을 가능하게 한 핵심 사건이다."**

다음 권: [Volume 46 — 데이터 증강 깊이](./volume_46_augmentation.md)

---

## 자가점검 키워드

`패치 임베딩`, `[CLS] 토큰`, `귀납적 편향`, `DeiT`, `Swin Transformer`, `멀티모달`

## 자가점검 질문

1. ViT 의 패치 임베딩 변환을 그림으로 설명하십시오.
2. CNN 과 ViT 의 귀납적 편향 차이를 적으십시오.
3. ViT 가 작은 데이터에서 약한 이유를 설명하십시오.
4. Swin Transformer 의 윈도우 어텐션 발상을 적으십시오.

## 다음 권

[Volume 46 — 데이터 증강 깊이](./volume_46_augmentation.md)
