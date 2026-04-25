# Volume 81 — 생성 제어 기법

> 이 권이 끝나면 *원하는 구성·자세·스타일·아이덴티티를 가진 이미지* 를 생성하는 도구상자를 갖게 됩니다.

## 목적

베이스 디퓨전 모델은 *프롬프트만으로* 이미지를 만들지만, 산업 응용에서는 *더 정교한 제어* 가 필요합니다. ControlNet·LoRA·IP-Adapter·DreamBooth 같은 기법이 *생성 제어* 의 표준 도구입니다.

## 선수 지식

- Volume 80 완료

## 학습 결과

1. ControlNet 의 *외부 조건 주입* 발상을 알 수 있습니다.
2. LoRA 가 *스타일·도메인 특화* 의 표준임을 안다.
3. IP-Adapter 의 *참조 이미지 가이드* 메커니즘을 이해합니다.
4. DreamBooth 와 Textual Inversion 의 *피사체 학습* 방식을 알 수 있습니다.

---

## 1. ControlNet

### 1.1 발상

기존 디퓨전 모델에 *외부 조건* (스케치·자세·깊이·세그멘테이션 마스크 등) 을 주입.

```
프롬프트 → U-Net (베이스)
                ↓
ControlNet (조건 인코더) → 잠재 공간 추가 신호
```

베이스 모델은 *동결*, ControlNet 만 학습.

### 1.2 표준 조건

- Canny Edge — 윤곽선
- HED — 더 부드러운 엣지
- OpenPose — 사람 자세
- Depth Map — 깊이
- Segmentation — 영역
- Normal Map — 표면 방향
- Scribble — 자유로운 스케치

### 1.3 효과

같은 프롬프트로도 *구성을 정확히 통제* 가능. *상품 사진의 구도 유지·캐릭터 포즈 고정* 같은 산업 응용에 핵심.

---

## 2. LoRA

### 2.1 발상

Vol 9 (선형대수 2), Vol 98 (PEFT) 와 같은 발상. 디퓨전 모델의 *어텐션 가중치에 작은 LoRA 어댑터* 학습.

### 2.2 사용

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("base-model")
pipe.load_lora_weights("./my-style-lora")
image = pipe("a portrait in <my-style>").images[0]
```

LoRA 어댑터는 *수 MB* 크기. 무한히 많이 보관 가능. 같은 베이스 모델 + 다양한 LoRA 로 *수많은 스타일* 가능.

Civitai 같은 사이트에 *수만 개 LoRA* 공유.

---

## 3. IP-Adapter

### 3.1 발상

*참조 이미지* 를 입력으로 받아, 그 *스타일·아이덴티티* 를 생성에 반영.

```
프롬프트 + [참조 이미지] → IP-Adapter → U-Net 에 추가 컨디셔닝
```

### 3.2 효과

- *참조 사람의 얼굴* 을 다른 자세·배경에 배치
- *참조 그림의 스타일* 을 다른 주제에 적용

학습 없이 *zero-shot 으로 작동*. 산업 응용 (가상 시착·아바타) 에 인기.

---

## 4. DreamBooth

### 4.1 발상

*특정 피사체 (사람·반려동물·물건) 의 3-10 장 사진* 으로 *베이스 모델 미세조정*. 이후 그 피사체를 다른 맥락에서 생성 가능.

```
3 장의 내 강아지 사진 + 미세조정
→ "내 강아지가 우주복을 입고 화성에 있는 모습"
```

### 4.2 단점

- 미세조정 비용 (한 피사체당 30 분 - 수 시간)
- 베이스 모델 전체 갱신 → 어댑터보다 무거움

---

## 5. Textual Inversion

DreamBooth 의 가벼운 변형. *새 토큰 (`<my-cat>`) 의 임베딩만 학습*. 모델 가중치는 동결.

장점: *매우 가벼움* (KB 단위). 단점: 표현력 제한.

---

## 6. 조합 사용

산업 응용에서는 여러 기법을 *조합*:

```
ControlNet (구도 통제) + LoRA (스타일) + IP-Adapter (피사체)
```

예: *내 캐릭터 (LoRA) 가 특정 자세 (ControlNet) 로 참조 이미지 스타일 (IP-Adapter) 로 생성*.

---

## 권 정리

- ControlNet = 외부 조건 (스케치·자세·깊이) 주입
- LoRA = 스타일·도메인 어댑터 (수 MB)
- IP-Adapter = 참조 이미지 가이드 (zero-shot)
- DreamBooth = 피사체 미세조정
- Textual Inversion = 토큰 임베딩만 학습
- 조합 사용이 산업 표준

가장 기억할 한 줄: **"디퓨전 생성 제어는 ControlNet·LoRA·IP-Adapter 의 조합으로 정밀해지며, 산업 응용은 거의 항상 조합을 사용한다."**

다음 권: [Volume 82 — Image-to-Image·Inpainting·Outpainting](./volume_82_img2img.md)

---

## 자가점검 키워드

`ControlNet`, `LoRA`, `IP-Adapter`, `DreamBooth`, `Textual Inversion`

## 자가점검 질문

1. ControlNet 의 표준 조건 5 가지를 적으십시오.
2. LoRA 와 DreamBooth 의 차이를 설명하십시오.
3. IP-Adapter 가 *zero-shot* 으로 작동하는 메커니즘을 적으십시오.
4. 조합 사용의 산업 응용 사례를 적으십시오.

## 다음 권

[Volume 82 — Image-to-Image·Inpainting·Outpainting](./volume_82_img2img.md)
