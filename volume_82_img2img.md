# Volume 82 — Image-to-Image·Inpainting·Outpainting

> 이 권이 끝나면 *기존 이미지를 변환·복원·확장* 하는 디퓨전 응용 패턴 전체를 설명할 수 있게 됩니다.

## 목적

디퓨전 모델은 *텍스트→이미지* 만이 아니라 *이미지→이미지* 변환에도 강력합니다. img2img·inpainting·outpainting 같은 패턴이 *상용 이미지 편집 앱* 의 핵심입니다.

## 선수 지식

- Volume 80 완료

## 학습 결과

1. img2img 의 *부분 노이즈 추가 + 디노이징* 발상을 알 수 있습니다.
2. Inpainting 의 *마스크 영역만 재생성* 메커니즘을 이해합니다.
3. Outpainting 으로 이미지를 *확장* 할 수 있습니다.
4. 산업 응용 (상품 사진 편집·배경 교체·복원) 을 안다.

---

## 1. Image-to-Image (img2img)

### 1.1 발상

원본 이미지에 *일부 노이즈만 추가* (T 의 일부 단계까지) → 그 시점부터 *역방향 디노이징*. 결과는 *원본의 구조를 유지하면서 변형* 된 이미지.

### 1.2 Strength 파라미터

`strength = 0` — 원본 그대로
`strength = 1` — 완전한 새 생성 (원본 무시)
`strength = 0.5` — 균형

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
new_image = pipe(
    prompt="anime style",
    image=original_image,
    strength=0.7,
).images[0]
```

응용: *사진→그림체 변환·아트 스타일 변환·색감 변경*.

---

## 2. Inpainting

### 2.1 발상

이미지의 *마스크 영역만 재생성*. 나머지는 보존.

```
원본 이미지 + 마스크 (영역 지정) + 프롬프트 → 마스크 영역만 새로 생성
```

### 2.2 사용

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(...)
result = pipe(
    prompt="a cat",
    image=original_image,
    mask_image=mask,   # 흰색 = 재생성, 검정 = 보존
).images[0]
```

응용:
- *얼굴 보정* (마스크 영역만 변경)
- *물건 제거* (마스크 → 배경으로 채움)
- *불완전한 이미지 복원*

---

## 3. Outpainting

### 3.1 발상

이미지의 *경계를 넘어 확장*. 원본 이미지를 *큰 캔버스의 한 부분* 으로 두고, *바깥 영역을 inpainting* 으로 생성.

```
원본:  [이미지]
확장:  [.....][이미지][.....]
                         ↑ AI 가 채움
```

응용: 인스타그램 정사각형을 *세로/가로 비율로 확장*. 영화 *aspect ratio 변경*.

---

## 4. SDXL 의 Refiner

SDXL 은 *2 단계 모델*: Base + Refiner.

- Base — 큰 구조 생성
- Refiner — 디테일 개선 (img2img 형태)

이 패턴이 *고품질 생성* 의 표준이 되어 가는 중.

---

## 5. 상품 사진 편집 사례

전자상거래에서 흔한 워크플로:

```
1. 모델이 옷을 입은 사진
2. Inpainting → 모델 얼굴 다른 사람으로
3. Outpainting → 가로/세로 비율 확장
4. img2img → 배경 교체 (예: 야외 → 스튜디오)
5. ControlNet (Vol 81) → 같은 자세로 다른 옷 입히기
```

전체가 *디퓨전 + 마스크 도구* 의 조합.

---

## 권 정리

- img2img = 원본 + 부분 노이즈 + 디노이징, strength 로 통제
- Inpainting = 마스크 영역만 재생성
- Outpainting = 경계 너머 확장
- SDXL Refiner = 2 단계 정제
- 상품 사진 편집이 가장 큰 산업 응용

가장 기억할 한 줄: **"디퓨전의 img2img·inpainting·outpainting 이 산업 이미지 편집의 표준 도구상자다."**

다음 권: [Volume 83 — 비디오·3D 생성](./volume_83_video_3d.md)

---

## 자가점검 키워드

`img2img`, `strength`, `inpainting`, `outpainting`, `mask`, `Refiner`

## 자가점검 질문

1. img2img 의 *strength* 가 결과에 미치는 영향을 설명하십시오.
2. Inpainting 의 마스크 형식을 적으십시오.
3. Outpainting 의 발상을 그리십시오.
4. 상품 사진 편집 워크플로 5 단계를 자기 도메인 예시로 그리십시오.

## 다음 권

[Volume 83 — 비디오·3D 생성](./volume_83_video_3d.md)
