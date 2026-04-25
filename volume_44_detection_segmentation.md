# Volume 44 — 객체 탐지와 세그멘테이션

> 이 권이 끝나면 *분류 ≠ 탐지 ≠ 세그멘테이션* 의 차이를 입력·출력 형태 관점에서 정확히 구분할 수 있게 됩니다.

## 목적

이미지에 무엇이 있는지뿐 아니라 *어디에 어떻게* 있는지를 다루는 작업입니다. 객체 탐지·세그멘테이션은 *입력은 이미지·출력 형태가 다르다* 는 차이만 있을 뿐 모델 골격은 비슷합니다.

## 선수 지식

- Volume 42, 43 완료

## 학습 결과

1. 분류·탐지·세그멘테이션의 출력 형태 차이를 안다.
2. YOLO·Faster R-CNN·Mask R-CNN·SAM 의 차별점을 알 수 있습니다.
3. IoU·mAP·Dice 같은 평가 지표를 적용할 수 있습니다.
4. *2-stage vs 1-stage* 탐지의 트레이드오프를 안다.

---

## 1. 세 작업의 출력 형태

```
+---------------+--------------------------------+
| 작업          | 출력                           |
+---------------+--------------------------------+
| 분류          | 클래스 라벨 (전체 이미지에 1 개) |
| 객체 탐지      | 바운딩 박스 + 클래스 (여러 개)   |
| 의미 분할      | 픽셀별 클래스 (전체 이미지)      |
| 인스턴스 분할  | 픽셀별 클래스 + 인스턴스 ID     |
+---------------+--------------------------------+
```

---

## 2. 객체 탐지

### 2.1 2-Stage — Faster R-CNN

1. *Region Proposal* — 객체가 있을 만한 영역 후보 제안
2. *Classification + 박스 회귀* — 각 후보 분류 + 박스 정제

장점: 정확도 높음. 단점: 느림.

### 2.2 1-Stage — YOLO

이미지를 *그리드로 나눠* 각 그리드 셀에서 *직접 박스 + 클래스 예측*.

장점: 빠름 (실시간 가능). 단점: 작은 객체 약함 (개선됨).

YOLOv1 (2016) → v8 (2023) 까지 빠른 진화.

```python
# YOLOv8 사용 예
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg')
results[0].show()
```

### 2.3 DETR — Transformer 기반

2020 년 등장. *NMS (Non-Maximum Suppression) 없이* 직접 박스 예측. 트랜스포머의 *집합 예측* 능력 활용.

---

## 3. 세그멘테이션

### 3.1 의미 분할 (Semantic Segmentation)

각 픽셀에 *클래스* 만 할당 (인스턴스 구분 없음).

대표 모델: U-Net (의료 영상 표준), DeepLabV3+, SegFormer.

### 3.2 인스턴스 분할 (Instance Segmentation)

같은 클래스라도 *서로 다른 인스턴스* 구분. *마스크 + 박스 + 클래스* 출력.

대표: Mask R-CNN.

### 3.3 SAM (Segment Anything Model, 2023)

Meta 의 *프롬프트 기반 분할* 모델. 점·박스·텍스트로 *임의의 객체 분할*. *zero-shot 일반화* 가능.

---

## 4. 평가 지표

- **IoU (Intersection over Union)** — 예측 박스와 정답 박스의 겹침 비율
- **mAP (mean Average Precision)** — 다양한 IoU 임계값에서의 평균 정확도
- **Dice Score** — 세그멘테이션의 표준
- **HD (Hausdorff Distance)** — 경계 정확도

---

## 권 정리

- 분류 → 탐지 → 의미 분할 → 인스턴스 분할 = 출력 정밀도 증가
- 2-stage = 정확·느림, 1-stage (YOLO) = 빠름
- DETR = 트랜스포머 기반 직접 예측
- U-Net = 의료 영상 표준, SAM = zero-shot 분할
- IoU·mAP·Dice = 표준 메트릭

가장 기억할 한 줄: **"이미지 작업은 출력 형태 (라벨·박스·픽셀 마스크) 에 따라 분류·탐지·세그멘테이션으로 나뉘며, 모델 골격은 비슷하다."**

다음 권: [Volume 45 — Vision Transformer (ViT)](./volume_45_vit.md)

---

## 자가점검 키워드

`분류/탐지/세그멘테이션`, `Faster R-CNN`, `YOLO`, `DETR`, `U-Net`, `Mask R-CNN`, `SAM`, `IoU/mAP/Dice`

## 자가점검 질문

1. 4 가지 이미지 작업의 출력 형태를 적으십시오.
2. 2-stage 와 1-stage 탐지의 트레이드오프를 비교하십시오.
3. SAM 이 가져온 새로운 것을 한 문단으로 설명하십시오.
4. IoU 와 Dice 의 차이를 적으십시오.

## 다음 권

[Volume 45 — Vision Transformer (ViT)](./volume_45_vit.md)
