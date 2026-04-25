# Volume 58 — 멀티모달 임베딩

> 이 권이 끝나면 *텍스트로 이미지를 검색* 하거나 *오디오로 비디오를 검색* 하는 시스템의 임베딩 측면을 설계할 수 있게 됩니다.

## 목적

CLIP·SigLIP·ImageBind 같은 멀티모달 임베딩 모델은 *이미지·텍스트·오디오* 를 *같은 벡터 공간* 에 매핑합니다. 이 덕분에 *모달리티 간 검색·매칭* 이 가능해집니다.

## 선수 지식

- Volume 48, 56, 57 완료

## 학습 결과

1. CLIP 임베딩으로 *텍스트→이미지 검색* 을 만들 수 있습니다.
2. SigLIP·ImageBind 의 차별점을 알 수 있습니다.
3. *교차 모달 정렬* 의 학습 방식을 이해합니다.
4. 멀티모달 검색 시스템을 설계할 수 있습니다.

---

## 1. CLIP 임베딩으로 검색

```python
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")

# 이미지 임베딩
images = [preprocess(Image.open(f)) for f in ['cat.jpg', 'dog.jpg', 'car.jpg']]
images = torch.stack(images)
with torch.no_grad():
    img_emb = model.encode_image(images)
img_emb /= img_emb.norm(dim=-1, keepdim=True)

# 텍스트 쿼리 임베딩
text = clip.tokenize(["a cat lying on a bed"])
with torch.no_grad():
    txt_emb = model.encode_text(text)
txt_emb /= txt_emb.norm(dim=-1, keepdim=True)

# 검색
scores = img_emb @ txt_emb.T
top = scores.argmax()
print(f"Best match: image {top}")
```

이 30 줄이 *CLIP 기반 시멘틱 이미지 검색의 코어* 입니다.

---

## 2. SigLIP

CLIP 의 후계자. *Sigmoid 손실* 사용 (CLIP 의 contrastive 대신).

장점: *큰 배치 의존성 감소*, *학습 효율*. Google 발표 (2023).

---

## 3. ImageBind

Meta 의 모델 (2023). *6 가지 모달리티* (이미지·텍스트·오디오·깊이·열화상·IMU) 를 같은 공간에.

응용: *오디오로 이미지 검색*, *텍스트로 비디오 검색* 등 모달리티 자유 변환.

---

## 4. 학습 방식

### 4.1 대조 학습

같은 (이미지, 텍스트) 쌍은 *임베딩이 가깝게*, 다른 쌍은 *멀게*.

CLIP 은 4 억 개의 (이미지, 캡션) 쌍으로 학습.

### 4.2 결과

서로 다른 모달리티가 *같은 벡터 공간* 에 매핑되어 *교차 모달 비교* 가능.

---

## 5. 멀티모달 검색 시스템 설계

```
사용자 입력 (텍스트) → CLIP 텍스트 인코더 → 임베딩
                                              ↓
                                          벡터 DB 검색
                                              ↓
                                       상위 K 이미지 반환
```

저장 시점:
- 모든 이미지를 CLIP 으로 *한 번 임베딩*
- 임베딩을 벡터 DB 에 저장

검색 시점:
- 텍스트를 CLIP 으로 임베딩
- 벡터 DB 에서 가장 가까운 K 개 검색

이 패턴이 *Pinterest·구글 이미지 검색·E-Commerce 시각 검색* 의 기본 구조.

---

## 권 정리

- CLIP = 이미지·텍스트 같은 공간 매핑
- SigLIP = sigmoid 손실, 효율 개선
- ImageBind = 6 모달리티 통합
- 학습 = 대조 학습 (4 억+ 쌍)
- 멀티모달 검색 시스템 = 인코더 + 벡터 DB

가장 기억할 한 줄: **"CLIP 임베딩 한 번이면 이미지를 텍스트로 검색하는 시스템을 30 줄로 만들 수 있다."**

다음 권: [Volume 59 — 임베딩 평가와 벤치마크](./volume_59_embedding_eval.md)

---

## 자가점검 키워드

`CLIP`, `SigLIP`, `ImageBind`, `대조 학습`, `교차 모달`

## 자가점검 질문

1. CLIP 으로 텍스트→이미지 검색하는 시스템의 흐름을 그리십시오.
2. SigLIP 이 CLIP 보다 개선한 점을 적으십시오.
3. ImageBind 의 6 모달리티를 적으십시오.

## 다음 권

[Volume 59 — 임베딩 평가와 벤치마크](./volume_59_embedding_eval.md)
