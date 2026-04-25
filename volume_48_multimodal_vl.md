# Volume 48 — 멀티모달 비전-언어

> 이 권이 끝나면 CLIP·BLIP·LLaVA 같은 모델이 *이미지와 텍스트를 같은 공간으로 모으는* 발상을 그림으로 그릴 수 있게 됩니다.

## 목적

현대 멀티모달 모델은 *이미지와 텍스트를 같은 임베딩 공간* 에 매핑하거나, *LLM 에 비전 입력을 추가* 하는 방식으로 만들어집니다. CLIP·BLIP·LLaVA·GPT-4V·Claude 4 Vision 같은 모델이 이 패러다임의 산물입니다.

## 선수 지식

- Volume 45, 47 완료

## 학습 결과

1. CLIP 의 *이미지-텍스트 대조 학습* 발상을 이해합니다.
2. BLIP·LLaVA 의 *비전-언어 결합* 패턴을 알 수 있습니다.
3. GPT-4V·Claude Vision 같은 LLM 의 비전 처리 구조를 그릴 수 있습니다.
4. 멀티모달 모델의 *Zero-shot 분류* 능력을 설명할 수 있습니다.

---

## 1. CLIP — 대조 학습으로 이미지·텍스트 정렬

### 1.1 발상

OpenAI 가 2021 년 발표. *4 억 개의 (이미지, 텍스트) 쌍* 으로 학습.

```
이미지 인코더 (ViT) → 이미지 임베딩
텍스트 인코더 (Transformer) → 텍스트 임베딩

같은 쌍은 임베딩이 가깝게, 다른 쌍은 멀게 (대조 학습)
```

### 1.2 Zero-shot 분류

분류기 학습 없이 *텍스트 프롬프트* 로 분류:

```python
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = clip.tokenize(["a dog", "a cat", "a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
sims = (image_features @ text_features.T).softmax(dim=-1)
print(sims)   # 'a dog' 가 가장 높음
```

### 1.3 영향

CLIP 임베딩은 *이미지 검색·생성 모델 가이드 (Stable Diffusion)·LLaVA 의 비전 인코더* 등 광범위하게 활용됩니다.

---

## 2. LLaVA 류 — LLM + 비전 인코더

### 2.1 구조

```
이미지 → ViT → 비전 토큰
                    ↓
                [Projector] (작은 MLP)
                    ↓
텍스트 → Tokenizer → 텍스트 토큰
                    ↓
[비전 토큰 + 텍스트 토큰] → LLM → 응답
```

핵심: *작은 Projector 만 학습* 해 비전 인코더의 출력을 LLM 의 토큰 공간으로 매핑.

### 2.2 결과

LLM 의 언어 능력 + 비전 인코더의 시각 능력 결합. *이미지에 대한 자연어 질의응답* 가능.

---

## 3. GPT-4V·Claude Vision

폐쇄형 모델. 자세한 구조는 비공개지만, 비슷한 패턴 추정.

응용: *문서 이해·차트 분석·스크린샷 분석·코드 스크린샷 OCR·UI 테스트* 등.

---

## 4. BLIP·BLIP-2

Salesforce 의 모델. *이미지 캡셔닝·VQA (Visual Question Answering)* 에 강점. Q-Former 라는 효율적 결합 모듈 사용.

---

## 권 정리

- CLIP = 이미지·텍스트 같은 공간 매핑 (대조 학습)
- LLaVA 류 = LLM + 비전 인코더 + Projector
- GPT-4V·Claude Vision = 폐쇄형 멀티모달 표준
- 멀티모달은 *Zero-shot 분류·VQA·문서 이해* 등 광범위 응용

가장 기억할 한 줄: **"멀티모달 모델은 이미지와 텍스트를 같은 임베딩 공간에 모으거나 LLM 에 비전을 결합한다."**

다음 권: [Volume 49 — RNN·LSTM·GRU 와 그 한계](./volume_49_rnn_lstm.md)

---

## 자가점검 키워드

`CLIP`, `대조 학습`, `Zero-shot`, `LLaVA`, `Projector`, `GPT-4V`, `BLIP`, `Q-Former`

## 자가점검 질문

1. CLIP 의 학습 목표를 한 문단으로 설명하십시오.
2. LLaVA 류 모델의 *Projector* 가 하는 역할을 적으십시오.
3. CLIP 의 Zero-shot 분류가 어떻게 가능한지 설명하십시오.
4. 멀티모달 모델의 산업 응용 3 가지를 적으십시오.

## 다음 권

[Volume 49 — RNN·LSTM·GRU 와 그 한계](./volume_49_rnn_lstm.md)
