# Volume 62 — 토크나이저

> 이 권이 끝나면 *LLM 비용·컨텍스트 길이·언어별 효율성* 을 토크나이저 관점에서 추론할 수 있게 됩니다.

## 목적

토크나이저는 LLM 의 *입력층* 입니다. 텍스트를 *토큰 ID 시퀀스* 로 변환하며, *비용·컨텍스트 길이·다국어 효율* 모두에 직접 영향을 줍니다. 무시되기 쉽지만 *LLM 성능의 보이지 않는 결정자* 입니다.

## 선수 지식

- Volume 51 완료

## 학습 결과

1. BPE·WordPiece·SentencePiece 의 차이를 알 수 있습니다.
2. *영어 1 토큰 ≈ 한국어 2-3 토큰* 의 함의를 이해합니다.
3. tiktoken·Hugging Face 토크나이저를 사용할 수 있습니다.
4. 토크나이저의 *사라지는 함정 (예: 띄어쓰기·이모지)* 을 인식합니다.

---

## 1. 토큰의 실체

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

text = "Hello, world!"
ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(ids)
print(ids, tokens)
# [15496, 11, 995, 0]  ['Hello', ',', 'Ġworld', '!']
```

`Ġ` 은 *공백*. GPT-2 의 BPE 는 단어 앞 공백을 같은 토큰에 포함.

---

## 2. 알고리즘

### 2.1 BPE (Byte-Pair Encoding)

가장 흔한 알고리즘. 자주 등장하는 *문자 쌍* 을 점진적으로 *토큰으로 합침*.

```
"low"   "lower"   "newer"
↓
초기: l, o, w, e, r, n
빈번한 쌍 (e, r) → er
빈번한 쌍 (l, o) → lo
...
```

GPT 시리즈 (BPE), LLaMA (BPE 변형) 사용.

### 2.2 WordPiece

BERT 계열. BPE 와 비슷하지만 *최대 우도* 기반 합치기.

### 2.3 SentencePiece

언어 무관 (whitespace 도 토큰). 다국어 모델에서 표준.

LLaMA·T5 가 SentencePiece 사용.

---

## 3. 언어별 효율성

같은 의미의 텍스트도 *언어에 따라 토큰 수가 다릅니다*.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

en = "The quick brown fox jumps over the lazy dog."
ko = "갈색 여우가 게으른 개를 뛰어넘었다."
print(len(enc.encode(en)))   # ~10
print(len(enc.encode(ko)))   # ~25
```

한국어는 영어 대비 *2-3 배 토큰*. 함의:

- 같은 컨텍스트 윈도우에 *더 적은 정보* 들어감
- API 비용이 *2-3 배*
- 응답 속도 *2-3 배 느림*

이 격차는 토크나이저가 *영어 코퍼스로 주로 학습* 되어 한국어 단어가 잘 합쳐지지 않기 때문. 한국어 특화 LLM (HyperCLOVA·Mistral Korean 등) 은 한국어 토큰 효율이 좋음.

---

## 4. tiktoken 사용

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
text = "Tokenization is important."
tokens = enc.encode(text)
print(len(tokens), tokens)
# 5 [3404, 4024, 374, 3062, 13]

# 비용 추정
input_tokens = len(enc.encode(prompt))
output_tokens_estimate = 200
cost = input_tokens * 0.005/1000 + output_tokens_estimate * 0.015/1000
```

---

## 5. 함정

### 5.1 띄어쓰기·줄바꿈

`"hello"` 와 `" hello"` 는 *다른 토큰*. 프롬프트 작성 시 의식해야 함.

### 5.2 이모지·특수문자

이모지는 *여러 토큰* 에 걸쳐 표현. 한 이모지 = 2-4 토큰.

### 5.3 코드

들여쓰기·괄호·연산자가 *각자 토큰*. 코드는 자연어보다 토큰 효율 낮음.

### 5.4 토큰 경계 문제

LLM 이 *토큰 경계를 인식하지 못해* 글자 단위 작업 (예: 글자 수 세기) 에 약함.

---

## 권 정리

- BPE·WordPiece·SentencePiece = 표준 알고리즘
- 한국어 = 영어 대비 2-3 배 토큰 (비용·속도·컨텍스트 영향)
- tiktoken = 비용 사전 추정 도구
- 함정 = 띄어쓰기·이모지·코드·글자 단위

가장 기억할 한 줄: **"토크나이저는 LLM 의 보이지 않는 비용·속도 결정자이며, 한국어는 영어보다 2-3 배 비싸다."**

다음 권: [Volume 63 — LLM 사전학습과 스케일링 법칙](./volume_63_llm_pretraining.md)

---

## 자가점검 키워드

`BPE`, `WordPiece`, `SentencePiece`, `tiktoken`, `토큰 경계`, `다국어 효율`

## 자가점검 질문

1. BPE·WordPiece·SentencePiece 의 차이를 적으십시오.
2. 한국어가 영어 대비 토큰이 많은 이유와 실무적 영향을 설명하십시오.
3. tiktoken 으로 GPT-4 호출 비용을 사전 계산하는 코드를 적으십시오.
4. 토크나이저의 4 가지 함정을 나열하십시오.

## 다음 권

[Volume 63 — LLM 사전학습과 스케일링 법칙](./volume_63_llm_pretraining.md)
