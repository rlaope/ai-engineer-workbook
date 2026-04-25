# Volume 101 — NER·정보 추출

> 이 권이 끝나면 *비정형 텍스트에서 사람·장소·금액·날짜 같은 구조화 정보를 추출하는* 시스템을 설계할 수 있게 됩니다.

## 목적

NER (Named Entity Recognition) 와 정보 추출은 *비정형 텍스트에서 구조화 데이터* 를 만드는 일입니다. 산업 응용은 *문서 처리·뉴스 분석·고객 지원·의료 차트·법률 문서* 등 매우 광범위합니다.

## 선수 지식

- Volume 22, 65 완료

## 학습 결과

1. NER 의 표준 라벨 (PER·ORG·LOC·DATE·MONEY) 을 알 수 있습니다.
2. spaCy·Transformers 로 NER 모델을 사용할 수 있습니다.
3. LLM Few-shot 으로 정보 추출 시스템을 만듭니다.
4. Constrained Generation 으로 출력 형식 강제.

---

## 1. NER 의 표준 라벨

CoNLL-2003 표준:

- **PER** — 사람 (Alice, Bob)
- **ORG** — 조직 (Google, NASA)
- **LOC** — 장소 (Seoul, Tokyo)
- **MISC** — 기타 (제품·이벤트)

확장:
- **DATE** (2025-01-15)
- **MONEY** ($1,000)
- **TIME** (3:00 PM)
- **PERCENT** (5%)
- **EMAIL**, **URL**, **PHONE**

도메인 특화:
- **DRUG**, **DISEASE** (의료)
- **CASE_LAW** (법률)
- **PRODUCT** (E-Commerce)

---

## 2. spaCy 로 NER

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. was founded by Steve Jobs in California in 1976.")

for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple Inc. ORG
# Steve Jobs PERSON
# California GPE
# 1976 DATE
```

장점: *빠르고 가벼움* (CPU 로 동작). 산업 NER 의 표준.

한국어: `ko_core_news_sm`. 또는 KLUE-NER 모델.

---

## 3. Transformers 로 NER

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
results = ner("Apple Inc. was founded by Steve Jobs.")
for r in results:
    print(r['word'], r['entity_group'], r['score'])
```

BERT 기반이 spaCy 보다 *정확하지만 느림*.

---

## 4. LLM Few-shot 정보 추출

### 4.1 자유 텍스트 추출

```python
prompt = """Extract structured information from text.

Text: "Apple Inc. was founded by Steve Jobs in California in 1976."

Extract:
- founder
- company
- location
- year"""

response = llm(prompt)
```

### 4.2 JSON Schema 강제

```python
from pydantic import BaseModel

class CompanyInfo(BaseModel):
    company: str
    founder: str
    location: str
    year: int

# OpenAI Structured Outputs
result = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": text}],
    response_format=CompanyInfo,
)
```

100% 유효 JSON 보장 (Vol 69).

---

## 5. 도메인 특화 추출

### 5.1 의료 차트

```
환자: "30 세 여성, 두통 3 일, BP 130/85, 약 ibuprofen 200mg 처방"

추출:
- age: 30
- gender: female
- symptom: ["두통"]
- duration: "3 일"
- vital_signs: {"BP": "130/85"}
- medication: [{"name": "ibuprofen", "dose": "200mg"}]
```

### 5.2 법률 문서

- 사건 번호
- 당사자
- 청구 금액
- 판결 일자

### 5.3 인보이스

- 발행 회사
- 받는 회사
- 항목별 금액
- 합계
- 결제 기한

---

## 6. 평가

### 6.1 메트릭

- *Precision* — 추출한 것 중 정답 비율
- *Recall* — 정답 중 추출한 비율
- *F1* — 조화 평균

### 6.2 주의

*경계 일치* — "Apple Inc." vs "Apple" 도 다른 결과. 평가 시 *정확 일치 vs 부분 일치* 정책 정해야.

---

## 7. 산업 응용

- **이메일 자동 분류·라우팅**
- **계약서 핵심 항목 추출**
- **뉴스 트렌드 분석**
- **고객 지원 티켓 정보 추출**
- **의료 EMR 구조화**
- **금융 거래 명세 분석**

LLM 의 등장으로 *전통적 NER 모델 → LLM Few-shot* 으로 산업의 무게중심이 이동 중.

---

## 권 정리

- NER 표준 라벨 = PER·ORG·LOC·DATE·MONEY 등
- spaCy = 빠르고 가벼움
- Transformers = 더 정확
- LLM + JSON Schema = 새 표준
- 도메인 특화는 추출 스키마가 결정
- F1 평가 + 경계 정책

가장 기억할 한 줄: **"비정형 텍스트에서 구조화 데이터를 만드는 일이 NER·정보 추출이며, LLM + JSON Schema 가 새 산업 표준이다."**

다음 권: [Volume 102 — 텍스트 분류 4 접근 비교](./volume_102_text_classification_compare.md)

---

## 자가점검 키워드

`NER`, `PER/ORG/LOC`, `spaCy`, `BERT NER`, `JSON Schema`, `Precision/Recall/F1`

## 자가점검 질문

1. NER 표준 라벨 5 가지를 적으십시오.
2. spaCy 와 BERT NER 의 트레이드오프를 적으십시오.
3. LLM 으로 구조화 추출하는 코드를 적으십시오.
4. 자기 도메인 (의료·법률·금융 중) 의 추출 스키마를 설계하십시오.

## 다음 권

[Volume 102 — 텍스트 분류 4 접근 비교](./volume_102_text_classification_compare.md)
