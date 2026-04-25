# Volume 55 — 단어 임베딩

> 이 권이 끝나면 *벡터로 의미를 표현한다* 는 한 문장이 어떤 알고리즘적 실체를 가지는지 알게 됩니다.

## 목적

단어 임베딩은 *단어를 고정 차원의 벡터로 표현* 하는 기법입니다. Word2Vec (2013) 의 등장이 NLP 의 표현 학습 시대를 열었으며, 이후 모든 신경망 기반 NLP 의 기반이 되었습니다.

## 선수 지식

- Volume 9, 32 완료

## 학습 결과

1. Word2Vec 의 *Skip-gram·CBOW* 발상을 알 수 있습니다.
2. *king - man + woman ≈ queen* 같은 *벡터 산술* 의 의미를 이해합니다.
3. GloVe·FastText 의 차이를 알 수 있습니다.
4. 단어 임베딩과 *문맥 임베딩 (BERT)* 의 차이를 구분합니다.

---

## 1. Word2Vec

### 1.1 발상

같은 *문맥* 에 등장하는 단어는 *비슷한 벡터* 가 되도록 학습.

```
"the dog runs fast"
"the cat runs fast"

→ dog 와 cat 의 임베딩이 비슷해짐
```

### 1.2 두 가지 변형

- **Skip-gram** — 중심 단어로 *주변 단어 예측*
- **CBOW (Continuous Bag of Words)** — 주변 단어로 *중심 단어 예측*

```python
from gensim.models import Word2Vec
sentences = [['I', 'love', 'AI'], ['Deep', 'learning', 'is', 'fun']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
print(model.wv['AI'])   # 100 차원 벡터
```

### 1.3 벡터 산술

학습된 임베딩은 *의미적 관계* 를 벡터 산술로 표현:

```
king - man + woman ≈ queen
paris - france + japan ≈ tokyo
walking - walked + ran ≈ running
```

이 발견이 *Word2Vec 을 유명하게 만든* 결과입니다.

---

## 2. GloVe

Stanford 의 GloVe (2014). *단어 동시 출현 행렬* 의 행렬 분해로 임베딩 학습.

Word2Vec 과 결과 비슷하지만 *전역 통계 활용*.

---

## 3. FastText

Facebook 의 FastText (2017). 단어를 *문자 n-gram* 의 합으로 표현.

장점: *학습 시 못 본 단어 (OOV)* 도 임베딩 가능. 형태소가 풍부한 언어 (한국어 등) 에 강함.

---

## 4. 단어 임베딩의 한계

### 4.1 다의어 문제

*"bank"* 가 *은행* 인지 *강둑* 인지 구분 못 함. Word2Vec 은 *한 단어 = 한 벡터*.

### 4.2 문맥 의존성

같은 단어도 *문맥에 따라 의미가 다름*. 단어 임베딩은 이를 표현 못 함.

### 4.3 해결: 문맥 임베딩 (BERT)

ELMo (2018), BERT (2018) 가 *문맥에 따라 다른 임베딩* 을 생성. 같은 단어라도 문장이 다르면 다른 벡터.

이 발견이 단어 임베딩 시대를 끝내고 *문맥 임베딩 시대* 를 열었습니다.

---

## 5. 현재의 위치

단어 임베딩은 *현재 NLP 의 주류는 아니지만*:

- *간단한 검색* — Word2Vec 임베딩으로 충분
- *작은 프로젝트* — 가벼움
- *임베딩 산술 직관* — 학습 가치 큼

문장·문서 임베딩 (Vol 56) 으로 자연스럽게 이어집니다.

---

## 권 정리

- Word2Vec = Skip-gram/CBOW 로 단어 → 벡터
- 벡터 산술이 *의미 관계* 표현
- GloVe = 행렬 분해, FastText = 문자 n-gram
- 한계 = 다의어·문맥 의존성
- 해결 = 문맥 임베딩 (BERT)

가장 기억할 한 줄: **"Word2Vec 은 단어를 벡터로 표현한 첫 폭발적 성공이며, 그 발상이 BERT 와 LLM 의 임베딩으로 진화했다."**

다음 권: [Volume 56 — 문장·문서 임베딩](./volume_56_sentence_embedding.md)

---

## 자가점검 키워드

`Skip-gram`, `CBOW`, `벡터 산술`, `GloVe`, `FastText`, `다의어`, `문맥 임베딩`

## 자가점검 질문

1. Skip-gram 과 CBOW 의 차이를 적으십시오.
2. *king - man + woman ≈ queen* 의 의미를 설명하십시오.
3. Word2Vec 과 BERT 의 본질적 차이를 적으십시오.
4. FastText 가 *형태소 풍부 언어* 에 강한 이유를 설명하십시오.

## 다음 권

[Volume 56 — 문장·문서 임베딩](./volume_56_sentence_embedding.md)
