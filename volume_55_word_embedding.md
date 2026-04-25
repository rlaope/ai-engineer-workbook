# Volume 15 — 단어 임베딩

> 이 권이 끝나면 *벡터로 의미를 표현한다*는 한 문장이 어떤 알고리즘적 실체를 가지는지 알게 됩니다.

## 목적

자연어를 신경망에 넣으려면 먼저 *문자나 단어를 벡터로 바꾸는 일*이 필요합니다. 단어 임베딩은 이 변환을 학습 가능한 형태로 만든 첫 시도였고, *왕 - 남자 + 여자 ≈ 여왕* 같은 의미 산술이 가능한 표현 공간을 만들어 냈습니다. 이 권은 Word2Vec·GloVe·FastText 의 원리를 손으로 코드까지 따라갑니다.

## 선수 지식

- Volume 8, 6, 19 완료
- 외부 지식: 단어와 문장의 일반 개념

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. One-Hot 표현과 임베딩 표현의 차이를 설명할 수 있습니다.
2. Word2Vec 의 Skip-Gram·CBOW 학습 목표를 그릴 수 있습니다.
3. Negative Sampling 의 동기를 알 수 있습니다.
4. GloVe 의 *공기 빈도 행렬 분해*가 무엇인지 설명할 수 있습니다.
5. FastText 가 *서브워드*를 다루는 이유를 알 수 있습니다.

## 챕터 목차

1. **자연어를 벡터로** — One-Hot 의 한계
2. **Word2Vec 의 직관** — *주변 단어로 자기를 예측*
3. **Skip-Gram vs CBOW**
4. **Negative Sampling**
5. **GloVe — 공기 빈도 행렬의 분해**
6. **FastText — 서브워드 임베딩**
7. **임베딩 평가** — 유추·유사도·다운스트림
8. **gensim 으로 직접 학습해 보기**

## 자가점검 키워드

`One-Hot`, `Word2Vec`, `Skip-Gram`, `CBOW`, `Negative Sampling`, `GloVe`, `FastText`, `유추`

## 다음 권

[Volume 34 — 문장·문서 임베딩](./volume_56_sentence_embedding.md)
