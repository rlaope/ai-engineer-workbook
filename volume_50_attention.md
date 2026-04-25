# Volume 50 — Attention 메커니즘

> 이 권이 끝나면 어텐션이 *유사도 기반 가중평균* 이라는 단순한 한 문장으로 요약된다는 사실에 안도하게 됩니다.

## 목적

어텐션은 트랜스포머의 핵심이며, 그 핵심은 의외로 단순합니다. *질의 (Query) 에 대해 키 (Key) 와의 유사도를 계산하고, 그 가중치로 값 (Value) 을 평균하는 일* 이 전부입니다.

## 선수 지식

- Volume 9, 49 완료

## 학습 결과

1. Q·K·V 의 의미를 비유로 설명할 수 있습니다.
2. 스케일드 닷프로덕트 어텐션의 수식을 외우지 않고도 유도할 수 있습니다.
3. 소프트맥스가 어텐션에서 하는 역할을 설명합니다.
4. *어텐션은 가중평균* 이라는 한 문장으로 요약합니다.

---

## 1. 어텐션의 직관

### 1.1 비유

도서관에서 책을 찾는 상황:
- *Query* — 찾고 싶은 주제 (검색어)
- *Key* — 각 책의 색인 정보
- *Value* — 책의 실제 내용

검색어와 색인의 *유사도가 높은 책의 내용을 더 많이* 가져옴. 이것이 어텐션의 본질.

### 1.2 수식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $QK^T$ — 모든 query-key 쌍의 유사도 (내적)
- $\sqrt{d_k}$ — 스케일링 (차원이 클수록 내적이 커지므로 정규화)
- $\text{softmax}$ — 유사도를 *합 1의 가중치* 로 변환
- $V$ — 가중치로 곱해질 값

---

## 2. NumPy 토이 어텐션

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

# 시퀀스 길이 4, 차원 8
Q = np.random.randn(4, 8)
K = np.random.randn(4, 8)
V = np.random.randn(4, 8)
print(attention(Q, K, V).shape)   # (4, 8)
```

이 5 줄이 *어텐션의 핵심* 입니다.

---

## 3. 소프트맥스의 역할

소프트맥스는 *모든 query-key 유사도를 합 1 의 분포로 변환*. 이 분포가 *가중치* 가 되어 V 에 곱해집니다.

만약 유사도가 *한 곳에 집중* 되면 → 그 V 만 거의 그대로 출력. *분산* 되면 → V 들의 평균에 가까움.

---

## 4. 마스킹

### 4.1 패딩 마스크

다른 길이의 시퀀스를 한 배치에서 처리하려면 *짧은 시퀀스를 패딩*. 패딩 위치는 *어텐션에 포함되지 않게* 마스킹.

### 4.2 인과 마스크 (Causal Mask)

자기회귀 모델 (LLM 학습) 에서, *미래 토큰을 보지 않게* 마스킹. 상삼각 행렬을 -∞ 로 채움 → softmax 후 0.

```python
mask = np.triu(np.ones((4, 4)), k=1) * -1e9
scores = Q @ K.T / np.sqrt(d_k) + mask
```

---

## 5. 의미

### 5.1 정보 병목 해소

RNN 은 *고정 크기 hidden state* 에 모든 과거를 압축. 어텐션은 *모든 과거 토큰을 직접 참조*. 정보 손실 없음.

### 5.2 병렬화

RNN 은 순차 계산. 어텐션은 *모든 위치를 동시에 계산*. GPU 의 강점 활용.

### 5.3 긴 의존성

RNN 은 멀리 떨어진 의존성 어려움. 어텐션은 *거리에 무관하게 직접 연결*.

---

## 권 정리

- 어텐션 = Q·K 유사도 → softmax → V 가중평균
- 5 줄 NumPy 로 구현 가능
- 마스킹 = 패딩·인과 처리
- RNN 의 한계 (정보 병목·순차·긴 의존성) 모두 해결

가장 기억할 한 줄: **"어텐션은 도서관에서 검색어로 책을 찾는 일과 같다 — Query 가 Key 와 매칭되어 Value 의 가중평균이 나온다."**

다음 권: [Volume 51 — Transformer 완전 정복](./volume_51_transformer.md)

---

## 자가점검 키워드

`Q/K/V`, `스케일드 닷프로덕트`, `소프트맥스`, `패딩 마스크`, `인과 마스크`, `정보 병목`

## 자가점검 질문

1. 어텐션을 도서관 비유로 설명하십시오.
2. 어텐션 수식의 4 부분 (QK^T·√d_k·softmax·V) 의 역할을 적으십시오.
3. NumPy 로 어텐션을 5 줄 안에 구현하십시오.
4. RNN 의 3 한계를 어텐션이 어떻게 해결하는지 적으십시오.

## 다음 권

[Volume 51 — Transformer 완전 정복](./volume_51_transformer.md)
