# Volume 51 — Transformer 완전 정복

> 이 권이 끝나면 빈 노트북에 100 줄 안쪽으로 미니 트랜스포머를 구현할 수 있게 됩니다.

## 목적

Transformer 는 2017 년 *Attention Is All You Need* 논문으로 등장한 이후 *현대 모든 LLM·Vision·멀티모달 모델의 기반* 이 되었습니다. 그 구조는 의외로 단순하며, *Self-Attention + FFN + Residual + Normalization* 의 결합입니다. 이 권은 표준 Transformer 의 모든 구성요소를 다집니다.

## 선수 지식

- Volume 32, 50 완료

## 학습 결과

1. Transformer Block 의 5 부품을 모두 적을 수 있습니다.
2. Multi-Head Attention 의 *왜 Multi* 인지 설명할 수 있습니다.
3. Position Encoding (Sinusoidal·RoPE·ALiBi) 의 차이를 알 수 있습니다.
4. Encoder vs Decoder 의 차이를 적을 수 있습니다.
5. PyTorch 로 미니 Transformer 를 구현할 수 있습니다.

---

## 1. Transformer Block 의 5 부품

```
입력 → LayerNorm → Multi-Head Attention → +
                                          ↓
        ←──────────────── 잔차 (Residual) ─┘
        
        ↓
        LayerNorm → FFN → +
                          ↓
        ←──── 잔차 ────────┘
        
        → 출력
```

5 부품:
1. **LayerNorm** — 입력 정규화
2. **Multi-Head Attention** — 토큰 간 정보 교환
3. **Residual** — 그래디언트 흐름
4. **FFN (Feed-Forward Network)** — 토큰별 비선형 변환
5. **두 번째 LayerNorm + Residual** — 같은 패턴 반복

---

## 2. Multi-Head Attention

### 2.1 Why Multi?

단일 어텐션은 *한 가지 관계만* 학습. Multi-Head 는 *여러 다른 관계 (구문·의미·위치)* 를 동시에.

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
# 8 개의 head, 각각 512/8 = 64 차원
out, attn = mha(query, key, value)
```

### 2.2 헤드 분할

```
Q, K, V (B, L, D) → 각 헤드 (B, L, D/H) 분할 → 어텐션 → 결합
```

각 헤드는 *독립적으로 어텐션 계산*, 결과를 *concatenate* 후 마지막 Linear.

---

## 3. FFN

```python
class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

각 토큰에 *독립적으로* 적용되는 2-layer MLP. 차원이 4 배로 확장 후 다시 축소.

현대 LLM 은 ReLU 대신 *GELU·SiLU·SwiGLU* 사용.

---

## 4. Position Encoding

Attention 자체는 *순서 정보 없음*. Position Encoding 으로 순서 주입.

### 4.1 Sinusoidal (원본 Transformer)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### 4.2 Learned

학습 가능한 임베딩. BERT 가 사용.

### 4.3 RoPE (Rotary Position Embedding)

회전 행렬로 *상대 위치* 인코딩. LLaMA·Mistral 표준.

### 4.4 ALiBi

위치에 비례한 *어텐션 페널티*. 외삽 (학습보다 긴 시퀀스) 에 강함.

---

## 5. Encoder vs Decoder

### 5.1 Encoder (BERT 류)

- 양방향 어텐션 (모든 토큰이 서로 봄)
- 분류·임베딩에 적합
- 마스킹 LM 으로 학습

### 5.2 Decoder (GPT 류)

- 인과 마스크 (미래 안 봄)
- 자기회귀 생성
- 다음 토큰 예측으로 학습

### 5.3 Encoder-Decoder (T5·번역 모델)

Encoder 가 입력 인코딩, Decoder 가 출력 생성. *Cross-Attention* 으로 연결.

---

## 6. 미니 Transformer (100 줄)

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x, mask=None):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.tok_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
```

이 50 줄이 *GPT 계열 LLM 의 핵심 골격* 입니다.

---

## 7. Transformer 의 역사적 등장과 영향

### 7.1 등장 (2017)

Transformer 는 2017 년 6 월 Google Brain 의 *Attention Is All You Need* 논문으로 발표되었습니다. 저자 8 명 (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin) 이 모두 *현재 AI 산업의 핵심 인물* 이 되었습니다. 처음에는 *기계 번역 (영어→독일어, 영어→프랑스어)* 의 SOTA 향상이 발표 이유였지만, 이후 영향은 *AI 분야 전체* 로 퍼졌습니다.

### 7.2 RNN 시대의 종결

2017 년 이전 NLP 의 표준은 *RNN/LSTM/GRU* 였습니다 (Vol 49 참고). Seq2Seq + Attention 의 조합이 번역의 SOTA 였지만:

- *순차 계산* 으로 GPU 활용 못 함
- *긴 의존성 (수백 토큰)* 학습 어려움
- *학습 시간이 매우 김*

Transformer 는 *RNN 자체를 버리고 Attention 만으로* 시퀀스를 처리해 이 세 한계를 동시에 해결했습니다. *Attention Is All You Need* 라는 제목이 이 선언을 압축합니다.

### 7.3 BERT 와 GPT 의 분기

2018 년, 같은 Transformer 구조가 *두 다른 방향* 으로 분기했습니다.

**BERT (Google, 2018 년 10 월)** — Encoder 만 사용. 양방향 어텐션. 마스킹 LM 으로 학습. *문장 이해·분류·임베딩* 에 강함.

**GPT (OpenAI, 2018 년 6 월)** — Decoder 만 사용. 인과 마스크. 다음 토큰 예측으로 학습. *생성·창작·대화* 에 강함.

처음에는 BERT 가 NLP 의 주류였지만, 2020 년 GPT-3 발표 이후 *생성 모델 (Decoder)* 이 압도적 주류가 되었습니다. 현재 모든 LLM (GPT-4, Claude, Gemini, LLaMA, Mistral 등) 은 *Decoder-only* 구조입니다.

### 7.4 NLP 를 넘어선 확장

2020 년 이후 Transformer 는 NLP 영역을 넘어 *모든 모달리티* 로 확장되었습니다.

- *비전* — Vision Transformer (Vol 45)
- *음성* — Whisper, wav2vec 2.0
- *멀티모달* — CLIP (Vol 48), Flamingo, LLaVA
- *생성 이미지* — DiT (Diffusion Transformer)
- *생성 비디오* — Sora (Spacetime Transformer)
- *코드* — Codex, CodeLlama
- *3D / 단백질* — AlphaFold 2/3

이 확장의 의미는 단순히 *한 알고리즘이 모든 곳에 쓰임* 이 아닙니다. *같은 구조 위에서 멀티모달 통합* 이 가능해진 것입니다. 텍스트와 이미지를 같은 트랜스포머에 넣을 수 있다는 것이 *멀티모달 모델 시대* 의 직접적 기반입니다.

### 7.5 산업·자본의 흐름

Transformer 가 만든 산업적 변화:

- *2019* — OpenAI 가 마이크로소프트로부터 10 억 달러 투자
- *2022* — ChatGPT 출시, 일주일에 100 만 사용자
- *2023* — NVIDIA 시가총액 1 조 달러 돌파
- *2024* — AI 분야에 *1 천억 달러+* 자본 유입
- *2025* — Foundation Model 시장의 주도권 경쟁 본격화

이 모든 흐름의 *기술적 기반* 이 2017 년의 한 논문입니다. Transformer 는 *컴퓨터 과학 역사상 가장 영향력 있는 단일 논문* 중 하나로 평가됩니다.

### 7.6 챕터 정리

Transformer 는 2017 년 기계 번역에서 출발해 *NLP·비전·음성·멀티모달·생성·과학* 의 모든 AI 영역으로 확장된 단일 알고리즘입니다. RNN 시대를 종결하고 Foundation Model 시대를 열었으며, 현재 모든 SOTA AI 모델의 직접적 기반입니다.

---

## 8. Self-Attention 의 수식 완전 유도

### 8.1 동기

Self-Attention 의 수식 $\text{softmax}(QK^T/\sqrt{d_k})V$ 가 *어떻게 도출되었는가* 를 단계별로 따라가면 그 의미가 손에 잡힙니다. 이 챕터는 *수학적 유도* 를 통해 직관을 강화합니다.

### 8.2 한 토큰의 어텐션

길이 $n$, 차원 $d$ 의 입력 $X \in \mathbb{R}^{n \times d}$ 가 있다고 합시다. 각 행이 한 토큰의 임베딩.

이 입력에서 *세 가지 사영* 을 만듭니다.

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

$W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 는 *학습 가능한 가중치* 입니다. $Q, K, V \in \mathbb{R}^{n \times d_k}$.

이 사영은 같은 입력에서 *세 가지 역할* 을 추출합니다.

- $Q$ — *내가 무엇을 찾고 있는가* (질의)
- $K$ — *나는 무엇을 가지고 있는가* (색인)
- $V$ — *나의 실제 내용* (값)

같은 토큰이지만 *다른 사영* 으로 다른 역할을 합니다. 이것이 self-attention 이 *self* 인 이유 — 자기 자신과 자기 자신이 상호작용.

### 8.3 어텐션 점수

토큰 $i$ 의 query $q_i$ 가 토큰 $j$ 의 key $k_j$ 와 *얼마나 일치하는가* 를 내적으로 측정합니다.

$$\text{score}_{ij} = q_i \cdot k_j$$

이를 모든 쌍에 대해 한 번에 계산하면:

$$S = QK^T \in \mathbb{R}^{n \times n}$$

$S_{ij}$ 가 *토큰 i 가 토큰 j 에 얼마나 주목해야 하는가* 의 raw 점수.

### 8.4 스케일링의 이유

$d_k$ 가 클수록 내적의 분산이 커집니다. 분산이 큰 값이 softmax 에 들어가면 *극단으로 sharp 한 분포* 가 되어 그래디언트가 사라집니다.

따라서 $\sqrt{d_k}$ 로 나눠 *분산을 일정하게 유지*:

$$S' = \frac{QK^T}{\sqrt{d_k}}$$

이 스케일링이 없으면 학습이 불안정해집니다. 이 단순한 트릭이 *Transformer 학습의 안정성* 을 만든 핵심 중 하나.

### 8.5 가중치 정규화

raw 점수를 *가중치 (합 1)* 로 변환하기 위해 softmax 적용:

$$A = \text{softmax}(S') \in \mathbb{R}^{n \times n}$$

$A_{ij}$ 가 *토큰 i 가 토큰 j 에 주는 가중치*. 각 행의 합이 1.

### 8.6 가중평균

마지막으로 가중치로 value 를 평균:

$$Y = AV \in \mathbb{R}^{n \times d_k}$$

$Y_i = \sum_j A_{ij} V_j$ — 토큰 i 의 출력은 *모든 토큰의 value 의 가중평균*.

### 8.7 완전한 식

이 모든 단계를 한 줄로:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

각 부분의 의미를 한번 더 정리합니다.

- $QK^T$ — 모든 토큰 쌍의 *유사도 매트릭스*
- $\sqrt{d_k}$ — *분산 통제* 를 위한 스케일
- $\text{softmax}$ — *합 1 의 가중치* 변환
- $V$ — *실제 정보* 의 가중평균

### 8.8 NumPy 검증

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

n, d = 4, 8
np.random.seed(0)
X = np.random.randn(n, d)
W_q = np.random.randn(d, d) * 0.1
W_k = np.random.randn(d, d) * 0.1
W_v = np.random.randn(d, d) * 0.1

Q = X @ W_q
K = X @ W_k
V = X @ W_v

S = Q @ K.T / np.sqrt(d)
A = softmax(S, axis=-1)
print("어텐션 가중치 (각 행의 합 1):", A.sum(axis=-1))
Y = A @ V
print("출력 모양:", Y.shape)
```

이 12 줄이 *Self-Attention 의 본질 전부* 입니다. PyTorch 의 `nn.MultiheadAttention` 도 내부적으로 같은 계산을 합니다.

### 8.9 챕터 정리

Self-Attention 은 *Q·K·V 사영 → 점수 매트릭스 → 스케일 → softmax → V 의 가중평균* 의 5 단계 변환입니다. 각 단계가 *왜 그런 형태인지* 직관적 동기가 있으며, 이를 따라가면 수식이 *외울 대상* 이 아니라 *논리적 결과* 가 됩니다.

---

## 9. Multi-Head 의 깊은 분석

### 9.1 왜 여러 head 가 필요한가

단일 어텐션은 *한 가지 관계만 학습할 수 있는가* 라는 의문이 핵심입니다. 실제로 한 어텐션 행렬은 *한 가지 가중치 분포* 만 표현하지만, 언어에서는 *여러 종류의 관계* 가 동시에 존재합니다.

예를 들어 문장 *"The dog that I saw at the park yesterday was friendly"* 에서:

- *주어 - 동사* 관계 — "dog" ↔ "was"
- *수식어 - 명사* 관계 — "friendly" ↔ "dog"
- *시간 부사* 관계 — "yesterday" ↔ "saw"
- *장소 부사* 관계 — "park" ↔ "saw"
- *관계 대명사* — "that" ↔ "dog"

단일 어텐션 head 는 이 모든 관계 중 하나에 *집중* 하면서 다른 관계는 약하게 잡습니다. *여러 head 가 각각 다른 관계를 학습* 하면 모델이 더 풍부한 표현을 만들 수 있습니다.

### 9.2 헤드 분할의 메커니즘

차원 $d_{\text{model}}$ 의 입력을 $H$ 개의 head 로 나눕니다. 각 head 의 차원은 $d_k = d_{\text{model}} / H$.

```python
# d_model=512, H=8 → d_k=64
def multi_head(X, W_qs, W_ks, W_vs, W_o):
    H = len(W_qs)
    head_outputs = []
    for h in range(H):
        Q = X @ W_qs[h]   # (n, d_k)
        K = X @ W_ks[h]
        V = X @ W_vs[h]
        head = softmax(Q @ K.T / np.sqrt(d_k)) @ V
        head_outputs.append(head)
    
    concat = np.concatenate(head_outputs, axis=-1)   # (n, d_model)
    return concat @ W_o
```

각 head 는 *독립적인 Q, K, V 사영* 을 가지므로 *서로 다른 관계* 를 학습할 자유가 있습니다.

### 9.3 효율적 구현 — 단일 행렬 곱

위 의사코드는 직관적이지만 비효율적입니다. 실제 구현은 *모든 head 의 사영을 한 번의 큰 행렬 곱* 으로 처리합니다.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # 한 번의 사영으로 Q, K, V 모두 계산
        qkv = self.W_qkv(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, L, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 모든 head 동시 계산
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)
        
        out = attn @ V  # (B, n_heads, L, d_k)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_o(out)
```

이 구조에서 *모든 head 가 같은 GPU 커널 호출* 안에서 처리되므로 매우 효율적입니다.

### 9.4 어텐션 시각화

학습된 모델의 어텐션 가중치 $A$ 를 *히트맵* 으로 그리면 *모델이 무엇을 보는가* 를 직관적으로 확인할 수 있습니다.

```python
import matplotlib.pyplot as plt

# attn: (n_heads, n_tokens, n_tokens)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for h in range(8):
    ax = axes[h // 4, h % 4]
    ax.imshow(attn[h].cpu().numpy(), cmap='viridis')
    ax.set_title(f'Head {h}')
plt.show()
```

여러 head 가 *서로 다른 패턴* 을 보이는 것이 시각적으로 확인됩니다 — 어떤 head 는 *대각선 (자기 자신)*, 어떤 head 는 *이전 토큰*, 어떤 head 는 *문장 끝* 같은 다양한 패턴.

### 9.5 헤드 수의 선택

표준 설정:

- BERT-base: $d_{\text{model}}=768$, $H=12$, $d_k=64$
- BERT-large: $d_{\text{model}}=1024$, $H=16$, $d_k=64$
- GPT-3: $d_{\text{model}}=12288$, $H=96$, $d_k=128$
- LLaMA 3 70B: $d_{\text{model}}=8192$, $H=64$, $d_k=128$

$d_k = 64$ 또는 $128$ 이 *경험적으로* 가장 좋은 균형입니다. 이보다 작으면 표현력 부족, 크면 head 다양성 손실.

### 9.6 챕터 정리

Multi-Head Attention 은 *여러 종류의 관계를 동시에 학습* 하기 위한 구조이며, 효율적 구현은 *모든 head 를 한 번의 행렬 곱으로 처리* 합니다. 어텐션 시각화를 통해 각 head 가 *다른 패턴* 을 학습함을 확인할 수 있고, 표준 설정은 $d_k = 64$ 또는 $128$ 입니다.

---

## 10. Position Encoding 의 깊은 비교

### 10.1 왜 위치 정보가 필요한가

Self-Attention 은 *집합 (set) 연산* 입니다 — 입력 토큰의 *순서를 바꿔도 결과가 같습니다*. 이는 *순서가 의미를 결정하는 자연어* 에 치명적입니다.

예: *"개가 사람을 물었다"* 와 *"사람이 개를 물었다"* 는 같은 단어 집합이지만 의미가 정반대.

따라서 *순서 정보를 어텐션에 명시적으로 주입* 해야 하며, 이것이 Position Encoding 입니다.

### 10.2 Sinusoidal — 원본 Transformer

원본 Transformer (2017) 의 방식:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

각 위치에 대해 *서로 다른 주파수의 사인·코사인 값* 을 만들어 임베딩에 더합니다.

장점:
- *학습 데이터보다 긴 시퀀스* 에 자연스럽게 외삽 가능 (이론상)
- *상대 위치 관계* 가 선형 변환으로 표현됨

단점:
- *외삽 성능이 실제로는 약함*
- *학습 가능한 파라미터 없음*

### 10.3 Learned Position Embedding

위치별 임베딩을 *학습 가능한 파라미터로* 만드는 방식. BERT 가 채택.

```python
self.pos_emb = nn.Embedding(max_len, d_model)
positions = torch.arange(seq_len)
x = token_emb + self.pos_emb(positions)
```

장점:
- *데이터에 맞춰 최적화*
- 단순함

단점:
- *학습 시 본 길이까지만 사용 가능* (max_len 제약)
- *외삽 불가*

### 10.4 RoPE (Rotary Position Embedding)

LLaMA, Mistral 등 *현대 LLM 의 표준*.

발상: Position 정보를 *덧셈* 이 아닌 *Q·K 의 회전* 으로 주입.

각 토큰의 Q·K 벡터를 *위치에 비례한 각도만큼 회전* 시킵니다. 두 벡터의 내적이 *상대 위치에만 의존* 하게 됩니다.

수학적으로는 Q·K 의 차원 쌍 $(2i, 2i+1)$ 을 다음과 같이 회전:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

여기서 $m$ 은 위치, $\theta_i = 10000^{-2i/d}$.

장점:
- *상대 위치만 의존* — 절대 위치 정보 없음
- *외삽 능력이 좋음* — 학습보다 긴 시퀀스에서도 작동
- *Long Context Scaling* 기법과 잘 결합 (NTK Scaling, YaRN)

### 10.5 ALiBi (Attention with Linear Biases)

위치 정보를 *임베딩에 더하지 않고 어텐션 점수에 페널티로* 추가.

$$\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|$$

$|i-j|$ 가 멀어질수록 *선형 페널티*. $m$ 은 head 별 다른 기울기.

장점:
- *외삽 능력 매우 좋음* — 학습 길이의 4-10 배까지 안정적
- 구현 단순

단점:
- 표현력은 RoPE 보다 약함
- 채택률은 RoPE 보다 낮음

### 10.6 비교 표

| 방식 | 채택 모델 | 외삽 | 표현력 | 구현 |
|------|----------|------|--------|------|
| Sinusoidal | 원본 Transformer | 약함 | 보통 | 단순 |
| Learned | BERT | 불가 | 강함 | 단순 |
| RoPE | LLaMA, Mistral, Qwen | 좋음 (확장 시 매우 좋음) | 강함 | 중간 |
| ALiBi | BLOOM 일부, MPT | 매우 좋음 | 보통 | 단순 |

### 10.7 챕터 정리

Position Encoding 은 *Self-Attention 이 집합 연산* 이라는 한계를 보완하기 위한 필수 부품입니다. 원본 Sinusoidal → Learned → RoPE → ALiBi 의 진화에서, 현재 LLM 의 사실상 표준은 *RoPE 입니다*. ALiBi 는 외삽 성능이 가장 좋지만 표현력 트레이드오프가 있습니다.

---

## 11. Transformer 변형 모델 계보

### 11.1 Encoder-only 계열 — BERT 가족

BERT (2018) 의 영향:

- *RoBERTa (2019)* — BERT 의 학습 개선 (NSP 제거, 더 긴 학습)
- *ALBERT (2019)* — 파라미터 공유로 더 작게
- *DistilBERT (2019)* — Teacher-Student 증류로 40% 작게
- *DeBERTa (2020)* — 분리된 어텐션 메커니즘
- *ELECTRA (2020)* — Generator-Discriminator 학습
- *모델별 다국어 변형* — XLM-R, mBERT

이 계열은 *분류·임베딩* 에 강하지만 *생성에 약함*. 현재는 *임베딩 모델 (Sentence-BERT)* 형태로 살아남음.

### 11.2 Decoder-only 계열 — GPT 가족

GPT (2018), GPT-2 (2019), GPT-3 (2020) 가 *현대 LLM 의 기반*. 이후 모든 LLM:

- OpenAI: GPT-3.5, GPT-4, GPT-4o, o1, o3
- Anthropic: Claude 1, 2, 3, 3.5, 3.7, 4
- Google: PaLM, Gemini 1, 1.5, 2
- Meta: LLaMA 1, 2, 3, 4
- Mistral: Mistral 7B, Mixtral, Codestral
- Alibaba: Qwen 1.5, 2, 2.5
- DeepSeek: V2, V3, R1
- Microsoft: Phi-3, Phi-4

모두 *Decoder-only Transformer* 의 변형이며, *스케일·데이터·정렬* 에서 차별화.

### 11.3 Encoder-Decoder 계열 — T5 가족

번역·요약 같은 *시퀀스→시퀀스* 작업에 자연스러운 구조.

- *T5 (2019)* — Text-to-Text Transfer Transformer
- *BART (2019)* — 디노이징 사전학습
- *FLAN-T5 (2022)* — instruction tuning

산업에서는 점차 *Decoder-only LLM* 이 잠식 중.

### 11.4 Vision 영역 — ViT 가족

- *ViT (2020)* — 이미지를 패치로 자르는 첫 시도
- *DeiT (2021)* — Distillation 으로 데이터 효율 개선
- *Swin Transformer (2021)* — 계층적 윈도우 어텐션
- *MAE (2021)* — Masked AutoEncoder 사전학습
- *DINO / DINOv2 (2021, 2023)* — 자가지도 학습

CNN 이 여전히 강한 영역도 있지만, *멀티모달 통합* 측면에서 ViT 가 표준이 되어 가는 중.

### 11.5 효율 변형 — Long Context

표준 Transformer 의 O(n²) 한계를 극복하려는 시도:

- *Sparse Transformer (2019)* — 희소 어텐션
- *Reformer (2020)* — LSH 기반 어텐션
- *Performer (2020)* — 커널 근사
- *LongFormer (2020)* — Sliding Window + Global
- *Mamba (2024)* — State Space Model 기반

Vol 52, 54 에서 자세히 다룸.

### 11.6 MoE 변형

- *Switch Transformer (2021)* — 첫 대규모 MoE
- *GLaM (2021)* — Google
- *Mixtral 8x7B (2023)* — Mistral, 오픈
- *DeepSeek-V2/V3 (2024)* — Multi-head Latent Attention + MoE
- *GPT-4 (추정)* — MoE 라는 강한 가설

Vol 53 에서 자세히 다룸.

### 11.7 챕터 정리

Transformer 의 변형은 *Encoder-only (BERT)·Decoder-only (GPT)·Encoder-Decoder (T5)* 의 세 가지 골격에서 출발해 *Vision·Long Context·MoE* 같은 다양한 가지로 진화했습니다. 현대 LLM 의 압도적 다수는 *Decoder-only* 입니다.

---

## 12. 흔한 함정과 디버깅

### 12.1 마스크 누락

자기회귀 학습에서 *인과 마스크 (causal mask)* 누락은 *데이터 누수* 를 일으킵니다. 모델이 *미래 토큰을 보면서 다음 토큰을 예측* 하므로 학습 손실은 매우 작아지지만 추론 성능은 무너집니다.

```python
# 인과 마스크 (상삼각이 -inf)
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

검증: 학습 손실이 *너무 빠르게 0 에 수렴* 한다면 마스크 누락 의심.

### 12.2 패딩 토큰 처리

다른 길이의 시퀀스를 한 배치에서 처리하려면 패딩 필요. *패딩 위치도 마스킹* 해야 합니다.

```python
# attention_mask: 1=실제 토큰, 0=패딩
attn_mask = (input_ids != pad_token_id)  # (B, L)
attn_mask_4d = attn_mask[:, None, None, :]  # (B, 1, 1, L)
scores = scores.masked_fill(~attn_mask_4d, float('-inf'))
```

검증: 같은 입력을 *다른 패딩 길이* 로 호출했을 때 결과가 같아야 함.

### 12.3 Position Encoding 길이 초과

Learned Position Embedding 은 *학습 시 본 max_len 이상* 의 입력에서 *인덱스 에러* 를 냅니다. 추론 시 입력 길이 검증 필수.

```python
assert seq_len <= self.max_position_embeddings, \
    f"Input too long: {seq_len} > {self.max_position_embeddings}"
```

### 12.4 그래디언트 폭발

깊은 Transformer 의 학습 초기에 *그래디언트 폭발* 이 흔합니다. Warmup 없이 큰 학습률로 시작하면 발산.

방어:
- *Linear Warmup* 100-1000 스텝 (Vol 40)
- *Gradient Clipping* `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 12.5 메모리 폭발

긴 시퀀스에서 *어텐션 매트릭스가 메모리 폭발*. seq=8192 면 어텐션 매트릭스가 64MB × heads, 학습 시 활성화·그래디언트까지 곱해져 GB 단위.

방어:
- *FlashAttention* (Vol 54) 자동 사용 (PyTorch 2+)
- *Activation Checkpointing*
- 시퀀스 길이 제한
- 더 큰 GPU

### 12.6 학습 불안정성

큰 모델 학습에서 *간헐적 손실 폭증* 이 발생합니다. 흔한 원인:

- 데이터에 *이상치 배치* 가 들어감
- *FP16 오버플로* — BF16 로 전환
- 옵티마이저 상태 불안정 — Adam 의 *β1, β2, eps* 튜닝

방어: *손실 모니터링 + 자동 롤백* 시스템 구축.

### 12.7 챕터 정리

Transformer 학습·운영의 흔한 함정은 *마스크 누락·패딩 처리·길이 초과·그래디언트 폭발·메모리 폭발·학습 불안정성* 의 6 가지입니다. 각 함정마다 *명확한 검증·방어 패턴* 이 있으므로 미리 알아 두면 디버깅 시간을 크게 줄일 수 있습니다.

---

## 13. 학습·추론 시점의 차이

### 13.1 학습

- 모든 토큰을 *한 번에 입력* (배치 + 시퀀스)
- 인과 마스크로 *미래 차단*
- 모든 위치에서 *동시 손실 계산*
- 그래디언트 역전파 + 옵티마이저 갱신
- 메모리: 가중치 + 그래디언트 + 옵티마이저 상태 + 활성화

### 13.2 추론 — Prefill 단계

처음 사용자 입력 (프롬프트) 처리:

- 모든 입력 토큰을 *한 번에 forward*
- KV 캐시 구축
- 첫 토큰의 logit 출력
- 이 단계가 *TTFT (Time To First Token)* 를 결정

### 13.3 추론 — Decode 단계

이후 토큰 한 번에 하나씩 생성:

- 새 토큰 *하나만* forward
- KV 캐시에서 과거 정보 가져옴
- 새로 생성한 K/V 를 캐시에 추가
- 다음 토큰 출력
- 이 단계가 *TPOT (Time Per Output Token)* 를 결정

### 13.4 메모리 차이

학습:
- 70B 모델 → 가중치 140GB + 그래디언트 140GB + 옵티마이저 560GB + 활성화 100GB+ ≈ 940GB
- H100 8 장 + FSDP 필요

추론:
- 70B 모델 → 가중치 140GB + KV 캐시 5-50GB ≈ 145-190GB
- H100 2-3 장으로 가능

추론 메모리는 *학습의 약 1/5*. 같은 모델도 *학습은 클러스터, 추론은 단일 노드* 가 표준.

### 13.5 챕터 정리

Transformer 의 학습·추론은 *완전히 다른 워크로드* 입니다. 학습은 모든 토큰 동시 처리, 추론은 Prefill (한 번에) + Decode (순차). 메모리·SLA·인프라가 모두 다르며, AI 엔지니어는 주로 *추론 단계* 를 다룹니다.

---

## 권 정리

- Transformer Block 5 부품 = LN·MHA·Res·FFN·LN+Res
- Multi-Head = 여러 관계 동시 학습
- Position Encoding (Sinusoidal·RoPE·ALiBi) 으로 순서 주입
- Encoder (BERT)·Decoder (GPT)·Encoder-Decoder (T5)
- 100 줄 미니 트랜스포머가 LLM 의 골격

가장 기억할 한 줄: **"Transformer = Self-Attention + FFN + 잔차 + 정규화 — 이 5 부품의 반복이 모든 현대 모델의 기반이다."**

다음 권: [Volume 52 — Long Context 기법](./volume_52_long_context.md)

---

## 자가점검 키워드

`Transformer Block`, `Multi-Head Attention`, `FFN`, `LayerNorm`, `Position Encoding`, `RoPE`, `Encoder/Decoder`

## 자가점검 질문

1. Transformer Block 의 5 부품을 적으십시오.
2. Multi-Head Attention 이 *왜 단일 head 보다 좋은가* 설명하십시오.
3. Position Encoding 4 종류를 비교하십시오.
4. Encoder·Decoder·Encoder-Decoder 의 차이를 적으십시오.
5. 미니 Transformer 를 50 줄 안에 구현하십시오.

## 다음 권

[Volume 52 — Long Context 기법](./volume_52_long_context.md)
