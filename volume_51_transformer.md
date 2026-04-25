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
