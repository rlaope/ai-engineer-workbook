# Volume 41 — 딥러닝 디버깅

> 이 권이 끝나면 *학습이 안 된다* 는 모호한 증상을 *어디가 어떻게 잘못되었는지* 로 좁힐 수 있는 진단 워크플로를 갖게 됩니다.

## 목적

신경망 학습은 *조용히 실패* 하는 경우가 많습니다. 손실이 줄지 않거나, 줄긴 하는데 검증 성능이 나쁘거나, NaN 으로 발산하거나. 일반 소프트웨어 디버깅보다 *진단이 어려운* 이유는 *비결정성·통계적 특성·여러 상호작용* 때문입니다. 이 권은 표준 진단 절차를 다집니다.

## 선수 지식

- Volume 32, 33, 36 완료

## 학습 결과

1. 학습 실패 4 가지 패턴을 진단할 수 있습니다.
2. NaN/Inf 발생을 단계별로 좁힐 수 있습니다.
3. 과적합·과소적합을 그래프로 식별합니다.
4. 데이터·코드·모델·옵티마이저의 어디 문제인지 분리할 수 있습니다.

---

## 1. 학습 실패 4 가지 패턴

```
1. 손실이 전혀 안 줄어듦      → 코드 또는 학습률 문제
2. 손실 NaN/Inf              → 수치 안정성 문제
3. 학습 손실 줄지만 검증 안 줄음 → 과적합
4. 학습·검증 둘 다 안 줄음    → 과소적합
```

각 패턴마다 *우선 점검 항목* 이 다릅니다.

---

## 2. 손실이 줄지 않을 때

### 2.1 첫 점검

```python
# 1. 데이터가 모델에 잘 들어가는지
batch = next(iter(loader))
print(batch[0].shape, batch[1].shape)

# 2. 모델이 출력을 만드는지
out = model(batch[0])
print(out.shape)

# 3. 손실이 계산되는지
loss = criterion(out, batch[1])
print(loss.item())

# 4. 그래디언트가 흐르는지
loss.backward()
total = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
print(f"Grad norm: {total}")
```

각 단계에서 *기대값과 다른 결과* 가 나오는 단계를 좁힘.

### 2.2 흔한 원인

- *학습률 너무 작음* — LR Range Test 권장
- *옵티마이저 `step()` 누락*
- *`zero_grad()` 누락* — 그래디언트 누적
- *모델이 `eval()` 모드* — Dropout/BN 비활성
- *`requires_grad=False`* — 그래디언트 추적 안 됨

### 2.3 작은 데이터 과적합 테스트

10-100 개 샘플로만 학습 → *손실이 0 에 가까이 가야 함*. 안 되면 모델 또는 코드 문제.

```python
small_loader = DataLoader(TensorDataset(X[:100], y[:100]), batch_size=32)
for epoch in range(100):
    for x, y in small_loader:
        ...
```

---

## 3. NaN/Inf 디버깅

### 3.1 단계별 점검

```python
# 입력에 NaN/Inf 있는가
assert torch.isfinite(batch).all()

# 모델 출력에 NaN/Inf
out = model(batch)
assert torch.isfinite(out).all()

# 손실
loss = criterion(out, target)
assert torch.isfinite(loss)

# 그래디언트
loss.backward()
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print(f"NaN gradient in {name}")
```

### 3.2 흔한 원인

- 학습률 과다
- log(0) 또는 1/0 (수치 안정성 결여)
- FP16 의 표현 범위 초과
- 데이터에 NaN/Inf
- softmax overflow

### 3.3 대응

- 학습률 절반으로
- BF16 으로 전환 (FP16 의 범위 부족이면)
- 그래디언트 클리핑 (`clip_grad_norm_`)
- 데이터 전처리 검증

---

## 4. 과적합 진단

### 4.1 학습 곡선

```
loss
 |
 |\           학습 손실
 | \          \
 |  \          \____
 |   ─────       ────
 |   /     \
 |  /  ─── /  검증 손실 (다시 올라감 = 과적합)
 |
 +─────────────────── epoch
```

검증 손실이 *최저점 후 다시 상승* 하면 과적합.

### 4.2 대응

- Early Stopping
- Dropout/Weight Decay 강도 증가
- 데이터 증강
- 모델 사이즈 줄이기
- 더 많은 데이터

---

## 5. 과소적합 진단

학습 손실도 줄지 않거나 *충분히 작아지지 않음*.

대응:
- 모델 사이즈 키우기
- 학습 시간 늘리기
- 학습률 조정
- 정칙화 약화
- 특성 공학 추가
- 데이터 품질 점검

---

## 6. 디버깅 도구

- **PyTorch Profiler** — 시간·메모리 분석
- **W&B** — 학습 곡선·시스템 메트릭
- **`torch.autograd.set_detect_anomaly(True)`** — NaN 발생 위치 추적 (느림)
- **`torch.save`** — 문제 시점 체크포인트로 재현

---

## 권 정리

- 학습 실패 4 패턴 = 손실 안 줄음·NaN·과적합·과소적합
- 작은 데이터 과적합 테스트 = 코드/모델 정상 검증
- NaN 디버깅 = 입력→출력→손실→그래디언트 단계별
- 과적합 = 정칙화·데이터·모델 축소
- 과소적합 = 모델 확대·학습 연장

가장 기억할 한 줄: **"학습 실패의 원인은 항상 데이터·코드·모델·하이퍼파라미터 4 곳 중 하나이며, 이 4 곳을 순서대로 좁히면 진단된다."**

다음 권: [Volume 42 — 합성곱 신경망(CNN)](./volume_42_cnn.md)

---

## 자가점검 키워드

`작은 데이터 과적합 테스트`, `NaN 디버깅`, `학습 곡선`, `Early Stopping`, `Profiler`

## 자가점검 질문

1. 학습 실패 4 패턴을 적으십시오.
2. 손실이 줄지 않을 때 점검 4 단계를 적으십시오.
3. NaN 디버깅의 단계별 검증 코드를 적으십시오.
4. 과적합과 과소적합의 학습 곡선 차이를 그리십시오.

## 다음 권

[Volume 42 — 합성곱 신경망(CNN)](./volume_42_cnn.md)
