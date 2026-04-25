# Volume 62 — PyTorch 실전

> 이 권이 끝나면 빈 노트북에서 시작해 *데이터 로드 → 모델 정의 → 학습 → 평가 → 체크포인트 저장*까지를 30 분 안에 작성할 수 있게 됩니다.

## 목적

PyTorch 는 AI 엔지니어의 가장 표준적인 도구입니다. 텐서 연산·자동미분·`nn.Module`·`DataLoader`·체크포인트 같은 기본 요소를 손에 익히지 않으면 다른 모든 학습이 *공중에 뜬 이야기*가 됩니다. 이 권은 PyTorch 의 가장 자주 쓰이는 패턴을 *외울 수 있을 만큼* 반복하는 데 집중합니다.

## 선수 지식

- Volume 13, 17–22 완료
- 외부 지식: Python 클래스, 파일 입출력

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. PyTorch 텐서를 NumPy 와 자유롭게 변환하고 GPU 로 옮길 수 있습니다.
2. `nn.Module` 을 상속해 모델을 정의할 수 있습니다.
3. `DataLoader` 를 사용한 학습 루프를 작성할 수 있습니다.
4. 체크포인트를 저장하고 복원할 수 있습니다.
5. 혼합 정밀도(AMP) 학습을 적용할 수 있습니다.

## 챕터 목차

1. **텐서의 기본** — 생성·연산·자료형·디바이스
2. **자동미분 API** — `requires_grad`·`backward`·`detach`
3. **`nn.Module` 의 구조** — `__init__`·`forward`·파라미터
4. **`Dataset` 과 `DataLoader`** — 미니배치·셔플·워커
5. **표준 학습 루프** — 한 화면 안에 끝내는 형태
6. **검증 루프와 평가 지표 계산**
7. **체크포인트 저장과 복원** — `state_dict`
8. **혼합 정밀도(AMP)** — `torch.cuda.amp.autocast`
9. **다중 GPU 학습 첫걸음** — DataParallel vs DistributedDataParallel

## 자가점검 키워드

`Tensor`, `nn.Module`, `Dataset/DataLoader`, `학습 루프`, `state_dict`, `AMP`, `DDP`, `requires_grad`

## 다음 권

[Volume 37 — 데이터 파이프라인과 실험 관리](./volume_37_data_pipeline.md)
