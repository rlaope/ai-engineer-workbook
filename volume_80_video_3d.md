# Volume 80 — 비디오·3D 생성

> 이 권이 끝나면 *왜 비디오 생성이 이미지 생성보다 훨씬 어려운가* 를 시간 일관성 관점에서 설명할 수 있게 됩니다.

## 목적

비디오는 *프레임 사이의 시간 일관성*을 유지해야 하므로 단순히 이미지 디퓨전을 N 번 반복하는 것으로는 만들어지지 않습니다. 3D 생성은 *여러 시점에서 일관된 형상*을 요구합니다. Sora·Runway·Pika 같은 비디오 모델, NeRF·Gaussian Splatting 같은 3D 표현이 이 분야의 표준입니다. 이 권은 차세대 생성 모델의 큰 그림을 정리합니다.

## 선수 지식

- Volume 46 완료
- 외부 지식: 동영상 = 시간축의 이미지 시퀀스

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. 비디오 디퓨전의 *시공간 어텐션* 발상을 그릴 수 있습니다.
2. Sora·Runway Gen-3·Pika 의 차별점을 알 수 있습니다.
3. NeRF 의 볼륨 렌더링 직관을 설명할 수 있습니다.
4. Gaussian Splatting 이 NeRF 의 *훈련·렌더 속도 한계*를 어떻게 풀었는지 알 수 있습니다.
5. 3D 자산 생성 워크플로를 인식합니다.

## 챕터 목차

1. **비디오 생성의 어려움** — 시간 일관성
2. **시공간 어텐션** — Spatial-Temporal
3. **Stable Video Diffusion·AnimateDiff**
4. **Sora·Runway Gen-3·Pika·Veo**
5. **NeRF** — Neural Radiance Fields
6. **Gaussian Splatting** — 빠른 표현
7. **TripoSR·Trellis** — 단일 이미지 → 3D
8. **응용** — 가상 캐릭터·디지털 트윈·게임 자산

## 자가점검 키워드

`시공간 어텐션`, `Sora`, `Runway`, `Pika`, `NeRF`, `Gaussian Splatting`, `TripoSR`, `Trellis`

## 다음 권

[Volume 81 — KV 캐시 깊이](./volume_81_kv_cache.md)
