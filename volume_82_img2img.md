# Volume 88 — Image-to-Image·Inpainting·Outpainting

> 이 권이 끝나면 *기존 이미지를 변환·복원·확장* 하는 디퓨전 응용 패턴 전체를 설명할 수 있게 됩니다.

## 목적

생성 모델의 가장 실용적인 응용 중 다수는 *완전한 새 이미지*가 아니라 *기존 이미지의 변형*입니다. Img2Img(스타일 전환), Inpainting(특정 영역 재생성), Outpainting(이미지 확장)은 모두 디퓨전의 *시작 노이즈와 마스크*를 조작하는 방법입니다. 이 권은 이 세 패턴의 원리와 실무 사용법을 정리합니다.

## 선수 지식

- Volume 80, 78 완료
- 외부 지식: 이미지 마스크의 직관

## 학습 결과

이 권을 마치면 다음을 할 수 있게 됩니다.

1. Img2Img 의 *Strength* 파라미터가 무엇을 통제하는지 설명할 수 있습니다.
2. Inpainting 의 마스크 처리·블렌딩 기법을 알 수 있습니다.
3. Outpainting 의 경계 일관성 문제를 인식합니다.
4. SDEdit·Differential Diffusion 같은 변형의 동기를 알 수 있습니다.
5. 산업 응용(가상 피팅·이미지 복원·배경 교체) 의 워크플로를 그릴 수 있습니다.

## 챕터 목차

1. **Img2Img 의 원리** — 노이즈 주입 + 디노이징
2. **Strength·Steps 파라미터**
3. **Inpainting** — 마스크 영역 재생성
4. **Outpainting** — 경계 확장
5. **SDEdit·Differential Diffusion**
6. **마스크 블렌딩 기법**
7. **응용 사례** — 가상 피팅·배경 교체·복원
8. **함정** — 경계 부조화·아이덴티티 변형

## 자가점검 키워드

`Img2Img`, `Strength`, `Inpainting`, `Outpainting`, `SDEdit`, `마스크 블렌딩`, `가상 피팅`, `경계 부조화`

## 다음 권

[Volume 80 — 비디오·3D 생성](./volume_83_video_3d.md)
