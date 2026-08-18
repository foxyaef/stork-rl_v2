# 원본 SWF 분석 기록

분석 대상: `황새오래걷기 게임하기.swf`

- SWF 버전: 8 (ActionScript 2)
- 화면: 550×400
- 프레임 속도: 30 FPS
- 메인 타임라인: 8 프레임
- 조작: 왼쪽, 오른쪽, 입력 없음

## 원본 변수와 프로젝트 변수

| ActionScript | Python | 의미 |
|---|---|---|
| `body._rotation` | `body_rotation` | 실제 낙상 판정 각도 |
| `bs` | `balance_velocity` | 각도에 더해지는 균형 속도 |
| `bi` | `difficulty` | 매 프레임 증가하는 입력/중력 계수 |
| `xs` | `forward_speed` | 배경 이동 및 거리 속도 |
| `_root.xx` | `distance_pixels` | 누적 이동량 |
| `_root.metres` | `metres` | 화면에 표시되는 정수 거리 |

## 프레임별 판정 순서

1. 이전 프레임의 `xs`를 누적 거리에 더한다.
2. `floor(xx / 550 * 3)`으로 표시 거리를 계산한다.
3. `xs = bi * 3`으로 다음 프레임 속도를 정한다.
4. `bi += 0.002`.
5. 왼쪽이면 `bs -= bi`, 오른쪽이면 `bs += bi`.
6. 기존 몸통 각도가 `< -110` 또는 `> 80`이면 낙상한다.
7. 생존 중이면 `bs += (body_rotation / 3) * bi / 100`.
8. `body_rotation += bs`.

`StorkEnv.step()`은 이 순서를 그대로 구현한다. 보상은 같은 누적 이동량의
연속값을 m로 환산한 값이다. 표시 점수의 정수 버림 때문에 학습 신호가
불필요하게 희소해지는 것만 피했으며, 최적화 목표 자체는 원작과 같다.

