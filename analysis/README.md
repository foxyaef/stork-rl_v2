# 분석 스크립트

[LIVE_RESULTS.md](../LIVE_RESULTS.md)의 표를 다시 만드는 스크립트입니다. 저장소
루트에서 실행하며, 학습용 가상환경(`gymnasium`, `numpy`)이면 충분합니다. PPO 정책은
`np_policy.py`가 저장된 `models_browser_v9`를 numpy로 직접 계산하므로 torch가
필요 없습니다(SB3 `predict(..., deterministic=True)`와 같은 행동).

| 파일 | 내용 | 실행 시간 |
|---|---|---|
| `rule_compare.py` | 시뮬레이터 200판: 기존 규칙, 개선 규칙, 최종 규칙(PD), PPO v9를 판단 4~8프레임과 2~3프레임 조건에서 비교 | 약 1분 |
| `timing_decomposition.py` | 완벽한 조건에서 잡음, 지연, 판단 간격을 하나씩 추가해 단순 규칙이 무엇에 무너지는지 확인 | 약 3분 |
| `live_replica.py` | 실제 파이프라인 재현기: 30FPS 게임, 캡처, 측정 잡음, 실게임 상태 추정기, 예측 지평, 정책, 키 입력을 실제 시간 단위로 시뮬레이션 | - |
| `replica_eval.py` | 재현기로 추정기 계수, 캡처 방식(스크린샷, 스트리밍), 에이전트별 거리를 비교 | 수 초 |
| `common.py` | 경로 설정과 에이전트 생성 (`ppo`, `heuristic`, `lookahead`, `pd`, `pd:<gain>:<deadband>`) | - |
| `np_policy.py` | torch 없이 SB3 PPO와 VecNormalize를 불러오는 numpy 구현 | - |

```powershell
python analysis\rule_compare.py
python analysis\rule_compare.py 40       # 판 수를 줄여 빠르게 확인
python analysis\timing_decomposition.py
python analysis\replica_eval.py
```

## 확인된 결과

`rule_compare.py` (200판):

```text
실제형 조건 (판단 4~8프레임): 기존 11.1m, 개선 29.4m, 최종 규칙 56.0m, PPO v9 56.4m
빠른 루프 (판단 2~3프레임):   기존 46.8m, 개선 151.6m, 최종 규칙 290.8m, PPO v9 285.5m
```

`replica_eval.py` (120판, 괄호는 실제 게임):

```text
PPO v9, 추정기 0.25/0.25, 스크린샷   17.4m  (약 25m, 로그 추정)
PPO v9, 추정기 0.8/0.25, 스크린샷    34.5m  (약 25m, 로그 추정)
PPO v9, 추정기 0.8/0.8, 스크린샷     51.4m  (49.4m)
규칙 기반, 스크린샷                  52.4m  (42.1m)
둔감형 규칙, 스크린샷                53.1m  (39.7m)
PPO v9, 화면 스트리밍               146.5m  (409.2m)
규칙 기반, 화면 스트리밍            176.4m  (483.8m)
```

재현기는 기준 설정(v9, 추정기 0.8)을 잘 맞혔고 캡처 속도가 결정적이라는 방향도
맞혔지만, 스트리밍 조건은 크게 낮게, 느린 캡처의 규칙 기반은 높게 예측합니다.
수치보다는 원인 분석용으로 사용하세요.
