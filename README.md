# 황새 오래 걷기 강화학습 프로젝트

원본 Flash 게임(SWF)의 ActionScript 2 물리·거리·낙상 판정을 옮긴
Gymnasium 강화학습 환경입니다. 학습과 평가는 화면 없이 빠르게 실행하고,
결과를 확인할 때만 Pygame 시뮬레이터 또는 실제 웹 게임을 사용합니다.

현재 실제 브라우저 기본 모델은 **v9**입니다. v10~v15는 각각 다른 보상
실험 결과이며 모두 보존되어 있습니다. 거리만 보면 더 좋은 버전도 있지만,
실게임에서 중요했던 평균 각도와 `|vel|>1` 체류율을 함께 고려해 v9을 기본으로
유지합니다.

**실제 웹 게임 결과 (2026-11-26):** 화면 스트리밍 캡처(`--capture screencast`)와
상태 추정기 보정(키 효과와 중력 0.8배)을 적용한 뒤 PPO v9는 5판 모두 200m를
넘겼고(평균 409.2m, 최고 518m), 규칙 기반 제어기(`--agent pd`)도 5판 모두
넘겼습니다(평균 483.8m, 최고 584m). 실험 기록과 분석은
[LIVE_RESULTS.md](LIVE_RESULTS.md)에 있습니다.

> 이 GitHub 저장소에는 현재 권장 모델인 v9의 최종 정책과 정규화 통계만
> 포함합니다. 다른 버전의 모델, 체크포인트, TensorBoard 기록과 실게임 로그는
> 포함하지 않으며 모델 버전별 설계와 평가 결과만 문서에 보존합니다.

## 1. 현재 진행 상황

- SWF에서 물리 업데이트 순서, 거리 계산 및 낙상 기준 추출 완료
- 행동 3개(`NONE`, `LEFT`, `RIGHT`)와 관측 4개로 Gymnasium 환경 구현 완료
- 화면 없는 PPO 병렬 학습, 체크포인트 및 이어 학습 구현 완료
- 원본 이미지 기반 Pygame 시각화와 수동 플레이 구현 완료
- 실제 브라우저의 캡처·판단·입력 지연을 재현하는 `BrowserStorkEnv` 구현 완료
- 실제 웹 게임의 목 각도 검출, 각속도 추정, 미래 상태 보정 및 키 입력 구현 완료
- 물리·관측 혼합 상태 추정, 각도 이상치 거부 및 짧은 검출 누락 예측 구현 완료
- 실제 로그를 바탕으로 4~6프레임 미래 관측을 학습에 반영한 v9 제작 완료
- 중앙 안정, 연속 입력, `NONE`, 각도·각속도 개선 및 고각속도 회복 보상 실험 완료
- 로컬 작업 폴더에 v1~v15 모델과 각 모델의 관측 정규화 통계 보존 완료
- GitHub용 문서에 모델 파일 없이도 v1~v15 실험 이력과 평가 결과 보존 완료
- 실제 게임 로그 분석으로 상태 추정기 계수 보정 완료 (키 효과와 중력 0.25 → 0.8)
- Chrome 화면 스트리밍 캡처 추가 완료 (판단 한 번 117ms → 33ms)
- 비교용 규칙 기반 에이전트 추가 완료 (`lookahead`, `pd`)
- 실제 웹 게임 200m 달성 (PPO v9 5/5판, 규칙 기반 5/5판)
- 실제 파이프라인 재현 시뮬레이터와 분석 스크립트 추가 (`analysis/`)
- v16 (실측 판단 간격 반영 재학습) 시뮬레이터 평가 완료, 실제 게임은 미측정
- 현재 자동 선택 모델: `models_browser_v9`
- 테스트 상태: 40개 통과

원본 판정의 상세한 업데이트 순서는 [ORIGINAL_LOGIC.md](ORIGINAL_LOGIC.md),
버전별 실험 결과는 [TRAINING_RESULTS.md](TRAINING_RESULTS.md), 실제 웹 게임
실험은 [LIVE_RESULTS.md](LIVE_RESULTS.md)에 있습니다.

## 2. 환경 구성

### 행동

| 값 | 행동 | 의미 |
|---:|---|---|
| 0 | `NONE` | 방향키를 누르지 않음 |
| 1 | `LEFT` | 왼쪽 방향키 입력 |
| 2 | `RIGHT` | 오른쪽 방향키 입력 |

### 관측

정책에는 다음 4개 값이 전달됩니다.

| 순서 | 값 | 설명 |
|---:|---|---|
| 0 | `body_rotation` | 몸통 각도(도) |
| 1 | `balance_velocity` | 프레임당 각속도에 해당하는 원본 상태값 |
| 2 | `difficulty` | 시간이 지날수록 증가하는 난이도 계수 |
| 3 | `forward_speed` | 전진 속도 |

### 원본 판정

- 왼쪽 낙상: 몸통 각도 `< -110°`
- 오른쪽 낙상: 몸통 각도 `> 80°`
- 거리: 원본 픽셀 이동량을 `550px = 3m`로 변환
- 난이도: 프레임마다 `0.002` 증가
- 화면 및 물리 기준: 550×400, 30FPS
- 기본 에피소드 상한: 18,000프레임(10분). 원작의 실패 조건과 별개인 안전 상한

### 정확 환경과 브라우저 환경의 차이

`StorkEnv`는 원본처럼 한 스텝에 한 Flash 프레임을 진행합니다.
`BrowserStorkEnv`는 실제 실행 로그에 맞춰 한 번의 정책 결정 동안 다음을
랜덤화합니다.

- 원본 물리 4~8프레임 진행
- 선택한 방향키는 1~2프레임만 적용
- 입력 시작 지연 0~1프레임
- 관측 지연 0~1프레임
- 3% 확률로 키 펄스 누락
- 각도, 각속도 및 난이도 측정 잡음
- v9 이후 실환경 모델은 관측을 4~6프레임 미래로 랜덤 투영

실제 로그에서 의사결정 간격 중앙값은 약 166.3ms(5프레임), 캡처 시간은
약 101ms(3프레임), 방향키 펄스는 약 44.6ms(1~2프레임)였습니다. 따라서
실게임용 모델은 매 프레임 완벽히 입력할 수 있다고 가정하면 안 됩니다.

## 3. 프로젝트 구조

```text
assets/                         시뮬레이션용 배경과 황새 PNG 프레임
src/stork_rl/env.py             원본 1프레임 물리 환경
src/stork_rl/browser_env.py     지연·잡음·보상 래퍼
tests/                          물리 및 보상 단위 테스트
train.py                        PPO 신규/이어 학습
evaluate.py                     정확 환경 평가
evaluate_browser.py             브라우저 환경 통계 평가
play.py                         수동/규칙/모델 Pygame 실행
live_browser_test.py            실제 웹 게임 인식 및 입력
loop_stats.py                   실게임 CSV 로그 요약 (루프 속도, 판별 추정 거리)
analysis/                       재현 시뮬레이터와 RL vs 규칙 기반 분석 스크립트
experiments/live_games.csv      실제 게임 판별 기록
setup.ps1                       기본·학습·테스트 환경 설치
setup-live.ps1                  OpenCV·Playwright·Chromium 설치
requirements.txt                전체 기능에 필요한 직접 의존성 고정 버전
environments.txt                검증 환경의 전이 의존성까지 포함한 전체 스냅샷
TRAINING_RESULTS.md             v1~v16 학습·평가 기록
LIVE_RESULTS.md                 실제 웹 게임 실험 결과와 Sim-to-Real 분석
ORIGINAL_LOGIC.md               원본 ActionScript 판정 정리
```

로컬에서 생성되지만 GitHub에는 올라가지 않는 항목:

```text
models*/                        모델, 정규화 통계, 체크포인트, TensorBoard
logs/                           실게임 CSV 로그와 마지막 캡처 화면
.venv*/                         Python 가상환경
*.swf                           원본 또는 다운로드한 Flash 파일
```

예외적으로 다음 v9 파일 두 개는 GitHub에 포함합니다.

```text
models_browser_v9/stork_ppo.zip       현재 권장 PPO 정책
models_browser_v9/vec_normalize.pkl   v9 관측·보상 정규화 통계
```

## 4. 설치와 가상환경

### 준비물

- Windows PowerShell
- Python 3.10 이상
- 실제 검증 환경: Python 3.14.3 64비트
- 실게임 테스트 시 인터넷 연결 및 Chromium 설치 공간

프로젝트 경로에 한글이나 긴 경로가 있어도 PyTorch 설치가 실패하지 않도록
가상환경은 프로젝트 밖의 짧은 경로에 만듭니다.

### 권장 자동 설치

PowerShell에서 이 README가 있는 폴더로 이동한 뒤 실행합니다.

```powershell
.\setup.ps1
& "$env:USERPROFILE\.venvs\stork-rl\Scripts\Activate.ps1"
```

생성되는 가상환경:

```text
C:\Users\사용자명\.venvs\stork-rl
```

`setup.ps1`은 프로젝트, PPO 학습 모듈과 테스트 모듈을 설치합니다. 실게임까지
연결하려면 이어서 실행합니다.

```powershell
.\setup-live.ps1
```

이 명령은 OpenCV, Playwright와 Playwright용 Chromium을 설치합니다.

### requirements.txt로 설치

자동 설치 스크립트 대신 표준 `requirements.txt`를 사용할 수도 있습니다.

```powershell
py -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m playwright install chromium
```

마지막 Chromium 설치는 실제 웹 게임 제어를 사용할 때만 필요합니다.
`requirements.txt`는 직접 사용하는 패키지의 검증 버전을, `environments.txt`는
그 당시 설치된 하위 패키지까지 포함한 전체 스냅샷을 기록합니다.

### 현재와 동일한 버전으로 복원

정확히 같은 패키지 버전이 필요하면 새 가상환경에서 다음 순서로 설치합니다.

```powershell
$venvPath = Join-Path $env:USERPROFILE ".venvs\stork-rl"
py -3.14 -m venv $venvPath
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$venvPath\Scripts\python.exe" -m pip install -r .\environments.txt
& "$venvPath\Scripts\python.exe" -m pip install -e . --no-deps
& "$venvPath\Scripts\python.exe" -m playwright install chromium
& "$venvPath\Scripts\Activate.ps1"
```

Python 3.14가 없다면 `py -m venv $venvPath`를 사용해도 됩니다. 이 경우
플랫폼에 따라 일부 하위 모듈 버전이 달라질 수 있으므로 `setup.ps1` 방식이
더 안전합니다.

### 가상환경을 활성화하지 않고 실행

PowerShell 실행 정책 때문에 활성화가 막히면 전용 Python을 직접 호출합니다.

```powershell
& "$env:USERPROFILE\.venvs\stork-rl\Scripts\python.exe" train.py --help
```

현재 버전 확인:

```powershell
python --version
python -m pip list
python -c "import gymnasium, numpy, pygame, torch, stable_baselines3; print(gymnasium.__version__, numpy.__version__, pygame.version.ver, torch.__version__, stable_baselines3.__version__)"
```

## 5. 빠른 사용법

### 규칙 기반 에이전트 보기

```powershell
python play.py --agent heuristic --profile browser
```

### 키보드로 직접 플레이

```powershell
python play.py --agent manual --profile exact
```

왼쪽/오른쪽 방향키로 조작하고 창을 닫으면 종료됩니다.

### 현재 권장 v9 시뮬레이션

v9 모델과 정규화 통계는 저장소에 포함되므로 clone 직후 실행할 수 있습니다.

```powershell
python play.py --agent model --profile browser `
  --model models_browser_v9\stork_ppo.zip `
  --stats models_browser_v9\vec_normalize.pkl
```

모델 경로를 생략하면 `play.py`와 `live_browser_test.py`가 v9을 먼저 찾습니다.

> 중요: `stork_ppo.zip`과 `vec_normalize.pkl`은 반드시 같은 모델 폴더의
> 파일끼리 사용해야 합니다. 서로 다른 버전의 정규화 통계를 섞으면 예측값이
> 크게 달라질 수 있습니다.

## 6. 화면 없이 학습

### 새 브라우저 모델 학습

```powershell
python train.py --profile browser --steps 500000 --envs 8 `
  --seed 7 --learning-rate 0.0003 `
  --output models_browser_experiment
```

### 원본 1프레임 환경 학습

```powershell
python train.py --profile exact --steps 1000000 --envs 8 `
  --output models_exact_experiment
```

### 기존 모델에서 이어 학습

```powershell
python train.py --profile browser --steps 500000 --envs 8 `
  --learning-rate 0.0001 `
  --projection-frames-min 4 --projection-frames-max 6 `
  --resume-model models_browser_v9\stork_ppo.zip `
  --resume-stats models_browser_v9\vec_normalize.pkl `
  --output models_browser_experiment
```

이어 학습에서도 모델과 정규화 통계를 함께 지정해야 합니다. `--output`은 원본
폴더와 다른 새 폴더를 권장합니다. PPO는 `envs × 2048` 단위의 롤아웃을
수집하므로 실제 학습 스텝은 요청값보다 조금 많을 수 있습니다.

학습 결과:

```text
출력폴더/stork_ppo.zip              정책 모델
출력폴더/vec_normalize.pkl          관측·보상 정규화 통계
출력폴더/checkpoints/               약 5만 스텝 간격 중간 저장
출력폴더/tensorboard/               학습 로그
```

### 학습 상황 보기

```powershell
tensorboard --logdir models_browser_experiment\tensorboard
```

출력된 주소(일반적으로 `http://localhost:6006`)를 브라우저에서 엽니다.

## 7. 보상 함수와 수정 방법

실제 보상 계산은 `src/stork_rl/browser_env.py`, 명령행 옵션은 `train.py`에
있습니다. 기본 총보상은 다음 항목의 합입니다.

```text
총보상 = 거리 증가
       + 중앙 안정 보상
       + 입력 직후 NONE 보상
       + |각도| 감소 보상
       + |각속도| 감소 보상
       + 저각속도 보상(기본 0)
       + 안전선 복귀 보상(기본 0)
       + 고각속도 회복 이벤트 보상
       - 방향키 사용 비용
       - 같은 방향 연속 입력 감점
       - 즉시 반대 방향 입력 감점
       - 안전선 초과 감점(기본 0)
       - 안전선 돌파 감점(기본 0)
       - 고각속도 진입 감점
       - 낙상 감점
```

### 현재 기본 보상값

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--center-reward-per-frame` | 0.015 | 각도와 각속도가 중앙에 가까울수록 프레임별 보상 |
| `--action-frame-penalty` | 0.002 | 방향키가 실제 적용된 프레임별 비용 |
| `--same-direction-penalty` | 0.06 | 같은 방향 연속 선택 누적 감점 |
| `--coast-after-input-reward` | 0.04 | 방향 입력 다음 결정에서 안정적으로 `NONE` 선택 시 보상 |
| `--immediate-reversal-penalty` | 0.04 | `NONE` 없이 즉시 반대키 입력 시 감점 |
| `--angle-progress-reward-scale` | 0.06 | `|angle|`을 실제로 줄인 정도의 보상 |
| `--velocity-progress-reward-scale` | 0.10 | 키 입력으로 `|vel|`을 실제로 줄인 정도의 보상 |
| `--safe-velocity-limit` | 1.0 | 선택형 안전 보상의 각속도 기준 |
| `--low-velocity-reward-per-frame` | 0.0 | 안전선 안에서 0에 가까울수록 주는 보상 |
| `--velocity-excess-penalty-per-frame` | 0.0 | 안전선 초과량의 제곱 감점 |
| `--velocity-crossing-penalty` | 0.0 | 안전선 안에서 밖으로 처음 넘어갈 때 감점 |
| `--velocity-recovery-reward-scale` | 0.0 | 안전선 초과량을 줄였을 때 보상 |
| `--high-velocity-threshold` | 1.5 | 고각속도 위험 구간 진입 기준 |
| `--recovered-velocity-threshold` | 1.0 | 위험 구간 회복 완료 기준 |
| `--high-velocity-entry-penalty` | 0.05 | 위험 구간에 들어갈 때 한 번 감점 |
| `--high-velocity-recovery-bonus` | 0.04 | 이후 회복했을 때 한 번 보상 |
| `--fall-penalty` | 1.0 | 낙상 시 추가 감점 |

각도·각속도 개선 보상은 악화에 별도 벌점을 주지 않고 실제로 감소한 경우만
보상합니다. 각속도 개선 보상은 정책이 방향키를 누른 경우에만 발생하므로,
자연스럽게 느려진 상태를 자신의 제어 성과로 받을 수 없습니다. 낙상 시 원본이
각속도를 0으로 초기화하지만 이것도 회복으로 계산하지 않습니다.

v13의 회복 이벤트는 `|vel|>1.5` 진입 감점 0.05와 `|vel|<1.0` 회복 보너스
0.04를 한 위험 구간에 각각 한 번만 적용합니다. 코드에서 회복 보너스는 진입
감점보다 클 수 없도록 제한하므로 일부러 각속도를 키우는 한 주기는 항상
순손실입니다.

### 보상 실험 권장 절차

1. 기준 모델을 덮어쓰지 말고 새 출력 폴더를 정합니다.
2. 한두 개 옵션만 바꿔 원인을 분리합니다.
3. 동일한 기준 모델과 정규화 통계에서 이어 학습합니다.
4. 고정 시드 200개로 기준 모델과 새 모델을 모두 평가합니다.
5. 평균 거리뿐 아니라 p10, 평균 각도, `|vel|>1` 체류율, 위험 진입과 회복률을 비교합니다.
6. 실제 웹에서는 먼저 입력 없는 모드로 확인한 뒤 제어를 켭니다.

예: 회복 보상을 더 약하게 실험하려면 다음처럼 새 폴더에 학습합니다.

```powershell
python train.py --profile browser --steps 500000 --envs 8 `
  --learning-rate 0.0001 `
  --projection-frames-min 4 --projection-frames-max 6 `
  --high-velocity-entry-penalty 0.05 `
  --high-velocity-recovery-bonus 0.02 `
  --resume-model models_browser_v9\stork_ppo.zip `
  --resume-stats models_browser_v9\vec_normalize.pkl `
  --output models_browser_recovery_test
```

새 보상 항목 자체를 추가하려면 다음 세 곳을 함께 수정합니다.

1. `BrowserStorkEnv.__init__`: 기본값, 범위 검증 및 멤버 저장
2. `BrowserStorkEnv.step`: 보상 계산과 `info` 진단값 추가
3. `train.py`: `argparse` 옵션과 `env_kwargs` 전달 추가

이후 `tests/test_browser_env.py`에 보상이 정확히 한 번만 발생하는지, 낙상이나
의도적 반복으로 악용되지 않는지 테스트를 추가합니다.

## 8. 평가와 테스트

### 전체 단위 테스트

```powershell
pytest -q
```

캐시 파일 없이 실행:

```powershell
python -m pytest -q -p no:cacheprovider
```

### 정확 환경 평가

```powershell
python evaluate.py --episodes 100
```

### 브라우저형 환경 평가

현재 권장 v9:

```powershell
python evaluate_browser.py --episodes 200 `
  --model models_browser_v9\stork_ppo.zip `
  --stats models_browser_v9\vec_normalize.pkl `
  --projection-frames-min 4 --projection-frames-max 6
```

비교용 v13:

```powershell
python evaluate_browser.py --episodes 200 `
  --model models_browser_v13\stork_ppo.zip `
  --stats models_browser_v13\vec_normalize.pkl `
  --projection-frames-min 4 --projection-frames-max 6
```

규칙 기반 에이전트 비교 (`heuristic`, `lookahead`, `pd`):

```powershell
python evaluate_browser.py --episodes 200 --agent pd `
  --projection-frames-min 4 --projection-frames-max 6
```

평가기는 거리의 평균·중앙값·p10·p90, 생존 시간, 방향키 사용률, 연속 입력,
즉시 반전, 입력 후 `NONE`, 평균 절대 각도, 각도·각속도 교정 성공률,
고각속도 체류율과 위험 구간 회복률을 출력합니다.

## 9. 실제 웹 게임 테스트

대상 페이지: <https://vidkidz.tistory.com/2825>

최초 한 번 실게임 모듈을 설치합니다.

```powershell
.\setup-live.ps1
```

### 1단계: 입력 없는 인식 확인

```powershell
python live_browser_test.py `
  --model models_browser_v9\stork_ppo.zip `
  --stats models_browser_v9\vec_normalize.pkl
```

브라우저에서 게임을 직접 시작한 뒤 황새가 나타나면 `F8`을 눌러 인식을
시작합니다. 이 모드에서는 방향키를 보내지 않습니다. 디버그 창에서 다음을
확인합니다.

- 오른쪽으로 기울 때 측정 각도가 `+`
- 왼쪽으로 기울 때 측정 각도가 `-`
- 직립 상태가 약 0도
- 목 검출선이 실제 목을 안정적으로 따라가는지
- `vel` 값이 프레임 사이에서 과도하게 튀지 않는지

부호가 반대면 `--angle-scale -0.75`, 중심이 어긋나면 `--angle-offset 값`을
사용합니다.

### 2단계: 실제 키 입력

```powershell
python live_browser_test.py --control `
  --model models_browser_v9\stork_ppo.zip `
  --stats models_browser_v9\vec_normalize.pkl
```

`F8` 전에는 입력하지 않으며 `Q`로 종료합니다. 즉시 입력을 시작하려면
`--arm-immediately`를 추가할 수 있지만, 새 보정값이나 새 모델은 먼저 인식
전용 모드로 확인하는 것이 안전합니다.

### 권장 실행: 화면 스트리밍 (200m 달성 설정)

```powershell
# PPO v9 (기본 모델 자동 선택)
python live_browser_test.py --control --capture screencast --debug-every 3 --log logs\cast_v9.csv

# 규칙 기반 제어기 (모델 없이 동작)
python live_browser_test.py --control --capture screencast --debug-every 3 --agent pd --log logs\cast_pd.csv
```

`--capture screencast`는 매번 스크린샷을 요청하는 대신(약 95ms) Chrome DevTools
Protocol의 `Page.startScreencast`로 크롬이 그리는 프레임을 스트리밍으로 받아
판단 한 번을 약 33ms(1 Flash 프레임)로 줄입니다. 이 설정에서 PPO v9는 평균
409.2m, 규칙 기반 제어기는 평균 483.8m를 기록했습니다. 로그는 실행할 때마다
덮어쓰므로 실험마다 `--log` 이름을 바꾸고, 실행 뒤 다음 명령으로 루프 속도와
판별 추정 거리를 확인합니다(공식 기록은 화면의 거리).

```powershell
python loop_stats.py logs\cast_v9.csv
```

### 상태 추정기

실게임 측정값은 모델로 바로 전달되지 않습니다. 상태 추정기는 이전 각도와
각속도를 먼저 진행한 뒤 새 측정값과 비교합니다.

- 신뢰도가 높은 정상 측정은 빠르게 반영
- 물리·운동 예측과 지나치게 다른 각도는 `outlier`로 거부
- 검출이 잠시 끊기면 최대 450ms 동안 `predicted` 상태 사용
- 450ms를 넘기면 `stale`로 전환하고 실제 방향키 입력 중지
- 800ms 이상 지난 뒤 정상 측정이 들어오면 해당 값으로 다시 동기화
- 2초 이상 `stale` 뒤 신뢰도 높은 정상 자세가 나타나면 새 게임으로 판정
- 새 게임에서는 난이도·전진 속도와 세션 경과 시간을 `1.0`·`0`부터 초기화
- 실제로 보낸 방향키 펄스와 측정 사이의 중력은 실로그에서 측정한 크기(원본 물리의
  약 0.8배)로 반영. 이전 기본값 0.25에서는 같은 키를 연타하는 문제가 있었음

디버그 창의 `state`는 `measured`, `outlier`, `predicted`, `stale` 중 하나이며,
`residual`은 측정 각도와 사전 예측 각도의 차이입니다. 최근 실패 로그를
재생했을 때 `+53.7° → +14.4°`의 비정상 점프는 `outlier`로 거부됐습니다.

### 주요 실환경 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--extra-latency-ms` | 35 | 캡처·처리 외에 예상할 입력 지연 |
| `--pulse-ms` | 38 | 방향키를 누르는 시간. 1 Flash 프레임은 약 33.3ms |
| `--angle-scale` | 0.75 | 화면 목 각도를 원본 몸통 상태로 바꾸는 비율 |
| `--angle-offset` | 0 | 중립 자세 보정값 |
| `--min-confidence` | 0.25 | 검출 신뢰도가 낮을 때 입력하지 않는 기준 |
| `--max-prediction-gap-ms` | 450 | 측정 누락 시 예측 상태로 제어할 최대 시간 |
| `--reacquire-after-ms` | 800 | 장시간 누락 후 새 측정값으로 재동기화할 시간 |
| `--innovation-gate-deg` | 10 | 측정 이상치 판정의 기본 허용 각도 |
| `--confidence-gate-deg` | 18 | 신뢰도에 따라 추가되는 허용 각도 |
| `--estimator-action-scale` | 0.8 | 키 펄스 효과를 추정 상태에 반영하는 비율 |
| `--estimator-physics-scale` | 0.8 | 측정 사이 원본 물리 가속을 혼합하는 비율 |
| `--session-reset-gap-ms` | 2000 | 이 시간 이상 공백 뒤 정상 황새가 나타나면 새 게임으로 초기화 |
| `--clip` | 자동 | 게임 영역 자동 탐색 실패 시 `x,y,w,h` 지정 |
| `--load-wait` | 12 | 페이지 로딩 대기 시간(초) |
| `--capture` | screenshot | `screencast`: Chrome 화면 스트리밍 (권장, 판단 약 33ms) |
| `--screencast-quality` | 70 | 스트리밍 JPEG 품질 |
| `--screencast-max-width` | 1280 | 스트리밍 프레임 최대 너비 |
| `--screencast-age-ms` | 20 | 도착한 프레임의 추정 나이 (측정 시각 = 도착 시각 − 이 값) |
| `--debug-every` | 1 | 디버그 창을 N번에 한 번만 그림 (3 권장) |
| `--agent` | model | `model`(PPO), `heuristic`, `lookahead`, `pd` |
| `--pd-gain` | 12 | 규칙 기반: `각도 + gain × 각속도`의 각속도 가중치 |
| `--pd-deadband` | 12 | 규칙 기반: 불감대 = deadband × 난이도 |
| `--fast-capture`, `--fixed-clip` | 꺼짐 | JPEG 스크린샷 (효과 작음) |
| `--overlap-key` | 꺼짐 | 사용 금지: 키 누름이 약 97ms로 두 배가 되어 거리 급감 |

실행 기록은 `logs/live_test.csv`, 마지막 화면은 `logs/live_test.last.png`에
저장됩니다. 실제와 시뮬레이션 차이를 다시 조정할 때 이 로그의 측정 각도,
필터 각도, 미래 예측 각도, 각속도, 행동, 캡처 시간과 펄스 시간을 사용합니다.
새 로그에는 `measurement_accepted`, `innovation_deg`, `state_source`,
`measurement_age_ms`, `session_id`, `session_elapsed_s`, `session_reset`도
기록되어 어떤 측정이 거부됐고 예측 상태를 얼마나 오래 사용했는지, 어느
시점에 새 게임으로 초기화됐는지 확인할 수 있습니다.

## 10. 버전 이력과 모델 선택

| 버전 | 핵심 변경 | 상태 |
|---|---|---|
| 기존 | 원본처럼 매 프레임 행동하는 정확 환경 모델 | 비교용 |
| v1 | 실제 입력 간격과 펄스를 반영한 첫 브라우저 모델, 50만 | 보존 |
| v2 | v1에서 약 100만 추가 학습 | 보존 |
| v3 | 중앙 안정 보상과 같은 방향 연속 입력 감점 강화 | 보존 |
| v4 | 입력 뒤 `NONE` 보상, 즉시 반전 감점 | 이전 메인 |
| v5 | v4 보상으로 처음부터 약 250만 학습 | 비교용 |
| v6 | 실제 `|angle|`, `|vel|` 감소량 보상 | 시뮬레이션 권장 |
| v7 | `vel=0.75` 선제 안전선 보상·감점 | 안전성 비교용 |
| v8 | v6에서 약 100만 추가 학습 | 비교용 |
| v9 | 실제 과대 예측을 재현하는 4~6프레임 미래 관측 | **실게임 권장/현재 메인** |
| v10 | 중앙·각도·각속도 보상을 소폭 강화 | 비교용 |
| v11 | v10에서 약 100만 추가 학습 | 비교용 |
| v12 | `|vel|<1`에서 작은 저각속도 보너스 | 비교용 |
| v13 | `1.5 초과 → 1.0 미만` 회복 이벤트 보상, 약 50.8만 | 비교용 |
| v14 | 빠른 중앙 접근 시작 상태와 중앙 통과 전 감속 보상, 약 101.6만 | 실게임 시험용 |
| v15 | 위험 접근 `NONE` 감점과 감속·통과 보상 강화, 약 50.8만 | 실패 분석용 |
| v16 | v9에서 실측 판단 간격(3~5프레임), 관측 지연 최대 2로 50만 추가 학습 | 시뮬레이터 75.1m (v9 72.1m), 실게임 미측정 |

실환경형 동일 200개 시드의 주요 결과:

| 모델 | 평균 거리 | 중앙 거리 | p10 | 평균 생존 | `|vel|>1` 체류 |
|---|---:|---:|---:|---:|---:|
| v9 | 55.6m | 53.5m | 27m | 46.08초 | 40.0% |
| v10 | 59.4m | 57.5m | 24m | 47.80초 | 42.0% |
| v11 | 60.9m | 60.0m | 25m | 48.61초 | 42.0% |
| v12 | 61.9m | 60.0m | 31m | 49.37초 | 41.4% |
| v13 | 59.6m | 59.0m | 30m | 48.24초 | 41.1% |
| v14 | 58.0m | 57.0m | 28m | 47.27초 | 40.6% |
| v15 | 56.5m | 53.5m | 27m | 46.56초 | 40.8% |

v13은 v9보다 거리는 개선됐지만 위험 진입은 결정 100회당 12.62회에서
12.71회로, `|vel|>1` 체류는 40.0%에서 41.1%로 소폭 늘었습니다. 회복률도
96.6%에서 96.7%로 거의 같았습니다. 이 때문에 실제 플레이 안정성을 우선해
v9을 계속 기본 모델로 사용합니다.

v14는 깨끗한 실게임 로그에서 공통으로 나타난 "빠르게 중앙으로 접근하면서
`NONE`을 선택하고, 중앙을 지난 뒤에야 반대키로 제동"하는 패턴을 겨냥했습니다.
v9에서 1,015,808 결정을 추가 학습했으며 시작 상태는 일반 60%, 중앙으로 빠르게
접근 30%, 큰 각도·고각속도 복구 10%입니다. 200개 동일 시드에서 빠른 중앙 통과는
결정 100회당 11.78회에서 11.10회, 즉시 반전은 32.0%에서 28.6%로 감소했고,
입력 뒤 `NONE`은 47.6%에서 51.6%, 평균 거리는 56.4m에서 58.0m로 증가했습니다.
다만 평균 절대 각도와 `|vel|>1` 체류율은 각각 10.75도에서 11.60도,
39.5%에서 40.6%로 악화되어 자동 기본 모델은 실제 웹 시험 전까지 v9로 유지합니다.

v14 실게임 시험 명령:

```powershell
python live_browser_test.py --control `
  --model models_browser_v14\stork_ppo.zip `
  --stats models_browser_v14\vec_normalize.pkl `
  --log logs\live_v14.csv
```

각속도 안정성을 더 우선한 중간 후보는
`models_browser_v14/checkpoints/stork_ppo_4363232_steps.zip`과 같은 번호의
`stork_ppo_vecnormalize_4363232_steps.pkl`입니다.

v15는 v14의 `NONE` 증가가 실제 제동 학습이 아닐 가능성을 검증하기 위해 v9에서
다시 시작했습니다. 감속 성공 보상과 과속 중앙 통과 감점을 각각 0.06으로 높이고,
8프레임 이내 중앙 통과가 예상되는 위험 접근에서 `NONE`을 선택하면 0.02 감점과
함께 기존 coast 보상을 차단했습니다. 그러나 200개 동일 시드에서 제동 누락은
v9의 결정 100회당 4.63회에서 5.77회, 입력 뒤 `NONE`은 47.6%에서 52.0%로
증가했습니다. 빠른 중앙 통과는 11.78회에서 11.26회로 감소했지만 평균 절대
각도와 `|vel|>1` 체류도 11.65도와 40.8%로 악화됐습니다. 따라서 추가 학습은
중단하고 v15는 실패 분석용으로만 보존합니다.

## 11. 문제 해결

### `pygame` 설치 중 빌드 도구 오류

이 프로젝트는 `pygame` 소스 빌드 대신 Windows 휠을 제공하는 `pygame-ce`를
사용합니다. `setup.ps1`을 다시 실행하고 `pygame`이 아니라 `pygame-ce`가
설치되는지 확인합니다.

### 모델 로딩 후 행동이 이상함

- 모델 ZIP과 `vec_normalize.pkl`의 폴더가 같은지 확인
- v9~v15는 `--projection-frames-min 4 --projection-frames-max 6`으로 평가
- 실게임의 `--angle-scale`, `--angle-offset`과 각도 부호 확인
- 새 로그에서 측정 각도와 예측 각도의 차이 확인

### 학습이 중간에 중단됨

`출력폴더/checkpoints`의 같은 스텝 모델 ZIP과 `vecnormalize` 파일을 짝으로
사용해 이어 학습할 수 있습니다. 마지막 완성 파일이 있다면 그것을 우선합니다.

### Playwright 브라우저가 없음

```powershell
python -m playwright install chromium
```

### 메모리가 부족함

`--envs 8`을 `--envs 4` 또는 `--envs 2`로 줄입니다. 학습 속도는 느려지지만
환경 동작과 보상 정의는 바뀌지 않습니다.

## 12. 재현성과 주의사항

- `--seed`를 같게 해도 병렬 처리 및 PyTorch 연산 때문에 완전히 동일한 모델이
  보장되지는 않으므로 최종 비교는 충분한 수의 고정 평가 시드로 수행합니다.
- 학습 중에는 `render_mode=None`이라 이미지나 창을 만들지 않습니다.
- Pygame 렌더링은 원본 자산을 사용하지만 Flash 무비클립 보간과 회전 방식
  차이 때문에 픽셀 단위로 완전히 같지는 않습니다.
- 상태 전이, 거리와 낙상 판정은 원본 ActionScript 순서를 따릅니다.
- 실제 웹 게임은 캡처 지연, 프레임 드롭, 목 검출 오차, 운영체제 키 입력 지연이
  존재하므로 시뮬레이션 점수가 실제 성능을 그대로 보장하지 않습니다.
- 직접 의존성은 [requirements.txt](requirements.txt), 전체 패키지 스냅샷은
  [environments.txt](environments.txt)에 기록되어 있습니다.
- `assets/`의 이미지와 게임 명칭에 관한 권리는 원 저작권자에게 있습니다.
  공개 배포 전 자산 재배포 권한을 확인하고, 권한이 불분명하면 해당 이미지를
  제외하거나 직접 제작한 대체 자산으로 교체해야 합니다.
- 별도 `LICENSE` 파일은 아직 선택하지 않았습니다. 오픈소스로 배포하려면 본인이
  원하는 이용 조건에 맞는 라이선스를 선택해 추가하세요.

## 13. GitHub 게시 준비 상태

이 폴더 자체가 Git 저장소 루트입니다. `.gitignore`는 다음 대용량·로컬 파일을
커밋 대상에서 제외합니다.

- v9 최종 모델 두 파일을 제외한 모든 `models*` 폴더와 학습 체크포인트
- `logs/`의 CSV 및 캡처 이미지
- `.venv*` 가상환경과 Python 캐시
- TensorBoard 및 테스트 실행 결과
- 다운로드한 SWF 파일과 로컬 비밀 설정

업로드 전에는 다음 명령으로 포함될 파일을 확인합니다.

```powershell
git status --short
git status --ignored --short
```

첫 번째 목록에는 소스, 문서, 테스트와 시뮬레이션용 PNG 자산만 보여야 합니다.
두 번째 명령에서 모델 폴더가 `!!`로 표시되면 정상적으로 제외된 것입니다.
