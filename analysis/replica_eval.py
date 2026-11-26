"""Evaluate agents in the live-loop replica (see live_replica.py).

The replica simulates the real pipeline in wall-clock time: game frames at
30 FPS, capture, noisy neck measurement, the live state estimator, the
prediction horizon, the policy and a blocking key press. It reproduced the
live PPO v9 result with the 0.8 estimator (replica about 51 m, real 49.4 m)
and was used to find which part of the loop limits the distance.

    python analysis/replica_eval.py          # 120 games per row
    python analysis/replica_eval.py 30       # quicker check
"""

from __future__ import annotations

import statistics
import sys
from multiprocessing import Pool

from common import make_agent
from live_replica import LiveConfig, LiveReplica

SCREENCAST = dict(capture=(0.025, 0.040), image_lag=(0.040, 0.080))  # conservative streaming timing
ROWS = [
    ("PPO v9, 추정기 0.25/0.25, 스크린샷 (실제 약 25~28m, 로그 추정)", "ppo", dict(action_scale=0.25, physics_scale=0.25)),
    ("PPO v9, 추정기 0.8/0.25, 스크린샷 (실제 약 25m, 로그 추정)", "ppo", dict(action_scale=0.8, physics_scale=0.25)),
    ("PPO v9, 추정기 0.8/0.8, 스크린샷 (실제 49.4m)", "ppo", {}),
    ("규칙 기반 12/12, 스크린샷 (실제 42.1m)", "pd", {}),
    ("둔감형 규칙 16/20, 스크린샷 (실제 39.7m)", "pd:16:20", {}),
    ("PPO v9, 화면 스트리밍 (실제 409.2m)", "ppo", SCREENCAST),
    ("규칙 기반 12/12, 화면 스트리밍 (실제 483.8m)", "pd", SCREENCAST),
    ("기존 규칙, 화면 스트리밍 (실제 미측정)", "heuristic", SCREENCAST),
    ("개선 규칙, 화면 스트리밍 (실제 미측정)", "lookahead", SCREENCAST),
]
_agents: dict = {}


def run(job: tuple[str, dict, int]) -> int:
    name, config, seed = job
    if name not in _agents:
        _agents[name] = make_agent(name)
    metres, _seconds = LiveReplica(LiveConfig(**config), seed).run(_agents[name])
    return metres


def main(games: int) -> None:
    with Pool() as pool:
        for label, agent, config in ROWS:
            jobs = [(agent, config, seed) for seed in range(5000, 5000 + games)]
            metres = sorted(pool.map(run, jobs))
            reach = 100 * sum(m >= 200 for m in metres) / len(metres)
            print(
                f"{label}: replica mean {statistics.mean(metres):.1f} m, "
                f"median {statistics.median(metres):.0f} m, 200m+ {reach:.0f}%",
                flush=True,
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
