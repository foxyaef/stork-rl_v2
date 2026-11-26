"""Simulator comparison of rule-based controllers and PPO v9.

Reproduces the table "RL vs 규칙 기반" in LIVE_RESULTS.md: 200 fixed seeds of
BrowserStorkEnv in two timing conditions.

    python analysis/rule_compare.py            # 200 episodes per cell
    python analysis/rule_compare.py 40         # quicker check
"""

from __future__ import annotations

import statistics
import sys
from multiprocessing import Pool

from common import make_agent
from stork_rl import BrowserStorkEnv

CONDITIONS = {
    "실제형 조건 (판단 4~8프레임, 예측 4~6)": dict(
        decision_frames_range=(4, 8), observation_projection_frames_range=(4, 6)
    ),
    "빠른 루프 (판단 2~3프레임, 예측 2~3)": dict(
        decision_frames_range=(2, 3), observation_projection_frames_range=(2, 3)
    ),
}
AGENTS = {
    "heuristic": "기존 규칙 (1프레임 앞만 예측)",
    "lookahead": "개선 규칙 (2프레임 앞까지 예측)",
    "pd": "최종 규칙 (각도 + 12×각속도, 난이도 비례 불감대)",
    "ppo": "PPO v9",
}
_agents: dict = {}


def run(job: tuple[str, dict, int]) -> int:
    name, env_kwargs, seed = job
    if name not in _agents:
        _agents[name] = make_agent(name)
    act = _agents[name]
    env = BrowserStorkEnv(render_mode=None, max_steps=18_000, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(act(obs))
        if terminated or truncated:
            return int(info["metres"])


def main(episodes: int) -> None:
    with Pool() as pool:
        for condition, env_kwargs in CONDITIONS.items():
            print(f"## {condition}")
            for name, label in AGENTS.items():
                jobs = [(name, env_kwargs, seed) for seed in range(1000, 1000 + episodes)]
                metres = sorted(pool.map(run, jobs, chunksize=4))
                reach = 100 * sum(m >= 200 for m in metres) / len(metres)
                print(
                    f"{label}: mean {statistics.mean(metres):.1f} m, "
                    f"median {statistics.median(metres):.0f} m, 200m+ {reach:.0f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
