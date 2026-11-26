"""Which real-world imperfection breaks the simple rule controller?

Starts from perfect conditions (a decision every frame, no noise, no delay) and
adds one imperfection at a time to BrowserStorkEnv, driving it with
``heuristic_action`` (one-frame look-ahead rule). 40 seeds per case; episodes
stop at 18,000 frames (10 minutes = 5,595 m).

    python analysis/timing_decomposition.py
"""

from __future__ import annotations

import statistics
import sys
from multiprocessing import Pool

import common  # noqa: F401  (sets up sys.path)
from stork_rl import BrowserStorkEnv
from stork_rl.agents import heuristic_action

CLEAN = dict(
    decision_frames_range=(1, 1),
    pulse_frames_range=(1, 1),
    action_delay_frames_range=(0, 0),
    observation_delay_frames_range=(0, 0),
    missed_pulse_probability=0.0,
    angle_noise_std=0.0,
    velocity_noise_std=0.0,
    difficulty_noise_std=0.0,
)
CASES = [
    ("매 프레임 판단, 잡음과 지연 없음", {}),
    ("+ 관측 잡음 (각도 1°, 각속도 0.12)", dict(angle_noise_std=1.0, velocity_noise_std=0.12, difficulty_noise_std=0.003)),
    ("+ 강한 관측 잡음 (각도 5°, 각속도 1.0)", dict(angle_noise_std=5.0, velocity_noise_std=1.0, difficulty_noise_std=0.003)),
    ("+ 관측 지연 0~1프레임", dict(observation_delay_frames_range=(0, 1))),
    ("+ 입력 지연 0~1프레임", dict(action_delay_frames_range=(0, 1))),
    ("+ 키 누름 1~2프레임", dict(pulse_frames_range=(1, 2))),
    ("+ 키 누락 3%", dict(missed_pulse_probability=0.03)),
    ("판단 간격 2프레임", dict(decision_frames_range=(2, 2))),
    ("판단 간격 3프레임", dict(decision_frames_range=(3, 3))),
    ("판단 간격 4~8프레임", dict(decision_frames_range=(4, 8))),
]


def run(job: tuple[dict, int]) -> int:
    env_kwargs, seed = job
    env = BrowserStorkEnv(render_mode=None, max_steps=18_000, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(heuristic_action(obs))
        if terminated or truncated:
            return int(info["metres"])


def main(episodes: int) -> None:
    with Pool() as pool:
        for label, extra in CASES:
            kwargs = {**CLEAN, **extra}
            metres = pool.map(run, [(kwargs, seed) for seed in range(1000, 1000 + episodes)])
            print(
                f"{label}: mean {statistics.mean(metres):.0f} m, "
                f"median {statistics.median(metres):.0f} m, min {min(metres)}, max {max(metres)}",
                flush=True,
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
