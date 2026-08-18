from __future__ import annotations

import argparse
from statistics import mean

from stork_rl import StorkEnv
from stork_rl.agents import heuristic_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless heuristic benchmark.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=18_000)
    args = parser.parse_args()

    env = StorkEnv(render_mode=None, max_steps=args.max_steps)
    scores: list[int] = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=episode)
        while True:
            observation, _, terminated, truncated, info = env.step(
                heuristic_action(observation)
            )
            if terminated or truncated:
                scores.append(info["metres"])
                break
    env.close()
    print(f"episodes={len(scores)} mean={mean(scores):.1f}m best={max(scores)}m")


if __name__ == "__main__":
    main()

