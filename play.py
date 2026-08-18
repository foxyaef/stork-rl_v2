from __future__ import annotations

import argparse
from pathlib import Path

from stork_rl import BrowserStorkEnv, StorkEnv
from stork_rl.agents import heuristic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch or manually play the simulator.")
    parser.add_argument(
        "--agent",
        choices=("heuristic", "manual", "model"),
        default="heuristic",
    )
    parser.add_argument("--profile", choices=("browser", "exact"), default="browser")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--max-steps", type=int, default=18_000)
    parser.add_argument("--projection-frames-min", type=int)
    parser.add_argument("--projection-frames-max", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.model is None) != (args.stats is None):
        raise ValueError("Pass --model and --stats together")
    if args.agent == "model" and args.model is None:
        candidates = (
            (
                Path("models_browser_v9/stork_ppo.zip"),
                Path("models_browser_v9/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v11/stork_ppo.zip"),
                Path("models_browser_v11/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v12/stork_ppo.zip"),
                Path("models_browser_v12/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v13/stork_ppo.zip"),
                Path("models_browser_v13/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v6/stork_ppo.zip"),
                Path("models_browser_v6/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v7/stork_ppo.zip"),
                Path("models_browser_v7/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v4/stork_ppo.zip"),
                Path("models_browser_v4/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v5/stork_ppo.zip"),
                Path("models_browser_v5/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v3/stork_ppo.zip"),
                Path("models_browser_v3/vec_normalize.pkl"),
            ),
            (
                Path("models_browser_v2/stork_ppo.zip"),
                Path("models_browser_v2/vec_normalize.pkl"),
            ),
            (
                Path("models_browser/stork_ppo.zip"),
                Path("models_browser/vec_normalize.pkl"),
            ),
            (Path("models/stork_ppo.zip"), Path("models/vec_normalize.pkl")),
        )
        for model_path, stats_path in candidates:
            if model_path.exists() and stats_path.exists():
                args.model, args.stats = model_path, stats_path
                break

    if (args.projection_frames_min is None) != (
        args.projection_frames_max is None
    ):
        raise ValueError("Pass both projection frame bounds together")
    projected_live_model = (
        args.model is not None
        and args.model.parent.name
        in {
            "models_browser_v9",
            "models_browser_v10",
            "models_browser_v11",
            "models_browser_v12",
            "models_browser_v13",
        }
    )
    if args.projection_frames_min is not None:
        projection_range = (
            args.projection_frames_min,
            args.projection_frames_max,
        )
    elif projected_live_model:
        projection_range = (4, 6)
    else:
        projection_range = (0, 0)

    env_class = BrowserStorkEnv if args.profile == "browser" else StorkEnv
    env_kwargs = {
        "render_mode": "human",
        "max_steps": args.max_steps,
        "realtime": True,
    }
    if args.profile == "browser":
        env_kwargs["observation_projection_frames_range"] = projection_range
    env = env_class(**env_kwargs)
    observation, info = env.reset()

    model = None
    normalizer = None
    if args.agent == "model":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        if (
            args.model is None
            or args.stats is None
            or not args.model.exists()
            or not args.stats.exists()
        ):
            raise FileNotFoundError(
                "Train first, or pass both --model and --stats paths."
            )
        wrapper_kwargs = dict(env_kwargs)
        wrapper_kwargs["render_mode"] = None
        wrapper_kwargs["realtime"] = False
        wrapper = DummyVecEnv([lambda: env_class(**wrapper_kwargs)])
        normalizer = VecNormalize.load(args.stats, wrapper)
        normalizer.training = False
        normalizer.norm_reward = False
        model = PPO.load(args.model)

    try:
        while not env.user_closed:
            if args.agent == "manual":
                action = env.poll_human_action()
            elif args.agent == "heuristic":
                action = heuristic_action(observation)
            else:
                normalized = normalizer.normalize_obs(observation[None, :])
                action, _ = model.predict(normalized, deterministic=True)
                action = int(action[0])

            observation, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(
                    f"Result: {info['metres']}m "
                    f"({info['distance_metres_exact']:.2f}m, {info['steps']} frames)"
                )
                observation, info = env.reset()
    finally:
        env.close()
        if normalizer is not None:
            normalizer.close()


if __name__ == "__main__":
    main()
