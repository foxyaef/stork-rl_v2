from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from stork_rl import BrowserStorkEnv
from stork_rl.agents import heuristic_action, lookahead_action, pd_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy under randomized browser timing."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--agent", choices=("model", "heuristic", "lookahead", "pd"), default="model"
    )
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-weight", type=float, default=16.0)
    parser.add_argument("--pd-gain", type=float, default=12.0)
    parser.add_argument("--pd-deadband", type=float, default=12.0)
    # Timing of the simulated control loop (defaults = BrowserStorkEnv defaults).
    parser.add_argument("--decision-frames-min", type=int, default=4)
    parser.add_argument("--decision-frames-max", type=int, default=8)
    parser.add_argument("--pulse-frames-min", type=int, default=1)
    parser.add_argument("--pulse-frames-max", type=int, default=2)
    parser.add_argument("--action-delay-max", type=int, default=1)
    parser.add_argument("--observation-delay-max", type=int, default=1)
    parser.add_argument("--missed-pulse-probability", type=float, default=0.03)
    parser.add_argument(
        "--model", type=Path, default=Path("models_browser/stork_ppo.zip")
    )
    parser.add_argument(
        "--stats", type=Path, default=Path("models_browser/vec_normalize.pkl")
    )
    parser.add_argument("--max-frames", type=int, default=18_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--projection-frames-min", type=int, default=0)
    parser.add_argument("--projection-frames-max", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = {
        "render_mode": None,
        "max_steps": args.max_frames,
        "observation_projection_frames_range": (
            args.projection_frames_min,
            args.projection_frames_max,
        ),
        "decision_frames_range": (
            args.decision_frames_min,
            args.decision_frames_max,
        ),
        "pulse_frames_range": (args.pulse_frames_min, args.pulse_frames_max),
        "action_delay_frames_range": (0, args.action_delay_max),
        "observation_delay_frames_range": (0, args.observation_delay_max),
        "missed_pulse_probability": args.missed_pulse_probability,
    }
    env = BrowserStorkEnv(**env_kwargs)
    model = normalizer = None

    if args.agent == "model":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        wrapper = DummyVecEnv(
            [lambda: BrowserStorkEnv(**env_kwargs)]
        )
        normalizer = VecNormalize.load(args.stats, wrapper)
        normalizer.training = False
        normalizer.norm_reward = False
        model = PPO.load(args.model)

    metres: list[int] = []
    frames: list[int] = []
    truncations = 0
    decisions = 0
    active_actions = 0
    same_direction_repeats = 0
    immediate_reversals = 0
    none_after_active = 0
    after_active_opportunities = 0
    absolute_angles: list[float] = []
    high_velocity_actions = 0
    successful_velocity_corrections = 0
    large_angle_decisions = 0
    successful_angle_corrections = 0
    velocity_over_limit_frames = 0
    velocity_over_one_frames = 0
    evaluated_flash_frames = 0
    velocity_limit_crossings = 0
    fast_center_crossings = 0
    missed_braking_decisions = 0
    high_velocity_excursions = 0
    high_velocity_recoveries = 0
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            previous_action = 0
            while True:
                if args.agent == "lookahead":
                    action = lookahead_action(
                        observation,
                        horizon=args.lookahead_horizon,
                        velocity_weight=args.lookahead_weight,
                    )
                elif args.agent == "pd":
                    action = pd_action(
                        observation,
                        velocity_gain=args.pd_gain,
                        deadband=args.pd_deadband,
                    )
                elif model is None:
                    action = heuristic_action(observation)
                else:
                    normalized = normalizer.normalize_obs(observation[None, :])
                    predicted, _ = model.predict(normalized, deterministic=True)
                    action = int(predicted[0])
                observation, _, terminated, truncated, info = env.step(action)
                decisions += 1
                active_actions += int(action != 0)
                same_direction_repeats += int(
                    action != 0 and action == previous_action
                )
                immediate_reversals += int(
                    action in (1, 2)
                    and previous_action in (1, 2)
                    and action != previous_action
                )
                none_after_active += int(
                    action == 0 and previous_action in (1, 2)
                )
                after_active_opportunities += int(previous_action in (1, 2))
                absolute_angles.append(abs(float(info["body_rotation"])))
                start_abs_velocity = abs(float(info["balance_velocity"])) + float(
                    info["velocity_abs_reduction"]
                )
                if action != 0 and start_abs_velocity >= 2.0:
                    high_velocity_actions += 1
                    successful_velocity_corrections += int(
                        float(info["velocity_abs_reduction"]) > 0.0
                    )
                start_abs_angle = abs(float(info["body_rotation"])) + float(
                    info["angle_abs_reduction"]
                )
                if start_abs_angle >= 30.0:
                    large_angle_decisions += 1
                    successful_angle_corrections += int(
                        float(info["angle_abs_reduction"]) > 0.0
                    )
                velocity_over_limit_frames += int(
                    info["velocity_over_limit_frames"]
                )
                velocity_over_one_frames += int(info["velocity_over_one_frames"])
                evaluated_flash_frames += int(info["decision_frames"])
                velocity_limit_crossings += int(
                    bool(info["velocity_limit_crossed"])
                )
                fast_center_crossings += int(
                    bool(info.get("fast_center_crossing", False))
                )
                missed_braking_decisions += int(
                    bool(info.get("missed_braking", False))
                )
                high_velocity_excursions += int(
                    float(info["high_velocity_entry_penalty"]) < 0.0
                )
                high_velocity_recoveries += int(
                    float(info["high_velocity_recovery_bonus"]) > 0.0
                )
                previous_action = action
                if terminated or truncated:
                    metres.append(int(info["metres"]))
                    frames.append(int(info["steps"]))
                    truncations += int(truncated)
                    break
    finally:
        env.close()
        if normalizer is not None:
            normalizer.close()

    sorted_metres = sorted(metres)
    p10 = sorted_metres[max(0, int(len(sorted_metres) * 0.10) - 1)]
    p90 = sorted_metres[min(len(sorted_metres) - 1, int(len(sorted_metres) * 0.90))]
    print(f"agent={args.agent} episodes={len(metres)}")
    print(
        f"metres mean={mean(metres):.1f} median={median(metres):.1f} "
        f"p10={p10} p90={p90} best={max(metres)}"
    )
    print(
        f"survival mean={mean(frames) / 30.0:.2f}s "
        f"median={median(frames) / 30.0:.2f}s "
        f"reached_limit={truncations}/{len(metres)}"
    )
    print(
        f"control active_action_rate={active_actions / decisions:.3f} "
        f"same_direction_repeat_rate={same_direction_repeats / decisions:.3f} "
        f"immediate_reversal_after_active_rate="
        f"{immediate_reversals / max(1, after_active_opportunities):.3f} "
        f"none_after_active_rate="
        f"{none_after_active / max(1, after_active_opportunities):.3f} "
        f"mean_abs_angle={mean(absolute_angles):.2f}deg"
    )
    print(
        f"correction high_velocity_success_rate="
        f"{successful_velocity_corrections / max(1, high_velocity_actions):.3f} "
        f"large_angle_toward_zero_rate="
        f"{successful_angle_corrections / max(1, large_angle_decisions):.3f}"
    )
    print(
        f"velocity_safety over_one_frame_rate="
        f"{velocity_over_one_frames / max(1, evaluated_flash_frames):.3f} "
        f"over_margin_frame_rate="
        f"{velocity_over_limit_frames / max(1, evaluated_flash_frames):.3f} "
        f"crossings_per_100_decisions="
        f"{100.0 * velocity_limit_crossings / decisions:.2f}"
    )
    print(
        f"fast_center_crossings_per_100_decisions="
        f"{100.0 * fast_center_crossings / decisions:.2f}"
    )
    print(
        f"missed_braking_per_100_decisions="
        f"{100.0 * missed_braking_decisions / decisions:.2f}"
    )
    print(
        f"excursion recovery_rate="
        f"{high_velocity_recoveries / max(1, high_velocity_excursions):.3f} "
        f"entries_per_100_decisions="
        f"{100.0 * high_velocity_excursions / decisions:.2f} "
        f"recovered={high_velocity_recoveries}/{high_velocity_excursions}"
    )


if __name__ == "__main__":
    main()
