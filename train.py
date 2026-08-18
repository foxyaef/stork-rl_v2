from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from stork_rl import BrowserStorkEnv, StorkEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Walk the Stork.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=18_000)
    parser.add_argument("--profile", choices=("browser", "exact"), default="browser")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decision-frames-min", type=int, default=4)
    parser.add_argument("--decision-frames-max", type=int, default=8)
    parser.add_argument("--action-delay-max", type=int, default=1)
    parser.add_argument("--pulse-frames-min", type=int, default=1)
    parser.add_argument("--pulse-frames-max", type=int, default=2)
    parser.add_argument("--observation-delay-max", type=int, default=1)
    parser.add_argument("--projection-frames-min", type=int, default=0)
    parser.add_argument("--projection-frames-max", type=int, default=0)
    parser.add_argument("--missed-pulse-probability", type=float, default=0.03)
    parser.add_argument("--center-reward-per-frame", type=float, default=0.015)
    parser.add_argument("--action-frame-penalty", type=float, default=0.002)
    parser.add_argument("--same-direction-penalty", type=float, default=0.06)
    parser.add_argument("--coast-after-input-reward", type=float, default=0.04)
    parser.add_argument("--immediate-reversal-penalty", type=float, default=0.04)
    parser.add_argument("--angle-progress-reward-scale", type=float, default=0.06)
    parser.add_argument("--velocity-progress-reward-scale", type=float, default=0.10)
    parser.add_argument("--safe-velocity-limit", type=float, default=1.0)
    parser.add_argument("--low-velocity-reward-per-frame", type=float, default=0.0)
    parser.add_argument(
        "--velocity-excess-penalty-per-frame", type=float, default=0.0
    )
    parser.add_argument("--velocity-crossing-penalty", type=float, default=0.0)
    parser.add_argument("--velocity-recovery-reward-scale", type=float, default=0.0)
    parser.add_argument("--high-velocity-threshold", type=float, default=1.5)
    parser.add_argument("--recovered-velocity-threshold", type=float, default=1.0)
    parser.add_argument("--high-velocity-entry-penalty", type=float, default=0.05)
    parser.add_argument("--high-velocity-recovery-bonus", type=float, default=0.04)
    parser.add_argument("--approaching-center-probability", type=float, default=0.0)
    parser.add_argument("--recovery-start-probability", type=float, default=0.0)
    parser.add_argument("--approaching-velocity-reward-scale", type=float, default=0.0)
    parser.add_argument("--fast-center-crossing-penalty", type=float, default=0.0)
    parser.add_argument("--fast-center-crossing-threshold", type=float, default=1.2)
    parser.add_argument("--missed-braking-penalty", type=float, default=0.0)
    parser.add_argument("--braking-horizon-frames", type=float, default=8.0)
    parser.add_argument("--fall-penalty", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-stats", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.resume_model is None) != (args.resume_stats is None):
        raise ValueError("Pass --resume-model and --resume-stats together")
    if args.output is None:
        args.output = Path("models_browser" if args.profile == "browser" else "models_exact")
    args.output.mkdir(parents=True, exist_ok=True)

    env_class = BrowserStorkEnv if args.profile == "browser" else StorkEnv
    env_kwargs = {
        "render_mode": None,
        "max_steps": args.max_episode_steps,
    }
    if args.profile == "browser":
        env_kwargs.update(
            {
                "decision_frames_range": (
                    args.decision_frames_min,
                    args.decision_frames_max,
                ),
                "action_delay_frames_range": (0, args.action_delay_max),
                "pulse_frames_range": (
                    args.pulse_frames_min,
                    args.pulse_frames_max,
                ),
                "observation_delay_frames_range": (
                    0,
                    args.observation_delay_max,
                ),
                "observation_projection_frames_range": (
                    args.projection_frames_min,
                    args.projection_frames_max,
                ),
                "missed_pulse_probability": args.missed_pulse_probability,
                "center_reward_per_frame": args.center_reward_per_frame,
                "action_frame_penalty": args.action_frame_penalty,
                "same_direction_penalty": args.same_direction_penalty,
                "coast_after_input_reward": args.coast_after_input_reward,
                "immediate_reversal_penalty": args.immediate_reversal_penalty,
                "angle_progress_reward_scale": args.angle_progress_reward_scale,
                "velocity_progress_reward_scale": (
                    args.velocity_progress_reward_scale
                ),
                "safe_velocity_limit": args.safe_velocity_limit,
                "low_velocity_reward_per_frame": (
                    args.low_velocity_reward_per_frame
                ),
                "velocity_excess_penalty_per_frame": (
                    args.velocity_excess_penalty_per_frame
                ),
                "velocity_crossing_penalty": args.velocity_crossing_penalty,
                "velocity_recovery_reward_scale": (
                    args.velocity_recovery_reward_scale
                ),
                "high_velocity_threshold": args.high_velocity_threshold,
                "recovered_velocity_threshold": (
                    args.recovered_velocity_threshold
                ),
                "high_velocity_entry_penalty": (
                    args.high_velocity_entry_penalty
                ),
                "high_velocity_recovery_bonus": (
                    args.high_velocity_recovery_bonus
                ),
                "approaching_center_probability": (
                    args.approaching_center_probability
                ),
                "recovery_start_probability": args.recovery_start_probability,
                "approaching_velocity_reward_scale": (
                    args.approaching_velocity_reward_scale
                ),
                "fast_center_crossing_penalty": (
                    args.fast_center_crossing_penalty
                ),
                "fast_center_crossing_threshold": (
                    args.fast_center_crossing_threshold
                ),
                "missed_braking_penalty": args.missed_braking_penalty,
                "braking_horizon_frames": args.braking_horizon_frames,
                "fall_penalty": args.fall_penalty,
            }
        )

    vec_env = make_vec_env(
        env_class,
        n_envs=args.envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
    )
    if args.resume_stats is not None:
        vec_env = VecNormalize.load(args.resume_stats, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1),
        save_path=str(args.output / "checkpoints"),
        name_prefix="stork_ppo",
        save_vecnormalize=True,
    )
    if args.resume_model is not None:
        model = PPO.load(
            args.resume_model,
            env=vec_env,
            tensorboard_log=str(args.output / "tensorboard"),
            learning_rate=args.learning_rate,
        )
        reset_num_timesteps = False
        print(f"Resuming from {args.resume_model}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(args.output / "tensorboard"),
            n_steps=2048,
            batch_size=256,
            learning_rate=args.learning_rate,
            # A browser decision spans roughly 4-8 Flash frames. 0.997 per macro
            # decision is close to the former 0.999 per-frame horizon.
            gamma=0.997 if args.profile == "browser" else 0.999,
        )
        reset_num_timesteps = True
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(args.output / "stork_ppo")
    vec_env.save(args.output / "vec_normalize.pkl")
    vec_env.close()

    print(f"Profile: {args.profile}")
    print(f"Saved model to {args.output / 'stork_ppo.zip'}")
    print(f"Saved normalization stats to {args.output / 'vec_normalize.pkl'}")


if __name__ == "__main__":
    main()
