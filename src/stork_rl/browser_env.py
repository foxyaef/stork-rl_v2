from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from stork_rl.env import StorkEnv


class BrowserStorkEnv(gym.Env[np.ndarray, int]):
    """Macro-action environment matching the measured browser control loop.

    One agent step represents several 30 FPS Flash frames. The selected arrow
    key is applied only as a short pulse, with small randomized delay, while the
    remaining frames receive no input. Random timing and observation error make
    the learned policy robust to screenshot and keyboard jitter.
    """

    metadata = StorkEnv.metadata

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int | None = 18_000,
        realtime: bool = True,
        decision_frames_range: tuple[int, int] = (4, 8),
        action_delay_frames_range: tuple[int, int] = (0, 1),
        pulse_frames_range: tuple[int, int] = (1, 2),
        observation_delay_frames_range: tuple[int, int] = (0, 1),
        observation_projection_frames_range: tuple[int, int] = (0, 0),
        missed_pulse_probability: float = 0.03,
        angle_noise_std: float = 1.0,
        velocity_noise_std: float = 0.12,
        difficulty_noise_std: float = 0.003,
        center_reward_per_frame: float = 0.015,
        action_frame_penalty: float = 0.002,
        same_direction_penalty: float = 0.06,
        coast_after_input_reward: float = 0.04,
        immediate_reversal_penalty: float = 0.04,
        angle_progress_reward_scale: float = 0.06,
        velocity_progress_reward_scale: float = 0.10,
        angle_progress_normalizer: float = 10.0,
        velocity_progress_normalizer: float = 1.0,
        safe_velocity_limit: float = 1.0,
        low_velocity_reward_per_frame: float = 0.0,
        velocity_excess_penalty_per_frame: float = 0.0,
        velocity_crossing_penalty: float = 0.0,
        velocity_recovery_reward_scale: float = 0.0,
        high_velocity_threshold: float = 1.5,
        recovered_velocity_threshold: float = 1.0,
        high_velocity_entry_penalty: float = 0.05,
        high_velocity_recovery_bonus: float = 0.04,
        approaching_center_probability: float = 0.0,
        recovery_start_probability: float = 0.0,
        approaching_velocity_reward_scale: float = 0.0,
        fast_center_crossing_penalty: float = 0.0,
        fast_center_crossing_threshold: float = 1.2,
        missed_braking_penalty: float = 0.0,
        braking_horizon_frames: float = 8.0,
        fall_penalty: float = 1.0,
        center_angle_scale: float = 30.0,
        center_velocity_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.base_env = StorkEnv(
            render_mode=render_mode,
            max_steps=max_steps,
            realtime=realtime,
        )
        self.render_mode = render_mode
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

        self.decision_frames_range = self._validate_range(
            "decision_frames_range", decision_frames_range, minimum=1
        )
        self.action_delay_frames_range = self._validate_range(
            "action_delay_frames_range", action_delay_frames_range, minimum=0
        )
        self.pulse_frames_range = self._validate_range(
            "pulse_frames_range", pulse_frames_range, minimum=1
        )
        self.observation_delay_frames_range = self._validate_range(
            "observation_delay_frames_range",
            observation_delay_frames_range,
            minimum=0,
        )
        self.observation_projection_frames_range = self._validate_range(
            "observation_projection_frames_range",
            observation_projection_frames_range,
            minimum=0,
        )
        if not 0.0 <= missed_pulse_probability <= 1.0:
            raise ValueError("missed_pulse_probability must be in [0, 1]")
        self.missed_pulse_probability = missed_pulse_probability
        self.angle_noise_std = max(0.0, angle_noise_std)
        self.velocity_noise_std = max(0.0, velocity_noise_std)
        self.difficulty_noise_std = max(0.0, difficulty_noise_std)
        self.center_reward_per_frame = max(0.0, center_reward_per_frame)
        self.action_frame_penalty = max(0.0, action_frame_penalty)
        self.same_direction_penalty = max(0.0, same_direction_penalty)
        self.coast_after_input_reward = max(0.0, coast_after_input_reward)
        self.immediate_reversal_penalty = max(
            0.0, immediate_reversal_penalty
        )
        self.angle_progress_reward_scale = max(
            0.0, angle_progress_reward_scale
        )
        self.velocity_progress_reward_scale = max(
            0.0, velocity_progress_reward_scale
        )
        self.angle_progress_normalizer = max(
            1e-6, angle_progress_normalizer
        )
        self.velocity_progress_normalizer = max(
            1e-6, velocity_progress_normalizer
        )
        self.safe_velocity_limit = max(1e-6, safe_velocity_limit)
        self.low_velocity_reward_per_frame = max(
            0.0, low_velocity_reward_per_frame
        )
        self.velocity_excess_penalty_per_frame = max(
            0.0, velocity_excess_penalty_per_frame
        )
        self.velocity_crossing_penalty = max(
            0.0, velocity_crossing_penalty
        )
        self.velocity_recovery_reward_scale = max(
            0.0, velocity_recovery_reward_scale
        )
        self.high_velocity_threshold = max(1e-6, high_velocity_threshold)
        self.recovered_velocity_threshold = max(
            0.0,
            min(recovered_velocity_threshold, self.high_velocity_threshold),
        )
        self.high_velocity_entry_penalty = max(
            0.0, high_velocity_entry_penalty
        )
        self.high_velocity_recovery_bonus = max(
            0.0,
            min(high_velocity_recovery_bonus, self.high_velocity_entry_penalty),
        )
        if not 0.0 <= approaching_center_probability <= 1.0:
            raise ValueError("approaching_center_probability must be in [0, 1]")
        if not 0.0 <= recovery_start_probability <= 1.0:
            raise ValueError("recovery_start_probability must be in [0, 1]")
        if approaching_center_probability + recovery_start_probability > 1.0:
            raise ValueError("curriculum probabilities must sum to at most 1")
        self.approaching_center_probability = approaching_center_probability
        self.recovery_start_probability = recovery_start_probability
        self.approaching_velocity_reward_scale = max(
            0.0, approaching_velocity_reward_scale
        )
        self.fast_center_crossing_penalty = max(
            0.0, fast_center_crossing_penalty
        )
        self.fast_center_crossing_threshold = max(
            1e-6, fast_center_crossing_threshold
        )
        self.missed_braking_penalty = max(0.0, missed_braking_penalty)
        self.braking_horizon_frames = max(0.0, braking_horizon_frames)
        self.fall_penalty = max(0.0, fall_penalty)
        self.center_angle_scale = max(1e-6, center_angle_scale)
        self.center_velocity_scale = max(1e-6, center_velocity_scale)

        self.decision_count = 0
        self.last_nonzero_action: int | None = None
        self.last_decision_action = 0
        self.velocity_excursion_active = False
        self.same_direction_streak = 0.0
        self._last_true_observation = np.zeros(4, dtype=np.float32)

    @staticmethod
    def _validate_range(
        name: str, value: tuple[int, int], minimum: int
    ) -> tuple[int, int]:
        low, high = map(int, value)
        if low < minimum or high < low:
            raise ValueError(f"Invalid {name}: {value}")
        return low, high

    def _sample_int(self, bounds: tuple[int, int]) -> int:
        low, high = bounds
        return int(self.np_random.integers(low, high + 1))

    def _noisy_observation(self, observation: np.ndarray) -> np.ndarray:
        result = observation.astype(np.float32, copy=True)
        result[0] += self.np_random.normal(0.0, self.angle_noise_std)
        result[1] += self.np_random.normal(0.0, self.velocity_noise_std)
        result[2] = max(
            1.0,
            result[2] + self.np_random.normal(0.0, self.difficulty_noise_std),
        )
        # Speed is deterministic from difficulty in the SWF. Rebuild it from
        # the noisy difficulty so the observation remains internally coherent.
        result[3] = max(0.0, (result[2] - 0.002) * 3.0)
        return result

    @staticmethod
    def _project_observation(
        observation: np.ndarray, frames: int
    ) -> np.ndarray:
        """Project an observed state forward with the live no-input model."""
        result = observation.astype(np.float32, copy=True)
        angle = float(result[0])
        velocity = float(result[1])
        difficulty = float(result[2])
        for _ in range(frames):
            difficulty += 0.002
            velocity += (angle / 3.0) * difficulty / 100.0
            angle += velocity
        result[0] = angle
        result[1] = velocity
        result[2] = difficulty
        result[3] = max(0.0, (difficulty - 0.002) * 3.0)
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.base_env.reset(seed=seed, options=options)
        curriculum_draw = float(self.np_random.random())
        start_type = "normal"
        if curriculum_draw < self.approaching_center_probability:
            start_type = "approaching_center"
            angle_sign = -1.0 if self.np_random.random() < 0.5 else 1.0
            self.base_env.body_rotation = angle_sign * float(
                self.np_random.uniform(5.0, 30.0)
            )
            self.base_env.balance_velocity = -angle_sign * float(
                self.np_random.uniform(1.2, 3.0)
            )
            self.base_env.difficulty = float(self.np_random.uniform(1.5, 5.5))
        elif curriculum_draw < (
            self.approaching_center_probability + self.recovery_start_probability
        ):
            start_type = "high_velocity_recovery"
            angle_sign = -1.0 if self.np_random.random() < 0.5 else 1.0
            velocity_sign = -1.0 if self.np_random.random() < 0.5 else 1.0
            self.base_env.body_rotation = angle_sign * float(
                self.np_random.uniform(30.0, 65.0)
            )
            self.base_env.balance_velocity = velocity_sign * float(
                self.np_random.uniform(1.5, 3.5)
            )
            self.base_env.difficulty = float(self.np_random.uniform(1.5, 5.5))
        if start_type != "normal":
            self.base_env.forward_speed = (
                self.base_env.difficulty - self.base_env.DIFFICULTY_INCREMENT
            ) * self.base_env.SPEED_MULTIPLIER
            observation = self.base_env._observation()
            info = self.base_env._info()
        self.decision_count = 0
        self.last_nonzero_action = None
        self.last_decision_action = 0
        self.velocity_excursion_active = False
        self.same_direction_streak = 0.0
        self._last_true_observation = observation.copy()
        projection_frames = self._sample_int(
            self.observation_projection_frames_range
        )
        info = dict(info)
        info.update(
            {
                "decision_count": 0,
                "decision_frames": 0,
                "pulse_frames": 0,
                "action_delay_frames": 0,
                "observation_delay_frames": 0,
                "observation_projection_frames": projection_frames,
                "pulse_missed": False,
                "distance_reward": 0.0,
                "center_reward": 0.0,
                "action_penalty": 0.0,
                "same_direction_penalty": 0.0,
                "coast_reward": 0.0,
                "immediate_reversal_penalty": 0.0,
                "angle_progress_reward": 0.0,
                "velocity_progress_reward": 0.0,
                "approaching_velocity_reward": 0.0,
                "fast_center_crossing_penalty": 0.0,
                "fast_center_crossing": False,
                "missed_braking_penalty": 0.0,
                "missed_braking": False,
                "curriculum_start_type": start_type,
                "angle_abs_reduction": 0.0,
                "velocity_abs_reduction": 0.0,
                "repeat_recovery_discount": 0.0,
                "low_velocity_reward": 0.0,
                "velocity_excess_penalty": 0.0,
                "velocity_crossing_penalty": 0.0,
                "velocity_limit_crossed": False,
                "velocity_limit_recovery_reward": 0.0,
                "velocity_over_limit_frames": 0,
                "velocity_over_one_frames": 0,
                "max_abs_velocity": 0.0,
                "high_velocity_entry_penalty": 0.0,
                "high_velocity_recovery_bonus": 0.0,
                "velocity_excursion_active": False,
                "fall_penalty": 0.0,
                "same_direction_streak": 0.0,
            }
        )
        projected = self._project_observation(observation, projection_frames)
        return self._noisy_observation(projected), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be 0, 1, or 2; got {action!r}")

        start_angle = float(self.base_env.body_rotation)
        start_velocity = float(self.base_env.balance_velocity)

        decision_frames = self._sample_int(self.decision_frames_range)
        action_delay = self._sample_int(self.action_delay_frames_range)
        pulse_frames = self._sample_int(self.pulse_frames_range)
        pulse_missed = bool(
            action != 0 and self.np_random.random() < self.missed_pulse_probability
        )
        if pulse_missed:
            pulse_frames = 0

        total_reward = 0.0
        distance_reward = 0.0
        center_reward = 0.0
        low_velocity_reward = 0.0
        velocity_excess_penalty = 0.0
        velocity_over_limit_frames = 0
        velocity_over_one_frames = 0
        max_abs_velocity = abs(start_velocity)
        previous_angle = start_angle
        fast_center_crossing = False
        terminated = truncated = False
        info: dict[str, Any] = {}
        observations: list[np.ndarray] = []
        applied_frames = 0

        for frame_index in range(decision_frames):
            pulse_active = (
                action != 0
                and action_delay <= frame_index < action_delay + pulse_frames
            )
            frame_action = action if pulse_active else 0
            if pulse_active:
                applied_frames += 1
            observation, reward, terminated, truncated, info = self.base_env.step(
                frame_action
            )
            observations.append(observation.copy())
            current_angle = float(observation[0])
            current_velocity = abs(float(observation[1]))
            crossed_center = (
                previous_angle != 0.0
                and current_angle != 0.0
                and np.sign(previous_angle) != np.sign(current_angle)
            )
            if crossed_center and current_velocity > self.fast_center_crossing_threshold:
                fast_center_crossing = True
            previous_angle = current_angle
            distance_reward += reward
            angle_score = math.exp(
                -0.5 * (float(observation[0]) / self.center_angle_scale) ** 2
            )
            velocity_score = math.exp(
                -0.5
                * (float(observation[1]) / self.center_velocity_scale) ** 2
            )
            center_reward += (
                self.center_reward_per_frame * angle_score * velocity_score
            )
            absolute_velocity = abs(float(observation[1]))
            max_abs_velocity = max(max_abs_velocity, absolute_velocity)
            velocity_over_one_frames += int(absolute_velocity > 1.0)
            velocity_excess = max(
                0.0, absolute_velocity - self.safe_velocity_limit
            )
            if velocity_excess > 0.0:
                velocity_over_limit_frames += 1
                velocity_excess_penalty += (
                    self.velocity_excess_penalty_per_frame
                    * min(velocity_excess, 3.0) ** 2
                )
            else:
                low_velocity_reward += (
                    self.low_velocity_reward_per_frame
                    * (1.0 - absolute_velocity / self.safe_velocity_limit)
                )
            if terminated or truncated:
                break

        self.decision_count += 1
        if action != 0 and applied_frames > 0:
            if action == self.last_nonzero_action:
                self.same_direction_streak += 1.0
            else:
                self.last_nonzero_action = int(action)
                self.same_direction_streak = 1.0
        else:
            # Two no-input decisions erase one level of directional momentum
            # memory. This permits a necessary later correction without making
            # alternating same-direction pulses a free loophole.
            self.same_direction_streak = max(
                0.0, self.same_direction_streak - 0.5
            )

        action_penalty = self.action_frame_penalty * applied_frames
        repeat_excess = max(0.0, self.same_direction_streak - 1.0)
        final_observation = observations[-1]
        angle_abs_reduction = abs(start_angle) - abs(float(final_observation[0]))
        velocity_abs_reduction = abs(start_velocity) - abs(
            float(final_observation[1])
        )
        angle_urgency = float(np.clip(abs(start_angle) / 30.0, 0.0, 1.0))
        velocity_urgency = float(
            np.clip(abs(start_velocity) / 2.0, 0.0, 1.0)
        )
        angle_progress = float(
            np.clip(
                angle_abs_reduction / self.angle_progress_normalizer,
                -1.0,
                1.0,
            )
        )
        velocity_progress = float(
            np.clip(
                velocity_abs_reduction / self.velocity_progress_normalizer,
                -1.0,
                1.0,
            )
        )
        # Reward demonstrated recovery without punishing unavoidable inertial
        # drift. Velocity recovery only counts when the policy actively pulses
        # a key; naturally slowing down while choosing NONE is not credited.
        angle_progress_reward = (
            self.angle_progress_reward_scale
            * angle_urgency
            * max(0.0, angle_progress)
        )
        velocity_progress_reward = 0.0
        if action != 0:
            velocity_progress_reward = (
                self.velocity_progress_reward_scale
                * velocity_urgency
                * max(0.0, velocity_progress)
            )
        approaching_center = (
            start_angle * start_velocity < 0.0
            and abs(start_velocity) >= self.fast_center_crossing_threshold
        )
        approaching_velocity_reward = 0.0
        if action != 0 and approaching_center:
            approaching_velocity_reward = (
                self.approaching_velocity_reward_scale
                * max(0.0, velocity_progress)
            )
        time_to_center_frames = (
            abs(start_angle / start_velocity)
            if approaching_center and start_velocity != 0.0
            else math.inf
        )
        missed_braking = (
            action == 0
            and approaching_center
            and time_to_center_frames <= self.braking_horizon_frames
        )
        missed_braking_cost = (
            self.missed_braking_penalty if missed_braking else 0.0
        )
        if terminated:
            # The SWF zeros velocity on a fall; that must not look like a
            # successful angular-velocity correction.
            angle_progress_reward = 0.0
            velocity_progress_reward = 0.0
            approaching_velocity_reward = 0.0
        recovery_discount = 0.6 * max(0.0, velocity_progress)
        repeated_penalty = (
            self.same_direction_penalty
            * repeat_excess
            * applied_frames
            * (1.0 - recovery_discount)
        )
        coast_score = math.exp(
            -0.5 * (float(final_observation[0]) / 55.0) ** 2
        ) * math.exp(-0.5 * (float(final_observation[1]) / 3.0) ** 2)
        coast_reward = 0.0
        if (
            action == 0
            and self.last_decision_action in (1, 2)
            and not missed_braking
        ):
            # Reward pulse-and-coast control, but only while the resulting
            # state is reasonably calm. Waiting near a fall earns almost none.
            coast_reward = self.coast_after_input_reward * coast_score
        immediate_reversal = (
            action in (1, 2)
            and self.last_decision_action in (1, 2)
            and action != self.last_decision_action
        )
        reversal_penalty = (
            self.immediate_reversal_penalty if immediate_reversal else 0.0
        )
        start_limit_excess = max(
            0.0, abs(start_velocity) - self.safe_velocity_limit
        )
        end_limit_excess = max(
            0.0,
            abs(float(final_observation[1])) - self.safe_velocity_limit,
        )
        velocity_limit_recovery_reward = (
            self.velocity_recovery_reward_scale
            * min(1.0, max(0.0, start_limit_excess - end_limit_excess))
        )
        crossed_velocity_limit = (
            abs(start_velocity) <= self.safe_velocity_limit
            and max_abs_velocity > self.safe_velocity_limit
        )
        crossing_penalty = (
            self.velocity_crossing_penalty if crossed_velocity_limit else 0.0
        )
        fast_crossing_penalty = (
            self.fast_center_crossing_penalty if fast_center_crossing else 0.0
        )
        excursion_entry_penalty = 0.0
        if (
            not self.velocity_excursion_active
            and max_abs_velocity > self.high_velocity_threshold
        ):
            self.velocity_excursion_active = True
            excursion_entry_penalty = self.high_velocity_entry_penalty

        excursion_recovery_bonus = 0.0
        if (
            self.velocity_excursion_active
            and abs(float(final_observation[1]))
            < self.recovered_velocity_threshold
            and not terminated
        ):
            excursion_recovery_bonus = self.high_velocity_recovery_bonus
            self.velocity_excursion_active = False
        terminal_penalty = self.fall_penalty if terminated else 0.0
        if terminated:
            velocity_limit_recovery_reward = 0.0
        total_reward = (
            distance_reward
            + center_reward
            + coast_reward
            + angle_progress_reward
            + velocity_progress_reward
            + approaching_velocity_reward
            + low_velocity_reward
            + velocity_limit_recovery_reward
            + excursion_recovery_bonus
            - action_penalty
            - repeated_penalty
            - reversal_penalty
            - velocity_excess_penalty
            - crossing_penalty
            - fast_crossing_penalty
            - missed_braking_cost
            - excursion_entry_penalty
            - terminal_penalty
        )
        self.last_decision_action = int(action)
        self._last_true_observation = observations[-1]
        maximum_delay = min(
            self.observation_delay_frames_range[1], len(observations) - 1
        )
        minimum_delay = min(
            self.observation_delay_frames_range[0], maximum_delay
        )
        observation_delay = int(
            self.np_random.integers(minimum_delay, maximum_delay + 1)
        )
        selected_observation = observations[-1 - observation_delay]
        projection_frames = self._sample_int(
            self.observation_projection_frames_range
        )
        projected_observation = self._project_observation(
            selected_observation, projection_frames
        )

        info = dict(info)
        info.update(
            {
                "decision_count": self.decision_count,
                "decision_frames": len(observations),
                "requested_decision_frames": decision_frames,
                "pulse_frames": applied_frames,
                "requested_pulse_frames": pulse_frames,
                "action_delay_frames": action_delay,
                "observation_delay_frames": observation_delay,
                "observation_projection_frames": projection_frames,
                "pulse_missed": pulse_missed,
                "selected_action": int(action),
                "distance_reward": float(distance_reward),
                "center_reward": float(center_reward),
                "action_penalty": float(-action_penalty),
                "same_direction_penalty": float(-repeated_penalty),
                "coast_reward": float(coast_reward),
                "immediate_reversal_penalty": float(-reversal_penalty),
                "angle_progress_reward": float(angle_progress_reward),
                "velocity_progress_reward": float(velocity_progress_reward),
                "approaching_velocity_reward": float(
                    approaching_velocity_reward
                ),
                "fast_center_crossing_penalty": float(-fast_crossing_penalty),
                "fast_center_crossing": bool(fast_center_crossing),
                "missed_braking_penalty": float(-missed_braking_cost),
                "missed_braking": bool(missed_braking),
                "curriculum_start_type": "continuing",
                "angle_abs_reduction": float(angle_abs_reduction),
                "velocity_abs_reduction": float(velocity_abs_reduction),
                "repeat_recovery_discount": float(recovery_discount),
                "low_velocity_reward": float(low_velocity_reward),
                "velocity_excess_penalty": float(-velocity_excess_penalty),
                "velocity_crossing_penalty": float(-crossing_penalty),
                "velocity_limit_crossed": bool(crossed_velocity_limit),
                "velocity_limit_recovery_reward": float(
                    velocity_limit_recovery_reward
                ),
                "velocity_over_limit_frames": int(velocity_over_limit_frames),
                "velocity_over_one_frames": int(velocity_over_one_frames),
                "max_abs_velocity": float(max_abs_velocity),
                "high_velocity_entry_penalty": float(
                    -excursion_entry_penalty
                ),
                "high_velocity_recovery_bonus": float(
                    excursion_recovery_bonus
                ),
                "velocity_excursion_active": bool(
                    self.velocity_excursion_active
                ),
                "fall_penalty": float(-terminal_penalty),
                "same_direction_streak": float(self.same_direction_streak),
            }
        )
        return (
            self._noisy_observation(projected_observation),
            float(total_reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray | None:
        return self.base_env.render()

    def poll_human_action(self) -> int:
        return self.base_env.poll_human_action()

    @property
    def user_closed(self) -> bool:
        return self.base_env.user_closed

    def close(self) -> None:
        self.base_env.close()
