import numpy as np

from stork_rl import BrowserStorkEnv, StorkEnv


def make_fixed_browser_env(delay: int, pulse: int, decision: int) -> BrowserStorkEnv:
    return BrowserStorkEnv(
        max_steps=None,
        decision_frames_range=(decision, decision),
        action_delay_frames_range=(delay, delay),
        pulse_frames_range=(pulse, pulse),
        observation_delay_frames_range=(0, 0),
        missed_pulse_probability=0.0,
        angle_noise_std=0.0,
        velocity_noise_std=0.0,
        difficulty_noise_std=0.0,
        center_reward_per_frame=0.0,
        action_frame_penalty=0.0,
        same_direction_penalty=0.0,
        coast_after_input_reward=0.0,
        immediate_reversal_penalty=0.0,
        angle_progress_reward_scale=0.0,
        velocity_progress_reward_scale=0.0,
        low_velocity_reward_per_frame=0.0,
        velocity_excess_penalty_per_frame=0.0,
        velocity_crossing_penalty=0.0,
        velocity_recovery_reward_scale=0.0,
        fall_penalty=0.0,
    )


def test_macro_step_matches_explicit_flash_frame_actions() -> None:
    browser_env = make_fixed_browser_env(delay=2, pulse=2, decision=5)
    exact_env = StorkEnv(max_steps=None)
    browser_env.reset(seed=3)
    exact_env.reset(seed=3)

    observation, reward, terminated, truncated, info = browser_env.step(1)
    exact_reward = 0.0
    for frame_action in (0, 0, 1, 1, 0):
        exact_observation, frame_reward, exact_terminated, exact_truncated, _ = (
            exact_env.step(frame_action)
        )
        exact_reward += frame_reward

    np.testing.assert_allclose(observation, exact_observation)
    np.testing.assert_allclose(reward, exact_reward)
    assert terminated == exact_terminated
    assert truncated == exact_truncated
    assert info["decision_frames"] == 5
    assert info["pulse_frames"] == 2
    assert info["action_delay_frames"] == 2


def test_zero_action_never_creates_a_pulse() -> None:
    env = make_fixed_browser_env(delay=0, pulse=2, decision=4)
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert info["pulse_frames"] == 0
    assert info["selected_action"] == 0


def test_browser_profile_random_ranges_are_respected() -> None:
    env = BrowserStorkEnv(max_steps=200)
    env.reset(seed=10)
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(1)
        assert 1 <= info["decision_frames"] <= 8
        assert 0 <= info["pulse_frames"] <= 2
        assert 0 <= info["action_delay_frames"] <= 1
        assert 0 <= info["observation_delay_frames"] <= 1
        if terminated or truncated:
            break


def test_centered_stable_state_receives_bonus() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.center_reward_per_frame = 0.1
    env.reset(seed=4)
    env.base_env.balance_velocity = 0.0

    _, reward, _, _, info = env.step(0)

    assert info["center_reward"] > 0.099
    np.testing.assert_allclose(
        reward, info["distance_reward"] + info["center_reward"]
    )


def test_repeating_one_direction_escalates_penalty_and_opposite_resets_it() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.same_direction_penalty = 0.25
    env.reset(seed=5)

    _, _, _, _, first = env.step(1)
    _, _, _, _, second = env.step(1)
    _, _, _, _, third = env.step(1)
    _, _, _, _, opposite = env.step(2)

    assert first["same_direction_penalty"] == 0.0
    assert second["same_direction_penalty"] == -0.25
    assert third["same_direction_penalty"] == -0.5
    assert opposite["same_direction_penalty"] == 0.0
    assert opposite["same_direction_streak"] == 1.0


def test_none_after_input_is_rewarded_but_direct_reversal_is_penalized() -> None:
    coast_env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    coast_env.coast_after_input_reward = 0.08
    coast_env.reset(seed=6)
    coast_env.step(1)
    _, _, _, _, coast = coast_env.step(0)

    reversal_env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    reversal_env.immediate_reversal_penalty = 0.07
    reversal_env.reset(seed=6)
    reversal_env.step(1)
    _, _, _, _, reversal = reversal_env.step(2)

    assert coast["coast_reward"] > 0.0
    assert coast["immediate_reversal_penalty"] == 0.0
    assert reversal["coast_reward"] == 0.0
    assert reversal["immediate_reversal_penalty"] == -0.07


def test_large_angular_velocity_is_rewarded_when_reduced() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.velocity_progress_reward_scale = 0.1
    env.reset(seed=7)
    env.base_env.balance_velocity = 5.0

    _, _, _, _, improved = env.step(1)

    assert improved["velocity_abs_reduction"] > 0.0
    assert improved["velocity_progress_reward"] > 0.0


def test_large_angle_only_earns_progress_reward_when_moving_toward_zero() -> None:
    improving = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    improving.angle_progress_reward_scale = 0.1
    improving.reset(seed=8)
    improving.base_env.body_rotation = 40.0
    improving.base_env.balance_velocity = -5.0
    _, _, _, _, toward_zero = improving.step(0)

    worsening = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    worsening.angle_progress_reward_scale = 0.1
    worsening.reset(seed=8)
    worsening.base_env.body_rotation = 40.0
    worsening.base_env.balance_velocity = 5.0
    _, _, _, _, away_from_zero = worsening.step(0)

    assert toward_zero["angle_progress_reward"] > 0.0
    assert away_from_zero["angle_progress_reward"] == 0.0


def test_effective_repeated_correction_reduces_repeat_penalty() -> None:
    effective = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    effective.same_direction_penalty = 0.1
    effective.reset(seed=9)
    effective.last_nonzero_action = 1
    effective.same_direction_streak = 1.0
    effective.base_env.balance_velocity = 5.0
    _, _, _, _, correcting = effective.step(1)

    ineffective = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    ineffective.same_direction_penalty = 0.1
    ineffective.reset(seed=9)
    ineffective.last_nonzero_action = 1
    ineffective.same_direction_streak = 1.0
    ineffective.base_env.balance_velocity = -5.0
    _, _, _, _, worsening = ineffective.step(1)

    assert correcting["repeat_recovery_discount"] > 0.0
    assert correcting["same_direction_penalty"] > -0.1
    assert worsening["repeat_recovery_discount"] == 0.0
    assert worsening["same_direction_penalty"] == -0.1


def test_velocity_safety_margin_rewards_below_one_and_penalizes_above() -> None:
    safe = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    safe.safe_velocity_limit = 1.0
    safe.low_velocity_reward_per_frame = 0.02
    safe.reset(seed=10)
    safe.base_env.balance_velocity = 0.25
    _, _, _, _, safe_info = safe.step(0)

    unsafe = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    unsafe.safe_velocity_limit = 1.0
    unsafe.velocity_excess_penalty_per_frame = 0.05
    unsafe.reset(seed=10)
    unsafe.base_env.balance_velocity = 2.0
    _, _, _, _, unsafe_info = unsafe.step(0)

    assert safe_info["low_velocity_reward"] > 0.0
    assert safe_info["velocity_excess_penalty"] == 0.0
    assert unsafe_info["low_velocity_reward"] == 0.0
    assert unsafe_info["velocity_excess_penalty"] < 0.0


def test_crossing_velocity_one_is_penalized_and_recovery_is_rewarded() -> None:
    crossing = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    crossing.safe_velocity_limit = 1.0
    crossing.velocity_crossing_penalty = 0.15
    crossing.reset(seed=11)
    crossing.base_env.balance_velocity = 0.5
    _, _, _, _, crossed = crossing.step(2)

    recovery = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    recovery.safe_velocity_limit = 1.0
    recovery.velocity_recovery_reward_scale = 0.2
    recovery.reset(seed=11)
    recovery.base_env.balance_velocity = 2.0
    _, _, _, _, recovered = recovery.step(1)

    assert crossed["velocity_crossing_penalty"] == -0.15
    assert crossed["velocity_limit_crossed"] is True
    assert recovered["velocity_limit_recovery_reward"] > 0.0


def test_default_profile_keeps_v7_velocity_penalties_disabled() -> None:
    env = BrowserStorkEnv(max_steps=20)

    assert env.velocity_excess_penalty_per_frame == 0.0
    assert env.velocity_crossing_penalty == 0.0
    assert env.velocity_recovery_reward_scale == 0.0


def test_default_profile_uses_original_v9_rewards() -> None:
    env = BrowserStorkEnv(max_steps=20)

    assert env.center_reward_per_frame == 0.015
    assert env.angle_progress_reward_scale == 0.06
    assert env.velocity_progress_reward_scale == 0.10
    assert env.low_velocity_reward_per_frame == 0.0


def test_projected_observation_matches_live_no_input_equations() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.observation_projection_frames_range = (2, 2)
    env.reset(seed=12)

    observation, _, _, _, info = env.step(0)
    true_state = env.base_env._observation()
    expected = BrowserStorkEnv._project_observation(true_state, 2)

    np.testing.assert_allclose(observation, expected)
    assert info["observation_projection_frames"] == 2


def test_high_velocity_excursion_rewards_recovery_only_once() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.high_velocity_entry_penalty = 0.05
    env.high_velocity_recovery_bonus = 0.04
    env.reset(seed=13)
    env.base_env.balance_velocity = 1.4

    _, _, _, _, entered = env.step(2)
    env.base_env.balance_velocity = 1.6
    _, _, _, _, recovered = env.step(1)
    _, _, _, _, after_recovery = env.step(0)

    assert entered["high_velocity_entry_penalty"] == -0.05
    assert entered["velocity_excursion_active"] is True
    assert recovered["high_velocity_recovery_bonus"] == 0.04
    assert recovered["velocity_excursion_active"] is False
    assert after_recovery["high_velocity_recovery_bonus"] == 0.0


def test_recovery_bonus_cannot_outweigh_excursion_entry_penalty() -> None:
    env = BrowserStorkEnv(
        high_velocity_entry_penalty=0.03,
        high_velocity_recovery_bonus=0.2,
    )

    assert env.high_velocity_recovery_bonus == 0.03


def test_approaching_center_curriculum_starts_inside_requested_ranges() -> None:
    env = BrowserStorkEnv(
        approaching_center_probability=1.0,
        recovery_start_probability=0.0,
        angle_noise_std=0.0,
        velocity_noise_std=0.0,
        difficulty_noise_std=0.0,
    )

    observation, info = env.reset(seed=14)

    assert info["curriculum_start_type"] == "approaching_center"
    assert 5.0 <= abs(float(observation[0])) <= 30.0
    assert 1.2 <= abs(float(observation[1])) <= 3.0
    assert float(observation[0]) * float(observation[1]) < 0.0
    assert 1.5 <= float(observation[2]) <= 5.5


def test_approaching_center_velocity_reduction_gets_targeted_reward() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.approaching_velocity_reward_scale = 0.03
    env.fast_center_crossing_threshold = 1.2
    env.reset(seed=15)
    env.base_env.body_rotation = 5.0
    env.base_env.balance_velocity = -2.0

    _, _, _, _, info = env.step(2)

    assert info["approaching_velocity_reward"] > 0.0


def test_fast_center_crossing_is_detected_and_penalized_once() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.fast_center_crossing_threshold = 1.2
    env.fast_center_crossing_penalty = 0.03
    env.reset(seed=16)
    env.base_env.body_rotation = 0.5
    env.base_env.balance_velocity = -2.0

    _, _, _, _, info = env.step(0)

    assert info["fast_center_crossing"] is True
    assert info["fast_center_crossing_penalty"] == -0.03


def test_none_near_fast_center_crossing_is_penalized() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.fast_center_crossing_threshold = 1.2
    env.missed_braking_penalty = 0.02
    env.braking_horizon_frames = 8.0
    env.reset(seed=17)
    env.base_env.body_rotation = 6.0
    env.base_env.balance_velocity = -2.0

    _, _, _, _, info = env.step(0)

    assert info["missed_braking"] is True
    assert info["missed_braking_penalty"] == -0.02


def test_dangerous_none_does_not_earn_coast_reward() -> None:
    env = make_fixed_browser_env(delay=0, pulse=1, decision=1)
    env.coast_after_input_reward = 0.04
    env.fast_center_crossing_threshold = 1.2
    env.missed_braking_penalty = 0.02
    env.braking_horizon_frames = 8.0
    env.reset(seed=18)
    env.last_decision_action = 1
    env.base_env.body_rotation = -6.0
    env.base_env.balance_velocity = 2.0

    _, _, _, _, info = env.step(0)

    assert info["missed_braking"] is True
    assert info["coast_reward"] == 0.0
