import numpy as np
import pytest

pytest.importorskip("cv2")

from live_browser_test import (
    AlphaBetaFilter,
    GameSessionClock,
    PhysicsStateEstimator,
    crop_to_game_aspect,
    predict_no_input,
    unwrap_near,
)


def test_angle_unwrap_uses_nearest_line_equivalent() -> None:
    assert unwrap_near(-88.0, 89.0) == pytest.approx(92.0)
    assert unwrap_near(88.0, -89.0) == pytest.approx(-92.0)


def test_alpha_beta_filter_tracks_constant_motion() -> None:
    filter_ = AlphaBetaFilter(alpha=0.8, beta=0.25)
    for index in range(20):
        angle, velocity = filter_.update(index * 2.0, index * 0.05)
    assert angle == pytest.approx(38.0, abs=1.0)
    assert velocity == pytest.approx(40.0, abs=6.0)


def test_latency_projection_matches_one_exact_no_input_frame() -> None:
    angle, velocity, difficulty = predict_no_input(30.0, 2.0, 1.0, 1.0)
    expected_difficulty = 1.002
    expected_velocity = 2.0 + (30.0 / 3.0) * expected_difficulty / 100.0
    assert difficulty == pytest.approx(expected_difficulty)
    assert velocity == pytest.approx(expected_velocity)
    assert angle == pytest.approx(30.0 + expected_velocity)


def test_projection_returns_finite_fractional_state() -> None:
    state = predict_no_input(-20.0, -0.5, 1.2, 2.4)
    assert np.all(np.isfinite(state))


def test_responsive_canvas_is_center_cropped_to_flash_aspect() -> None:
    frame = np.zeros((505, 960, 3), dtype=np.uint8)
    cropped = crop_to_game_aspect(frame)
    assert cropped.shape == (505, 694, 3)
    assert cropped.shape[1] / cropped.shape[0] == pytest.approx(550 / 400, abs=0.002)


def test_physics_estimator_rejects_large_low_confidence_jump() -> None:
    estimator = PhysicsStateEstimator(physics_drift_scale=1.0)
    estimator.update(0.0, 0.0, confidence=1.0, difficulty=1.0)
    trusted = estimator.update(2.0, 0.1, confidence=0.9, difficulty=1.006)
    assert trusted is not None and trusted.measurement_accepted

    expected_angle, expected_velocity, _ = predict_no_input(
        trusted.angle,
        trusted.velocity_per_frame,
        trusted.difficulty,
        3.0,
    )
    rejected = estimator.update(-30.0, 0.2, confidence=0.4, difficulty=1.012)

    assert rejected is not None
    assert rejected.measurement_accepted is False
    assert rejected.source == "outlier"
    assert rejected.angle == pytest.approx(expected_angle)
    assert rejected.velocity_per_frame == pytest.approx(expected_velocity)


def test_physics_estimator_bridges_short_gap_then_becomes_stale() -> None:
    estimator = PhysicsStateEstimator(max_prediction_gap_s=0.45)
    estimator.update(5.0, 0.0, confidence=1.0, difficulty=1.0)

    short_gap = estimator.update(None, 0.3, confidence=0.0, difficulty=1.018)
    long_gap = estimator.update(None, 0.6, confidence=0.0, difficulty=1.036)

    assert short_gap is not None and short_gap.usable
    assert short_gap.source == "predicted"
    assert long_gap is not None and not long_gap.usable
    assert long_gap.source == "stale"


def test_physics_estimator_applies_known_key_pulse() -> None:
    estimator = PhysicsStateEstimator(action_effect_scale=1.0)
    estimator.update(0.0, 0.0, confidence=1.0, difficulty=1.0)
    estimator.apply_action(
        action=1,
        timestamp=0.1,
        difficulty=1.006,
        pulse_frames=1.0,
    )
    state = estimator.state_at(0.1, difficulty=1.006)

    assert state is not None
    assert state.velocity_per_frame == pytest.approx(-1.006)


def test_physics_estimator_relocks_after_long_gap() -> None:
    estimator = PhysicsStateEstimator(
        max_prediction_gap_s=0.4,
        reacquire_after_s=0.8,
    )
    estimator.update(0.0, 0.0, confidence=1.0, difficulty=1.0)
    reacquired = estimator.update(
        25.0,
        0.9,
        confidence=0.9,
        difficulty=1.054,
    )

    assert reacquired is not None
    assert reacquired.measurement_accepted
    assert reacquired.source == "measured"
    assert reacquired.angle == pytest.approx(25.0)
    assert reacquired.velocity_per_frame == 0.0


def test_game_session_clock_resets_difficulty_after_long_gap() -> None:
    clock = GameSessionClock.start(100.0)
    assert clock.difficulty(110.0) == pytest.approx(1.6)

    reset = clock.maybe_reset(
        timestamp=110.0,
        measurement=12.0,
        confidence=0.8,
        measurement_age_s=4.0,
        reset_gap_s=2.0,
    )

    assert reset
    assert clock.session_id == 2
    assert clock.run_elapsed(110.0) == pytest.approx(10.0)
    assert clock.session_elapsed(110.0) == 0.0
    assert clock.difficulty(110.0) == 1.0


@pytest.mark.parametrize(
    ("measurement", "confidence", "age_s", "gap_s"),
    [
        (10.0, 0.9, 2.9, 3.0),
        (10.0, 0.4, 4.0, 3.0),
        (70.0, 0.9, 4.0, 3.0),
        (10.0, 0.9, 4.0, 0.0),
    ],
)
def test_game_session_clock_ignores_unsafe_reset_candidates(
    measurement: float,
    confidence: float,
    age_s: float,
    gap_s: float,
) -> None:
    clock = GameSessionClock.start(0.0)

    assert not clock.maybe_reset(
        timestamp=4.0,
        measurement=measurement,
        confidence=confidence,
        measurement_age_s=age_s,
        reset_gap_s=gap_s,
    )
    assert clock.session_id == 1
