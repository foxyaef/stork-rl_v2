import numpy as np

from stork_rl import StorkEnv


def test_reset_matches_swf_load_handler() -> None:
    env = StorkEnv(max_steps=None)
    observation, info = env.reset()
    np.testing.assert_allclose(observation, [0.0, 1.0, 1.0, 0.0])
    assert info["metres"] == 0
    assert info["distance_pixels"] == 0.0


def test_first_frame_matches_actionscript_order() -> None:
    env = StorkEnv(max_steps=None)
    env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
    np.testing.assert_allclose(observation, [1.0, 1.0, 1.002, 3.0])
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["metres"] == 0


def test_second_frame_left_input_matches_actionscript() -> None:
    env = StorkEnv(max_steps=None)
    env.reset()
    env.step(0)
    observation, reward, terminated, _, _ = env.step(1)

    expected_velocity = 1.0 - 1.004 + (1.0 / 3.0) * 1.004 / 100.0
    expected_angle = 1.0 + expected_velocity
    np.testing.assert_allclose(
        observation,
        [expected_angle, expected_velocity, 1.004, 3.006],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(reward, 3.0 / 550.0 * 3.0)
    assert not terminated


def test_fall_limits_are_strict_and_checked_before_rotation_update() -> None:
    env = StorkEnv(max_steps=None)
    env.reset()
    env.body_rotation = 80.0001
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["fall_side"] == "right"
    assert info["forward_speed"] == 0.0
    assert info["balance_velocity"] == 0.0


def test_rgb_render_shape() -> None:
    env = StorkEnv(render_mode="rgb_array", max_steps=10, realtime=False)
    env.reset()
    frame = env.render()
    assert frame.shape == (400, 550, 3)
    assert frame.dtype == np.uint8
    env.close()

