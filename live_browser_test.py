from __future__ import annotations

import argparse
import base64
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from playwright.sync_api import Locator, Page, sync_playwright

from stork_rl.agents import heuristic_action, lookahead_action, pd_action


GAME_URL = "https://vidkidz.tistory.com/2825"
FLASH_FPS = 30.0
GAME_ASPECT = 550.0 / 400.0


def wrap_line_angle(angle: float) -> float:
    """Normalize an unoriented line angle to [-90, 90)."""
    return (angle + 90.0) % 180.0 - 90.0


def unwrap_near(angle: float, reference: float) -> float:
    """Choose the 180-degree-equivalent angle nearest to reference."""
    while angle - reference > 90.0:
        angle -= 180.0
    while angle - reference < -90.0:
        angle += 180.0
    return angle


@dataclass
class Detection:
    angle: float
    line: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray


class NeckDetector:
    """Detect the stork's neck line inside a cropped 550x400 game image."""

    # Excludes the distance text, legs, and most fence posts.
    ROI = (0.36, 0.36, 0.71, 0.76)

    def __init__(self) -> None:
        self.last_angle: float | None = None
        self.last_midpoint: tuple[float, float] | None = None

    def reset(self) -> None:
        self.last_angle = None
        self.last_midpoint = None

    def detect(self, frame: np.ndarray) -> Detection | None:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 105)
        kernel_size = max(3, int(round(height * 0.007)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )

        left = int(width * self.ROI[0])
        top = int(height * self.ROI[1])
        right = int(width * self.ROI[2])
        bottom = int(height * self.ROI[3])
        roi = mask[top:bottom, left:right]

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 360.0,
            threshold=max(16, int(height * 0.04)),
            minLineLength=max(24, int(height * 0.065)),
            maxLineGap=max(8, int(height * 0.025)),
        )
        if lines is None:
            return None

        candidates: list[
            tuple[float, float, float, tuple[int, int, int, int], tuple[float, float]]
        ] = []
        target_x = width * 0.55
        target_y = height * 0.59

        for rx1, ry1, rx2, ry2 in lines[:, 0]:
            x1, y1 = int(rx1 + left), int(ry1 + top)
            x2, y2 = int(rx2 + left), int(ry2 + top)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            angle = wrap_line_angle(math.degrees(math.atan2(dx, -dy)))
            if self.last_angle is not None:
                angle = unwrap_near(angle, self.last_angle)

            mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            spatial_cost = math.hypot(
                (mid[0] - target_x) * 0.72,
                (mid[1] - target_y) * 0.35,
            )
            continuity_cost = 0.0
            if self.last_angle is not None:
                continuity_cost += abs(angle - self.last_angle) * 0.75
            if self.last_midpoint is not None:
                continuity_cost += math.hypot(
                    mid[0] - self.last_midpoint[0],
                    mid[1] - self.last_midpoint[1],
                ) * 0.25

            score = length * 1.65 - spatial_cost - continuity_cost
            candidates.append((score, length, angle, (x1, y1, x2, y2), mid))

        candidates.sort(reverse=True, key=lambda item: item[0])
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else best[0] - 30.0
        score_gap = max(0.0, best[0] - second_score)
        confidence = float(
            np.clip(0.55 * (best[1] / (height * 0.16)) + 0.45 * (score_gap / 25.0), 0, 1)
        )

        self.last_angle = best[2]
        self.last_midpoint = best[4]
        return Detection(best[2], best[3], confidence, mask)


class AlphaBetaFilter:
    """Low-latency angle/velocity estimator for irregular screenshot timing."""

    def __init__(self, alpha: float = 0.72, beta: float = 0.16) -> None:
        self.alpha = alpha
        self.beta = beta
        self.angle: float | None = None
        self.velocity_deg_s = 0.0
        self.timestamp: float | None = None

    def reset(self) -> None:
        self.angle = None
        self.velocity_deg_s = 0.0
        self.timestamp = None

    def update(self, measurement: float, timestamp: float) -> tuple[float, float]:
        if self.angle is None or self.timestamp is None:
            self.angle = measurement
            self.velocity_deg_s = 0.0
            self.timestamp = timestamp
            return self.angle, self.velocity_deg_s

        dt = float(np.clip(timestamp - self.timestamp, 1.0 / 240.0, 0.25))
        predicted = self.angle + self.velocity_deg_s * dt
        measurement = unwrap_near(measurement, predicted)
        residual = measurement - predicted
        self.angle = predicted + self.alpha * residual
        self.velocity_deg_s += self.beta * residual / dt
        self.timestamp = timestamp
        return self.angle, self.velocity_deg_s

    def state_at(self, timestamp: float) -> tuple[float, float]:
        if self.angle is None or self.timestamp is None:
            raise RuntimeError("Filter has not received a measurement")
        dt = max(0.0, timestamp - self.timestamp)
        return self.angle + self.velocity_deg_s * dt, self.velocity_deg_s


def predict_no_input(
    angle: float,
    velocity_per_frame: float,
    difficulty: float,
    frames_ahead: float,
) -> tuple[float, float, float]:
    """Project the exact SWF dynamics through the estimated pipeline delay."""
    whole_frames = int(max(0.0, frames_ahead))
    fraction = max(0.0, frames_ahead) - whole_frames
    for _ in range(whole_frames):
        difficulty += 0.002
        velocity_per_frame += (angle / 3.0) * difficulty / 100.0
        angle += velocity_per_frame
    if fraction:
        future_difficulty = difficulty + 0.002 * fraction
        future_velocity = velocity_per_frame + (
            (angle / 3.0) * future_difficulty / 100.0 * fraction
        )
        angle += future_velocity * fraction
        velocity_per_frame = future_velocity
        difficulty = future_difficulty
    return angle, velocity_per_frame, difficulty


@dataclass
class StateEstimate:
    angle: float
    velocity_per_frame: float
    difficulty: float
    measurement_accepted: bool
    innovation_deg: float | None
    source: str
    measurement_age_s: float

    @property
    def usable(self) -> bool:
        return self.source != "stale"


class PhysicsStateEstimator:
    """Fuse neck measurements with the original SWF dynamics.

    The browser detector occasionally switches to a different dark line. A
    plain alpha-beta filter turns that single jump into both a bad angle and a
    bad angular velocity. This estimator first advances the last trusted state
    with the game physics, rejects measurements that disagree too strongly,
    and can bridge a short detection gap using the predicted state.
    """

    def __init__(
        self,
        max_prediction_gap_s: float = 0.45,
        reacquire_after_s: float = 0.80,
        base_innovation_gate_deg: float = 10.0,
        confidence_gate_deg: float = 18.0,
        max_abs_velocity_per_frame: float = 6.0,
        action_effect_scale: float = 0.25,
        physics_drift_scale: float = 0.25,
    ) -> None:
        self.max_prediction_gap_s = max(0.0, max_prediction_gap_s)
        self.reacquire_after_s = max(
            self.max_prediction_gap_s, reacquire_after_s
        )
        self.base_innovation_gate_deg = max(1.0, base_innovation_gate_deg)
        self.confidence_gate_deg = max(0.0, confidence_gate_deg)
        self.max_abs_velocity_per_frame = max(
            0.1, max_abs_velocity_per_frame
        )
        self.action_effect_scale = float(
            np.clip(action_effect_scale, 0.0, 1.0)
        )
        self.physics_drift_scale = float(
            np.clip(physics_drift_scale, 0.0, 1.0)
        )
        self.reset()

    def reset(self) -> None:
        self.angle: float | None = None
        self.velocity_per_frame = 0.0
        self.difficulty = 1.0
        self.timestamp: float | None = None
        self.last_accepted_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self.angle is not None and self.timestamp is not None

    def measurement_age_at(self, timestamp: float) -> float | None:
        if self.last_accepted_timestamp is None:
            return None
        return max(0.0, timestamp - self.last_accepted_timestamp)

    def _initialize(
        self, measurement: float, timestamp: float, difficulty: float
    ) -> StateEstimate:
        self.angle = measurement
        self.velocity_per_frame = 0.0
        self.difficulty = difficulty
        self.timestamp = timestamp
        self.last_accepted_timestamp = timestamp
        return self._snapshot(
            timestamp,
            measurement_accepted=True,
            innovation_deg=0.0,
            source="measured",
        )

    def _propagate(self, timestamp: float, difficulty: float) -> float:
        if not self.initialized:
            return 0.0
        assert self.angle is not None
        assert self.timestamp is not None
        elapsed_s = max(0.0, timestamp - self.timestamp)
        if self._measurement_age(timestamp) > self.max_prediction_gap_s:
            # Once the state is too old to control from, freezing it avoids an
            # unbounded fallen-state extrapolation while waiting to reacquire.
            self.timestamp = timestamp
            self.difficulty = max(self.difficulty, difficulty)
            return 0.0
        frames = elapsed_s * FLASH_FPS
        if frames > 0.0:
            start_angle = self.angle
            start_velocity = self.velocity_per_frame
            physics_angle, physics_velocity, physics_difficulty = (
                predict_no_input(
                    start_angle,
                    start_velocity,
                    self.difficulty,
                    frames,
                )
            )
            kinematic_angle = start_angle + start_velocity * frames
            self.angle = kinematic_angle + self.physics_drift_scale * (
                physics_angle - kinematic_angle
            )
            self.velocity_per_frame = start_velocity + (
                self.physics_drift_scale
                * (physics_velocity - start_velocity)
            )
            self.difficulty = physics_difficulty
            self.timestamp = timestamp
            self.velocity_per_frame = float(
                np.clip(
                    self.velocity_per_frame,
                    -self.max_abs_velocity_per_frame,
                    self.max_abs_velocity_per_frame,
                )
            )
        # Wall-clock difficulty is known more accurately than accumulated
        # fractional updates and remains monotonic after a short capture stall.
        self.difficulty = max(self.difficulty, difficulty)
        return frames

    def _measurement_age(self, timestamp: float) -> float:
        if self.last_accepted_timestamp is None:
            return math.inf
        return max(0.0, timestamp - self.last_accepted_timestamp)

    def _snapshot(
        self,
        timestamp: float,
        *,
        measurement_accepted: bool,
        innovation_deg: float | None,
        source: str,
    ) -> StateEstimate:
        if not self.initialized or self.angle is None:
            raise RuntimeError("Estimator has no state")
        age_s = self._measurement_age(timestamp)
        if age_s > self.max_prediction_gap_s:
            source = "stale"
        return StateEstimate(
            angle=float(self.angle),
            velocity_per_frame=float(self.velocity_per_frame),
            difficulty=float(self.difficulty),
            measurement_accepted=measurement_accepted,
            innovation_deg=innovation_deg,
            source=source,
            measurement_age_s=age_s,
        )

    def update(
        self,
        measurement: float | None,
        timestamp: float,
        confidence: float,
        difficulty: float,
    ) -> StateEstimate | None:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if not self.initialized:
            if measurement is None:
                return None
            return self._initialize(measurement, timestamp, difficulty)

        if (
            measurement is not None
            and self._measurement_age(timestamp) > self.reacquire_after_s
        ):
            # After a long gap there is not enough information to distinguish
            # accumulated model error from a real state change. Re-lock cleanly.
            return self._initialize(measurement, timestamp, difficulty)

        elapsed_frames = self._propagate(timestamp, difficulty)
        if measurement is None:
            return self._snapshot(
                timestamp,
                measurement_accepted=False,
                innovation_deg=None,
                source="predicted",
            )

        assert self.angle is not None
        measurement = unwrap_near(measurement, self.angle)
        innovation = measurement - self.angle
        innovation_gate = (
            self.base_innovation_gate_deg
            + self.confidence_gate_deg * confidence
        )
        if abs(innovation) > innovation_gate:
            return self._snapshot(
                timestamp,
                measurement_accepted=False,
                innovation_deg=float(innovation),
                source="outlier",
            )

        # High-confidence measurements correct the angle quickly. Velocity is
        # adjusted more cautiously because one angle residual should not create
        # a large derivative spike.
        angle_gain = 0.35 + 0.45 * confidence
        velocity_gain = 0.08 + 0.10 * confidence
        self.angle += angle_gain * innovation
        self.velocity_per_frame += (
            velocity_gain * innovation / max(1.0, elapsed_frames)
        )
        self.velocity_per_frame = float(
            np.clip(
                self.velocity_per_frame,
                -self.max_abs_velocity_per_frame,
                self.max_abs_velocity_per_frame,
            )
        )
        self.last_accepted_timestamp = timestamp
        return self._snapshot(
            timestamp,
            measurement_accepted=True,
            innovation_deg=float(innovation),
            source="measured",
        )

    def state_at(
        self, timestamp: float, difficulty: float
    ) -> StateEstimate | None:
        if not self.initialized:
            return None
        self._propagate(timestamp, difficulty)
        return self._snapshot(
            timestamp,
            measurement_accepted=False,
            innovation_deg=None,
            source="predicted",
        )

    def apply_action(
        self,
        action: int,
        timestamp: float,
        difficulty: float,
        pulse_frames: float,
    ) -> None:
        if not self.initialized or action not in (1, 2):
            return
        self._propagate(timestamp, difficulty)
        direction = -1.0 if action == 1 else 1.0
        self.velocity_per_frame += (
            direction
            * self.difficulty
            * max(0.0, pulse_frames)
            * self.action_effect_scale
        )
        self.velocity_per_frame = float(
            np.clip(
                self.velocity_per_frame,
                -self.max_abs_velocity_per_frame,
                self.max_abs_velocity_per_frame,
            )
        )


@dataclass
class GameSessionClock:
    run_started_at: float
    session_started_at: float
    session_id: int = 1

    @classmethod
    def start(cls, timestamp: float) -> GameSessionClock:
        return cls(timestamp, timestamp)

    def run_elapsed(self, timestamp: float) -> float:
        return max(0.0, timestamp - self.run_started_at)

    def session_elapsed(self, timestamp: float) -> float:
        return max(0.0, timestamp - self.session_started_at)

    def difficulty(self, timestamp: float) -> float:
        return 1.0 + 0.002 * self.session_elapsed(timestamp) * FLASH_FPS

    def maybe_reset(
        self,
        timestamp: float,
        measurement: float | None,
        confidence: float,
        measurement_age_s: float | None,
        reset_gap_s: float,
    ) -> bool:
        if (
            reset_gap_s <= 0.0
            or measurement is None
            or measurement_age_s is None
            or measurement_age_s < reset_gap_s
            or confidence < 0.5
            or abs(measurement) > 60.0
        ):
            return False
        self.session_id += 1
        self.session_started_at = timestamp
        return True


class GameSurface:
    def __init__(
        self,
        page: Page,
        locator: Locator | None = None,
        clip: dict[str, float] | None = None,
        description: str = "",
    ) -> None:
        self.page = page
        self.locator = locator
        self.clip = clip
        self.description = description
        self.jpeg_quality: int | None = None
        self.screencast_enabled = False
        self._cdp: Any = None
        self._cast_box: tuple[float, float, float, float] | None = None
        self._cast_params: dict[str, Any] | None = None
        self._cast_arrival = 0.0
        self._cast_seq = 0
        self._cast_consumed = 0

    def start_screencast(self, quality: int, max_width: int) -> None:
        """Stream frames from Chrome (CDP Page.startScreencast) instead of
        requesting a screenshot every loop. A new frame arrives as soon as the
        page repaints, so the loop no longer waits ~100 ms per capture."""
        box = self.locator.bounding_box() if self.locator is not None else self.clip
        if not box:
            raise RuntimeError("Could not measure the game area for screencast cropping.")
        self._cast_box = (
            float(box["x"]),
            float(box["y"]),
            float(box["width"]),
            float(box["height"]),
        )
        self._cdp = self.page.context.new_cdp_session(self.page)
        self._cdp.on("Page.screencastFrame", self._on_screencast_frame)
        self._cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": int(min(100, max(1, quality))),
                "maxWidth": int(max_width),
                "maxHeight": int(max_width),
                "everyNthFrame": 1,
            },
        )
        self.screencast_enabled = True
        print(f"Screencast crop box (CSS px): {self._cast_box}")

    def _on_screencast_frame(self, params: dict[str, Any]) -> None:
        self._cast_params = params
        self._cast_arrival = time.perf_counter()
        self._cast_seq += 1
        try:
            self._cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        except Exception:
            pass

    def next_screencast_frame(
        self, timeout_s: float = 0.3
    ) -> tuple[np.ndarray | None, float | None]:
        """Wait for a frame newer than the last one used; return it cropped
        to the game area, with its arrival time (perf_counter seconds)."""
        deadline = time.perf_counter() + timeout_s
        while self._cast_seq == self._cast_consumed and time.perf_counter() < deadline:
            self.page.wait_for_timeout(1)
        params = self._cast_params
        if params is None or self._cast_seq == self._cast_consumed or self._cast_box is None:
            return None, None
        self._cast_consumed = self._cast_seq
        arrival = self._cast_arrival
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(params["data"]), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return None, None
        meta = params.get("metadata") or {}
        device_width = float(meta.get("deviceWidth") or image.shape[1])
        scale = image.shape[1] / max(1.0, device_width)
        x, y, w, h = self._cast_box
        x0, y0 = max(0, int(round(x * scale))), max(0, int(round(y * scale)))
        x1 = min(image.shape[1], int(round((x + w) * scale)))
        y1 = min(image.shape[0], int(round((y + h) * scale)))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None, None
        return image[y0:y1, x0:x1], arrival

    def enable_fast_capture(self, quality: int, fixed_clip: bool) -> None:
        """JPEG at CSS pixel size instead of PNG at device pixel size.

        PNG encoding and Windows display scaling (125-200%) dominate the
        ~100 ms capture time. With ``fixed_clip`` the element box is measured
        once and later frames skip the per-call locator checks; check the
        debug window that the crop still covers the game.
        """
        self.jpeg_quality = int(min(100, max(1, quality)))
        if fixed_clip and self.locator is not None:
            box = self.locator.bounding_box()
            if box:
                self.clip = {key: float(box[key]) for key in ("x", "y", "width", "height")}
                self.locator = None
                print(f"Fixed capture clip: {self.clip}")

    def screenshot(self) -> bytes:
        if self.jpeg_quality is not None:
            fast = {"type": "jpeg", "quality": self.jpeg_quality, "scale": "css", "caret": "initial"}
            if self.locator is not None:
                return self.locator.screenshot(**fast)
            return self.page.screenshot(clip=self.clip, **fast)
        if self.locator is not None:
            return self.locator.screenshot(type="png")
        return self.page.screenshot(type="png", clip=self.clip)


def crop_to_game_aspect(frame: np.ndarray) -> np.ndarray:
    """Remove responsive letterboxing around the original 550x400 stage."""
    height, width = frame.shape[:2]
    current_aspect = width / height
    if abs(current_aspect - GAME_ASPECT) < 0.01:
        return frame
    if current_aspect > GAME_ASPECT:
        target_width = max(1, int(round(height * GAME_ASPECT)))
        left = (width - target_width) // 2
        return frame[:, left : left + target_width]
    target_height = max(1, int(round(width / GAME_ASPECT)))
    top = (height - target_height) // 2
    return frame[top : top + target_height, :]


def parse_clip(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--clip must be x,y,width,height")
    return dict(zip(("x", "y", "width", "height"), parts, strict=True))


def discover_game_surface(page: Page, manual_clip: str | None) -> GameSurface:
    clip = parse_clip(manual_clip)
    if clip is not None:
        return GameSurface(page, clip=clip, description=f"manual clip {clip}")

    selectors = (
        "ruffle-player",
        "[data-ruffle-polyfilled]",
        "canvas",
        "object[type='application/x-shockwave-flash']",
        "embed[type='application/x-shockwave-flash']",
    )
    choices: list[tuple[float, float, Locator, str]] = []
    target_ratio = 550.0 / 400.0
    for frame in page.frames:
        for selector in selectors:
            locator = frame.locator(selector)
            for index in range(min(locator.count(), 8)):
                item = locator.nth(index)
                box = item.bounding_box()
                if not box or box["width"] < 300 or box["height"] < 220:
                    continue
                area = box["width"] * box["height"]
                ratio_error = abs(box["width"] / box["height"] - target_ratio)
                choices.append(
                    (ratio_error, -area, item, f"{selector} in {frame.url}")
                )
    if not choices:
        raise RuntimeError(
            "Could not find the game surface. Pass --clip x,y,width,height using "
            "page coordinates."
        )
    choices.sort(key=lambda item: (item[0], item[1]))
    return GameSurface(page, locator=choices[0][2], description=choices[0][3])


def install_arm_listener(page: Page) -> None:
    script = """
        () => {
            if (typeof window.__storkRlArm !== 'boolean') {
                window.__storkRlArm = false;
            }
            if (!window.__storkRlListener) {
                window.__storkRlListener = true;
                document.addEventListener('keydown', (event) => {
                    if (event.code === 'F8') window.__storkRlArm = true;
                }, true);
            }
        }
    """
    for frame in page.frames:
        try:
            frame.evaluate(script)
        except Exception:
            pass


def is_armed(page: Page) -> bool:
    for frame in page.frames:
        try:
            if frame.evaluate("() => Boolean(window.__storkRlArm)"):
                return True
        except Exception:
            pass
    return False


def load_model(model_path: Path, stats_path: Path, max_steps: int) -> tuple[Any, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from stork_rl import StorkEnv

    wrapper = DummyVecEnv([lambda: StorkEnv(render_mode=None, max_steps=max_steps)])
    normalizer = VecNormalize.load(stats_path, wrapper)
    normalizer.training = False
    normalizer.norm_reward = False
    return PPO.load(model_path), normalizer


def choose_action(
    observation: np.ndarray,
    model: Any | None,
    normalizer: Any | None,
    agent: str = "model",
    lookahead_horizon: int = 2,
    lookahead_weight: float = 16.0,
    pd_gain: float = 12.0,
    pd_deadband: float = 12.0,
) -> int:
    if agent == "pd":
        return pd_action(observation, velocity_gain=pd_gain, deadband=pd_deadband)
    if agent == "lookahead":
        return lookahead_action(
            observation, horizon=lookahead_horizon, velocity_weight=lookahead_weight
        )
    if model is None:
        return heuristic_action(observation)
    normalized = normalizer.normalize_obs(observation[None, :])
    action, _ = model.predict(normalized, deterministic=True)
    return int(action[0])


def draw_debug(
    frame: np.ndarray,
    detection: Detection | None,
    measured_body: float | None,
    filtered_angle: float | None,
    predicted_angle: float | None,
    velocity: float | None,
    state_source: str,
    innovation_deg: float | None,
    measurement_age_ms: float | None,
    session_id: int,
    session_elapsed_s: float,
    session_reset: bool,
    action: int,
    capture_ms: float,
    control_enabled: bool,
) -> np.ndarray:
    debug = frame.copy()
    height, width = debug.shape[:2]
    left = int(width * NeckDetector.ROI[0])
    top = int(height * NeckDetector.ROI[1])
    right = int(width * NeckDetector.ROI[2])
    bottom = int(height * NeckDetector.ROI[3])
    cv2.rectangle(debug, (left, top), (right, bottom), (255, 120, 0), 1)
    if detection is not None:
        cv2.line(debug, detection.line[:2], detection.line[2:], (0, 0, 255), 3)

    names = {0: "NONE", 1: "LEFT", 2: "RIGHT"}
    mode = "CONTROL" if control_enabled else "DRY RUN"
    lines = [
        (
            f"{mode}  game={session_id}  t={session_elapsed_s:.1f}s  "
            f"action={names[action]}  capture={capture_ms:.1f}ms"
        ),
        f"measure={measured_body:+.1f}" if measured_body is not None else "measure=not found",
        (
            f"filter={filtered_angle:+.1f}  predict={predicted_angle:+.1f}  "
            f"vel={velocity:+.2f} deg/frame"
            if filtered_angle is not None and predicted_angle is not None and velocity is not None
            else "filter=waiting"
        ),
        (
            f"state={state_source}  residual="
            f"{innovation_deg:+.1f}  age={measurement_age_ms:.0f}ms"
            if innovation_deg is not None and measurement_age_ms is not None
            else f"state={state_source}"
        ),
        "NEW GAME: difficulty reset" if session_reset else "",
        "Q: quit",
    ]
    lines = [line for line in lines if line]
    for index, text in enumerate(lines):
        cv2.putText(
            debug,
            text,
            (12, 26 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 220) if index == 0 else (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    return debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a trained Stork RL policy against the live Ruffle game."
    )
    parser.add_argument("--url", default=GAME_URL)
    parser.add_argument("--control", action="store_true", help="Actually send arrow keys")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--clip", help="Fallback page clip: x,y,width,height")
    parser.add_argument("--angle-scale", type=float, default=0.75)
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--extra-latency-ms", type=float, default=35.0)
    parser.add_argument("--pulse-ms", type=float, default=38.0)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--max-prediction-gap-ms", type=float, default=450.0)
    parser.add_argument("--reacquire-after-ms", type=float, default=800.0)
    parser.add_argument("--innovation-gate-deg", type=float, default=10.0)
    parser.add_argument("--confidence-gate-deg", type=float, default=18.0)
    parser.add_argument("--estimator-action-scale", type=float, default=0.8)
    parser.add_argument("--estimator-physics-scale", type=float, default=0.8)
    parser.add_argument("--session-reset-gap-ms", type=float, default=2000.0)
    parser.add_argument("--max-steps", type=int, default=18_000)
    parser.add_argument("--load-wait", type=float, default=12.0)
    parser.add_argument("--arm-immediately", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("logs/live_test.csv"))
    parser.add_argument(
        "--agent",
        choices=("model", "heuristic", "lookahead", "pd"),
        default="model",
        help="model: PPO (default, auto-loads v9); heuristic/lookahead/pd: rule-based, no model",
    )
    parser.add_argument("--lookahead-horizon", type=int, default=2)
    parser.add_argument("--lookahead-weight", type=float, default=16.0)
    parser.add_argument("--pd-gain", type=float, default=12.0)
    parser.add_argument("--pd-deadband", type=float, default=12.0)
    parser.add_argument(
        "--fast-capture",
        action="store_true",
        help="Capture JPEG at CSS pixel size instead of PNG (faster loop)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--fixed-clip",
        action="store_true",
        help="With --fast-capture: measure the game box once, then screenshot that page area",
    )
    parser.add_argument(
        "--overlap-key",
        action="store_true",
        help="Hold the arrow key while the next frame is captured instead of blocking for --pulse-ms",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=1,
        help="Redraw the debug window every N loops (higher = faster loop)",
    )
    parser.add_argument(
        "--capture",
        choices=("screenshot", "screencast"),
        default="screenshot",
        help="screencast: stream frames from Chrome (much faster than a screenshot per loop)",
    )
    parser.add_argument("--screencast-quality", type=int, default=70)
    parser.add_argument("--screencast-max-width", type=int, default=1280)
    parser.add_argument(
        "--screencast-age-ms",
        type=float,
        default=20.0,
        help="Assumed age of a streamed frame when it arrives (measurement time = arrival - this)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.model is None) != (args.stats is None):
        raise ValueError("Pass --model and --stats together")
    if args.agent != "model":
        if args.model is not None:
            print(f"--agent {args.agent}: ignoring --model/--stats.")
        args.model = args.stats = None
    elif args.model is None:
        default_pairs = (
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
        for default_model, default_stats in default_pairs:
            if default_model.exists() and default_stats.exists():
                args.model = default_model
                args.stats = default_stats
                break
    if args.control:
        print("CONTROL MODE: arrow keys will be sent after arming.")
    else:
        print("DRY-RUN MODE: no game input will be sent.")

    model = normalizer = None
    if args.model is not None and args.stats is not None:
        model, normalizer = load_model(args.model, args.stats, args.max_steps)
        print(f"Loaded model: {args.model}")
    elif args.agent == "pd":
        print(f"Using the PD rule-based controller (gain={args.pd_gain}, deadband={args.pd_deadband}).")
    elif args.agent == "lookahead":
        print(
            "Using the lookahead rule-based controller "
            f"(horizon={args.lookahead_horizon}, weight={args.lookahead_weight})."
        )
    else:
        print("No model supplied; using the predictive heuristic baseline.")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "elapsed_s", "session_id", "session_elapsed_s", "session_reset",
            "capture_ms", "detect_ms", "neck_angle",
            "body_measurement", "filtered_angle", "velocity_per_frame",
            "predicted_angle", "difficulty", "confidence",
            "measurement_accepted", "innovation_deg", "state_source",
            "measurement_age_ms", "action", "input_ms",
        ]
    )

    detector = NeckDetector()
    state_estimator = PhysicsStateEstimator(
        max_prediction_gap_s=args.max_prediction_gap_ms / 1000.0,
        reacquire_after_s=args.reacquire_after_ms / 1000.0,
        base_innovation_gate_deg=args.innovation_gate_deg,
        confidence_gate_deg=args.confidence_gate_deg,
        action_effect_scale=args.estimator_action_scale,
        physics_drift_scale=args.estimator_physics_scale,
    )
    last_frame = np.zeros((400, 550, 3), dtype=np.uint8)
    action = 0
    armed_at = 0.0
    missing_count = 0
    pending_key: str | None = None
    pending_action = 0
    press_start = 0.0
    loop_index = 0

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=False,
                args=[
                    "--window-size=1100,800",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-background-timer-throttling",
                ],
            )
            page = browser.new_page(viewport={"width": 1050, "height": 720})
            page.goto(args.url, wait_until="domcontentloaded")
            print(f"Waiting {args.load_wait:.0f}s for the game to load...")
            page.wait_for_timeout(args.load_wait * 1000)
            surface = discover_game_surface(page, args.clip)
            print(f"Game surface: {surface.description}")
            if args.fast_capture:
                surface.enable_fast_capture(args.jpeg_quality, args.fixed_clip)
                print(f"Fast capture: JPEG q={surface.jpeg_quality}, CSS scale.")
            if args.overlap_key:
                print("Overlap key: arrow stays down during the next capture.")
            if args.capture == "screencast":
                surface.start_screencast(args.screencast_quality, args.screencast_max_width)
                print("Capture: CDP screencast.")
            install_arm_listener(page)

            if not args.arm_immediately:
                print(
                    "Start the game manually. When the stork is visible, press F8 "
                    "while the browser is focused."
                )
                while not is_armed(page):
                    page.wait_for_timeout(50)
                    # New frames can appear after ads/load, so install there too.
                    install_arm_listener(page)
                print("F8 received. Controller armed.")
            else:
                print("Controller armed immediately.")
            armed_at = time.perf_counter()
            session_clock = GameSessionClock.start(armed_at)

            while True:
                loop_index += 1
                capture_start = time.perf_counter()
                frame_arrival = None
                if surface.screencast_enabled:
                    png = None
                    cast_frame, frame_arrival = surface.next_screencast_frame()
                else:
                    png = surface.screenshot()
                capture_end = time.perf_counter()
                released_hold_ms = 0.0
                if pending_key is not None:
                    # --overlap-key: the arrow pressed last loop was held during
                    # this capture. Release it now (after at least --pulse-ms).
                    remaining_s = args.pulse_ms / 1000.0 - (time.perf_counter() - press_start)
                    if remaining_s > 0.0:
                        time.sleep(remaining_s)
                    page.keyboard.up(pending_key)
                    release_time = time.perf_counter()
                    released_hold_ms = (release_time - press_start) * 1000.0
                    hold_mid = (press_start + release_time) * 0.5
                    state_estimator.apply_action(
                        pending_action,
                        hold_mid,
                        session_clock.difficulty(hold_mid),
                        released_hold_ms / 1000.0 * FLASH_FPS,
                    )
                    pending_key = None
                if png is not None:
                    frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                else:
                    frame = cast_frame
                if frame is None:
                    continue
                frame = crop_to_game_aspect(frame)
                last_frame = frame

                detect_start = time.perf_counter()
                detection = detector.detect(frame)
                detect_end = time.perf_counter()
                measurement_time = (capture_start + capture_end) * 0.5
                if frame_arrival is not None:
                    measurement_time = frame_arrival - args.screencast_age_ms / 1000.0

                measured_body = filtered_angle = predicted_angle = velocity_pf = None
                confidence = 0.0
                measurement_accepted = False
                innovation_deg = None
                state_source = "waiting"
                measurement_age_ms = None
                # With --overlap-key this logs how long last loop's key was held.
                input_ms = released_hold_ms
                action = 0

                if detection is not None:
                    confidence = detection.confidence
                    measured_body = detection.angle * args.angle_scale + args.angle_offset

                measurement = (
                    measured_body
                    if detection is not None
                    and detection.confidence >= args.min_confidence
                    else None
                )
                session_reset = session_clock.maybe_reset(
                    measurement_time,
                    measurement,
                    confidence,
                    state_estimator.measurement_age_at(measurement_time),
                    args.session_reset_gap_ms / 1000.0,
                )
                if session_reset:
                    state_estimator.reset()
                    missing_count = 0
                    print(
                        f"New game detected: session {session_clock.session_id}; "
                        "difficulty reset to 1.0."
                    )
                measurement_difficulty = session_clock.difficulty(
                    measurement_time
                )
                estimate = state_estimator.update(
                    measurement,
                    measurement_time,
                    confidence,
                    measurement_difficulty,
                )
                if estimate is not None:
                    filtered_angle = estimate.angle
                    measurement_accepted = estimate.measurement_accepted
                    innovation_deg = estimate.innovation_deg
                    state_source = estimate.source
                    measurement_age_ms = estimate.measurement_age_s * 1000.0
                    if session_reset:
                        state_source = "restarted"

                if measurement_accepted:
                    missing_count = 0
                else:
                    missing_count += 1
                    if missing_count >= 4:
                        # Reset line-continuity memory so the detector can search
                        # again, but retain the physical state for a short coast.
                        detector.reset()
                        missing_count = 0

                now = time.perf_counter()
                difficulty = session_clock.difficulty(now)
                current = state_estimator.state_at(now, difficulty)
                if current is not None and current.usable:
                    current_angle = current.angle
                    velocity_pf = current.velocity_per_frame
                    pipeline_s = (
                        now - measurement_time
                        + args.extra_latency_ms / 1000.0
                        + args.pulse_ms / 2000.0
                    )
                    predicted_angle, predicted_velocity, predicted_difficulty = (
                        predict_no_input(
                            current_angle,
                            velocity_pf,
                            difficulty,
                            pipeline_s * FLASH_FPS,
                        )
                    )
                    observation = np.array(
                        [
                            predicted_angle,
                            predicted_velocity,
                            predicted_difficulty,
                            max(0.0, (predicted_difficulty - 0.002) * 3.0),
                        ],
                        dtype=np.float32,
                    )
                    action = choose_action(
                        observation,
                        model,
                        normalizer,
                        args.agent,
                        args.lookahead_horizon,
                        args.lookahead_weight,
                        args.pd_gain,
                        args.pd_deadband,
                    )

                    if args.control and action in (1, 2) and args.overlap_key:
                        key = "ArrowLeft" if action == 1 else "ArrowRight"
                        press_start = time.perf_counter()
                        page.keyboard.down(key)
                        pending_key = key
                        pending_action = action
                    elif args.control and action in (1, 2):
                        key = "ArrowLeft" if action == 1 else "ArrowRight"
                        input_start = time.perf_counter()
                        page.keyboard.press(key, delay=args.pulse_ms)
                        input_end = time.perf_counter()
                        input_ms = (input_end - input_start) * 1000.0
                        action_time = (input_start + input_end) * 0.5
                        action_difficulty = session_clock.difficulty(
                            action_time
                        )
                        state_estimator.apply_action(
                            action,
                            action_time,
                            action_difficulty,
                            input_ms / 1000.0 * FLASH_FPS,
                        )
                else:
                    if current is not None:
                        state_source = "stale"

                if loop_index % max(1, args.debug_every) == 0:
                    debug = draw_debug(
                        frame,
                        detection,
                        measured_body,
                        filtered_angle,
                        predicted_angle,
                        velocity_pf,
                        state_source,
                        innovation_deg,
                        measurement_age_ms,
                        session_clock.session_id,
                        session_clock.session_elapsed(now),
                        session_reset,
                        action,
                        (capture_end - capture_start) * 1000.0,
                        args.control,
                    )
                    cv2.imshow("Stork live perception", debug)
                writer.writerow(
                    [
                        session_clock.run_elapsed(time.perf_counter()),
                        session_clock.session_id,
                        session_clock.session_elapsed(time.perf_counter()),
                        session_reset,
                        (capture_end - capture_start) * 1000.0,
                        (detect_end - detect_start) * 1000.0,
                        detection.angle if detection else "",
                        measured_body if measured_body is not None else "",
                        filtered_angle if filtered_angle is not None else "",
                        velocity_pf if velocity_pf is not None else "",
                        predicted_angle if predicted_angle is not None else "",
                        session_clock.difficulty(time.perf_counter()),
                        confidence,
                        measurement_accepted,
                        innovation_deg if innovation_deg is not None else "",
                        state_source,
                        measurement_age_ms if measurement_age_ms is not None else "",
                        action,
                        input_ms,
                    ]
                )
                log_file.flush()
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    if pending_key is not None:
                        page.keyboard.up(pending_key)
                    break
    finally:
        try:
            cv2.imwrite(str(args.log.with_suffix(".last.png")), last_frame)
        except Exception:
            pass
        log_file.close()
        cv2.destroyAllWindows()
        if normalizer is not None:
            normalizer.close()
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
