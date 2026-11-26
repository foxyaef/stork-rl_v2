"""Replica of the live browser pipeline, for offline evaluation.

Simulates, in continuous wall-clock time:
  game frames at 30 FPS (original SWF physics) -> screenshot (image = end of
  capture window) -> noisy neck measurement -> the live PhysicsStateEstimator
  (copied logic) -> no-input projection over the live 'pipeline' horizon ->
  policy -> blocking key press (~47 ms) -> debug/CSV overhead -> next capture.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

FPS = 30.0


def predict_no_input(angle, vel, diff, frames):
    whole = int(max(0.0, frames)); frac = max(0.0, frames) - whole
    for _ in range(whole):
        diff += 0.002
        vel += (angle / 3.0) * diff / 100.0
        angle += vel
    if frac:
        fd = diff + 0.002 * frac
        fv = vel + (angle / 3.0) * fd / 100.0 * frac
        angle += fv * frac
        vel, diff = fv, fd
    return angle, vel, diff


class Estimator:
    """Same update rules as live_browser_test.PhysicsStateEstimator."""

    def __init__(self, action_scale=0.8, physics_scale=0.8, max_gap=0.45, reacquire=0.80,
                 gate=10.0, conf_gate=18.0, vmax=6.0, angle_gain=(0.35, 0.45), vel_gain=(0.08, 0.10)):
        self.a_s, self.p_s = action_scale, physics_scale
        self.max_gap, self.reacq = max_gap, max(max_gap, reacquire)
        self.gate, self.cgate, self.vmax = gate, conf_gate, vmax
        self.ag, self.vg = angle_gain, vel_gain
        self.angle = None; self.vel = 0.0; self.diff = 1.0; self.t = None; self.last_ok = None

    def _age(self, t):
        return math.inf if self.last_ok is None else max(0.0, t - self.last_ok)

    def _init(self, z, t, d):
        self.angle, self.vel, self.diff, self.t, self.last_ok = z, 0.0, d, t, t

    def _prop(self, t, d):
        if self.angle is None:
            return 0.0
        el = max(0.0, t - self.t)
        if self._age(t) > self.max_gap:
            self.t = t; self.diff = max(self.diff, d); return 0.0
        fr = el * FPS
        if fr > 0:
            pa, pv, pd = predict_no_input(self.angle, self.vel, self.diff, fr)
            ka = self.angle + self.vel * fr
            self.angle = ka + self.p_s * (pa - ka)
            self.vel = self.vel + self.p_s * (pv - self.vel)
            self.diff = pd; self.t = t
            self.vel = max(-self.vmax, min(self.vmax, self.vel))
        self.diff = max(self.diff, d)
        return fr

    def update(self, z, t, conf, d):
        if self.angle is None:
            if z is None:
                return None
            self._init(z, t, d); return "measured"
        if z is not None and self._age(t) > self.reacq:
            self._init(z, t, d); return "measured"
        fr = self._prop(t, d)
        if z is None:
            return "predicted"
        inn = z - self.angle
        if abs(inn) > self.gate + self.cgate * conf:
            return "outlier"
        self.angle += (self.ag[0] + self.ag[1] * conf) * inn
        self.vel += (self.vg[0] + self.vg[1] * conf) * inn / max(1.0, fr)
        self.vel = max(-self.vmax, min(self.vmax, self.vel))
        self.last_ok = t
        return "measured"

    def state_at(self, t, d):
        if self.angle is None:
            return None
        self._prop(t, d)
        return None if self._age(t) > self.max_gap else (self.angle, self.vel, self.diff)

    def apply_action(self, a, t, d, pulse_frames):
        if self.angle is None or a not in (1, 2):
            return
        self._prop(t, d)
        self.vel += (-1.0 if a == 1 else 1.0) * self.diff * max(0.0, pulse_frames) * self.a_s
        self.vel = max(-self.vmax, min(self.vmax, self.vel))


@dataclass
class LiveConfig:
    action_scale: float = 0.8
    physics_scale: float = 0.8
    extra_latency: float = 0.035      # s, live --extra-latency-ms
    pulse: float = 0.038              # s, live --pulse-ms (used in the horizon formula)
    capture: tuple = (0.088, 0.110)   # s, screenshot call duration
    image_lag: tuple = (0.0, 0.010)   # s, image older than capture end
    hold: tuple = (0.044, 0.052)      # s, actual key-press duration (blocking press)
    key_latency: tuple = (0.004, 0.012)
    overhead: tuple = (0.004, 0.012)  # s, debug draw + csv + waitKey
    noise: float = 0.85               # deg, measurement noise on body angle
    conf: tuple = (0.6, 1.0)
    outlier_p: float = 0.005
    miss_p: float = 0.003
    max_seconds: float = 600.0


class LiveReplica:
    def __init__(self, cfg: LiveConfig, seed: int):
        self.cfg = cfg; self.rng = random.Random(seed)
        # true game state (StorkEnv order)
        self.rot, self.bs, self.bi, self.xs, self.xx = 0.0, 1.0, 1.0, 0.0, 0.0
        self.frame = 0; self.fallen = False
        self.keys = []  # (start, end, direction)
        self.est = Estimator(cfg.action_scale, cfg.physics_scale)
        self.t = self.rng.uniform(0.0, 1.0 / FPS)  # wall clock (s since game start)

    def _key_dir(self, tick):
        for a, b, d in self.keys:
            if a <= tick <= b:
                return d
        return 0

    def _advance_to(self, t):
        while not self.fallen and (self.frame + 1) / FPS <= t:
            tick = (self.frame + 1) / FPS
            self.xx += self.xs; self.xs = self.bi * 3.0; self.bi += 0.002
            d = self._key_dir(tick)
            if d == 1: self.bs -= self.bi
            elif d == 2: self.bs += self.bi
            if self.rot < -110 or self.rot > 80:
                self.fallen = True; break
            self.bs += (self.rot / 3.0) * self.bi / 100.0
            self.rot += self.bs
            self.frame += 1
        self.keys = [k for k in self.keys if k[1] >= t - 0.2]

    def metres(self):
        return math.floor(self.xx / 550.0 * 3.0)

    def run(self, policy):
        c, r = self.cfg, self.rng
        while not self.fallen and self.t < c.max_seconds:
            cap_start = self.t
            cap = r.uniform(*c.capture)
            img_t = cap_start + cap - r.uniform(*c.image_lag)
            self._advance_to(img_t)
            if self.fallen: break
            z = self.rot + r.gauss(0.0, c.noise)
            conf = r.uniform(*c.conf)
            u = r.random()
            if u < c.miss_p: z = None
            elif u < c.miss_p + c.outlier_p: z = r.uniform(-60, 60)
            m_t = cap_start + cap / 2.0
            now = cap_start + cap + 0.003
            self._advance_to(now)
            if self.fallen: break
            d_now = 1.0 + 0.002 * (now * FPS)
            self.est.update(z, m_t, conf, 1.0 + 0.002 * (m_t * FPS))
            st = self.est.state_at(now, d_now)
            action = 0
            if st is not None:
                horizon = (now - m_t) + c.extra_latency + c.pulse / 2.0
                pa, pv, pd = predict_no_input(st[0], st[1], st[2], horizon * FPS)
                obs = (pa, pv, pd, max(0.0, (pd - 0.002) * 3.0))
                action = policy(obs)
            t_next = now
            if action in (1, 2):
                a = now + 0.001; h = r.uniform(*c.hold); b = a + h
                lat = r.uniform(*c.key_latency)
                self.keys.append((a + lat, b + lat, action))
                self._advance_to(b)
                mid = (a + b) / 2.0
                self.est.apply_action(action, mid, 1.0 + 0.002 * mid * FPS, h * FPS)
                t_next = b
            self.t = t_next + r.uniform(*c.overhead)
            self._advance_to(self.t)
        return self.metres(), self.t
