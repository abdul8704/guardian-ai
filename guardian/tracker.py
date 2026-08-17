"""Bar tracking: per-frame detection -> smoothed bar state.

Consumes YOLO detections (class 0 = barbell, class 1 = bar_plates) and produces
a BarState per frame: center, velocity (m/s), tilt (deg), plate axis positions,
pixel->meter scale. Includes an innovation-gated 1D Kalman filter on bar height
(so free-fall is never smoothed away) and a camera-cut guard.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

BAR_LENGTH_M = 2.2  # standard olympic bar


@dataclass
class BarState:
    frame_idx: int
    t: float                       # seconds
    detected: bool = False
    cx: float = 0.0                # bar center, px
    cy: float = 0.0
    cy_smooth: float = 0.0         # kalman-smoothed height
    vy_raw: float = 0.0            # m/s, +down (image y grows downward)
    vy: float = 0.0                # m/s, smoothed
    tilt: float | None = None      # deg (fused: OBB > plate-line > held)
    tilt_reliable: bool = False    # False when held/foreshortened -> never alarm on it
    tilt_src: str = "-"            # obb | plates | held
    bar_w: float = 0.0             # barbell box width px
    bar_h: float = 0.0
    px_per_m: float = 0.0
    conf: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    plates: list = field(default_factory=list)   # [(cx, cy, w, h, conf), ...] max 2
    plate_axis: list = field(default_factory=list)  # signed axis pos in bar-lengths
    cut: bool = False              # camera cut detected this frame


class KalmanY:
    """1D constant-velocity Kalman on bar height with innovation gate."""

    def __init__(self, q=60.0, r=25.0, gate_px=None):
        self.x = None       # [y, vy_px]
        self.P = np.eye(2) * 500.0
        self.q = q
        self.r = r
        self.gate_px = gate_px

    def reset(self):
        self.x = None
        self.P = np.eye(2) * 500.0

    def update(self, y_meas: float, dt: float, gate_px: float):
        if self.x is None:
            self.x = np.array([y_meas, 0.0])
            return y_meas, 0.0
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        innov = y_meas - x_pred[0]
        # innovation gate: a jump far beyond prediction means violent motion
        # (drop) -> trust the measurement, do not smooth it away
        if abs(innov) > gate_px:
            vy = (y_meas - self.x[0]) / dt if dt > 0 else 0.0
            self.x = np.array([y_meas, vy])
            self.P = np.eye(2) * 200.0
            return y_meas, vy
        S = P_pred[0, 0] + self.r
        K = P_pred[:, 0] / S
        self.x = x_pred + K * innov
        self.P = P_pred - np.outer(K, P_pred[0, :])
        return float(self.x[0]), float(self.x[1])


class CutGuard:
    """Detects hard camera cuts via global frame difference."""

    def __init__(self, thresh=28.0):
        self.prev = None
        self.thresh = thresh

    def check(self, frame) -> bool:
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = False
        if self.prev is not None:
            cut = float(np.abs(small - self.prev).mean()) > self.thresh
        self.prev = small
        return cut


class BarTracker:
    def __init__(self, fps: float):
        self.fps = fps
        self.kf = KalmanY()
        self.cutguard = CutGuard()
        self.scale_hist: deque[float] = deque(maxlen=90)   # bar width px history
        self.prev_state: BarState | None = None
        self.trail: deque[tuple[float, float]] = deque(maxlen=150)
        self.tilt_held: float | None = None

    def reset_motion(self):
        self.kf.reset()
        self.trail.clear()
        self.prev_state = None

    @property
    def px_per_m(self) -> float:
        if not self.scale_hist:
            return 0.0
        return float(np.median(self.scale_hist)) / BAR_LENGTH_M

    def update(self, frame, dets, frame_idx: int, obb_angle: float | None = None) -> BarState:
        """dets: list of (cls, conf, x1, y1, x2, y2) in pixel coords.
        obb_angle: bar angle in degrees from the OBB model (optional)."""
        t = frame_idx / self.fps
        st = BarState(frame_idx=frame_idx, t=t)
        st.frame_h, st.frame_w = frame.shape[:2]
        st.cut = self.cutguard.check(frame)
        if st.cut:
            self.reset_motion()
            self.tilt_held = None

        bars = [d for d in dets if d[0] == 0]
        plates = [d for d in dets if d[0] == 1]
        if not bars:
            if self.trail and self.trail[-1] is not None:
                self.trail.append(None)          # break trail across detection gaps
            self.prev_state = st
            return st

        _, conf, x1, y1, x2, y2 = max(bars, key=lambda d: d[1])
        st.detected = True
        st.conf = conf
        st.cx, st.cy = (x1 + x2) / 2, (y1 + y2) / 2
        st.bar_w, st.bar_h = x2 - x1, y2 - y1
        # scale: only trust near-level bar widths (tilt shrinks the AABB width)
        self.scale_hist.append(st.bar_w)
        st.px_per_m = self.px_per_m

        # plates: keep the two highest-conf plates that plausibly belong to the bar
        cand = []
        for p in plates:
            pcx, pcy = (p[2] + p[4]) / 2, (p[3] + p[5]) / 2
            if x1 - st.bar_w * 0.35 < pcx < x2 + st.bar_w * 0.35:
                cand.append((p[1], pcx, pcy, p[4] - p[2], p[5] - p[3]))
        cand.sort(reverse=True)
        # at most one plate per side of bar center
        left = [c for c in cand if c[1] < st.cx]
        right = [c for c in cand if c[1] >= st.cx]
        chosen = ([left[0]] if left else []) + ([right[0]] if right else [])
        st.plates = [(c[1], c[2], c[3], c[4], c[0]) for c in chosen]

        # tilt fusion: OBB (from the bar itself) > plate-pair line > held value
        plate_tilt = None
        if len(chosen) == 2:
            (c1, c2) = chosen[0], chosen[1]
            dx, dy = c2[1] - c1[1], c2[2] - c1[2]
            if abs(dx) > 1:
                plate_tilt = math.degrees(math.atan2(dy, dx))
                if plate_tilt > 90:
                    plate_tilt -= 180
                elif plate_tilt < -90:
                    plate_tilt += 180
        # foreshortening check: apparent bar length shrunk while nothing else explains it
        foreshortened = False
        if self.scale_hist and len(self.scale_hist) > 30:
            cal = float(np.median(self.scale_hist))
            if st.bar_w < 0.75 * cal and (plate_tilt is None or abs(plate_tilt) < 10):
                foreshortened = True   # bar rotated toward/away from camera
        if obb_angle is not None and not foreshortened:
            st.tilt, st.tilt_reliable, st.tilt_src = obb_angle, True, "obb"
            self.tilt_held = obb_angle
        elif plate_tilt is not None and not foreshortened:
            st.tilt, st.tilt_reliable, st.tilt_src = plate_tilt, True, "plates"
            self.tilt_held = plate_tilt
        elif self.tilt_held is not None:
            st.tilt, st.tilt_reliable, st.tilt_src = self.tilt_held, False, "held"

        # plate positions along bar axis, in bar lengths (for slide detection)
        bar_len_px = max(st.bar_w, 1.0)
        axis = np.array([1.0, 0.0])
        if st.tilt is not None:
            a = math.radians(st.tilt)
            axis = np.array([math.cos(a), math.sin(a)])
        for (pcx, pcy, _, _, _) in st.plates:
            rel = np.array([pcx - st.cx, pcy - st.cy])
            st.plate_axis.append(float(np.dot(rel, axis) / bar_len_px))

        # velocities (m/s, +down)
        dt = 1.0 / self.fps
        ppm = st.px_per_m or 1.0
        if self.prev_state is not None and self.prev_state.detected:
            st.vy_raw = (st.cy - self.prev_state.cy) / dt / ppm
        gate_px = max(0.12 * bar_len_px, 25.0)
        cy_s, vy_px = self.kf.update(st.cy, dt, gate_px)
        st.cy_smooth = cy_s
        st.vy = vy_px / ppm

        # trail: smoothed points; break the line on jumps (misdetection/reacquire)
        if self.trail:
            last = self.trail[-1]
            if last is not None and math.hypot(st.cx - last[0], cy_s - last[1]) > \
                    max(0.35 * bar_len_px, 60):
                self.trail.append(None)          # gap marker - no cross-screen slash
        self.trail.append((st.cx, cy_s))
        self.prev_state = st
        return st
