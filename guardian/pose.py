"""Pose signals: lifter identification, hands-on-bar, lying/standing.

Uses a pretrained YOLOv8-pose model (COCO keypoints). Inference runs on a CROP
around the bar/bench region so the lifter fills the input (COCO models are
poor on small, lying subjects at full-frame scale).

Hands are classified with POSITIVE evidence only:
  ON      confident wrist close to the bar axis
  OFF     confident wrist clearly away from the bar axis
  UNKNOWN wrist not confidently seen, or in the ambiguous ring
An unseen wrist is never counted as "off" - bad pose output alone must not be
able to raise an alarm.

COCO indices: 5 L-shoulder 6 R-shoulder 7 L-elbow 8 R-elbow 9 L-wrist
10 R-wrist 11 L-hip 12 R-hip.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

KP_CONF = 0.35        # selection / lying-standing
WRIST_CONF = 0.50     # wrists must clear this to produce hand evidence
GRIP_FRAC = 0.09      # within this fraction of bar length = ON the bar
OFF_FRAC = 0.16       # beyond this fraction = clearly OFF (between = unknown)
DEBOUNCE = 3          # pose-frames of agreement before off-evidence is believed


@dataclass
class PoseState:
    lifter_found: bool = False
    lying: bool = True
    hands_on: int = 0          # wrists confidently ON the bar
    hands_off: int = 0         # wrists confidently OFF the bar (debounced)
    wrists: list = field(default_factory=list)
    shoulders: list = field(default_factory=list)


class PoseTracker:
    def __init__(self, model_name="yolov8n-pose.pt", every_n=2, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.every_n = every_n
        self.device = device
        self.state = PoseState()
        self._off_hist: deque[int] = deque(maxlen=DEBOUNCE)
        self._frame_i = 0

    # ------------------------------------------------------------------ crop
    @staticmethod
    def _crop_box(bar_state, W, H):
        """Region around bar + lifter below it, clamped to frame."""
        bw = max(bar_state.bar_w, 80.0)
        x1 = bar_state.cx - 0.75 * bw
        x2 = bar_state.cx + 0.75 * bw
        y1 = bar_state.cy - 0.55 * bw
        y2 = bar_state.cy + 0.95 * bw
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 - x1 < 60 or y2 - y1 < 60:
            return None
        return x1, y1, x2, y2

    def update(self, frame, bar_state) -> PoseState:
        self._frame_i += 1
        if (self._frame_i - 1) % self.every_n != 0:
            return self.state
        if not bar_state.detected:
            self.state.lifter_found = False
            return self.state

        H, W = frame.shape[:2]
        box = self._crop_box(bar_state, W, H)
        if box is None:
            self.state.lifter_found = False
            return self.state
        cx1, cy1, cx2, cy2 = box
        crop = frame[cy1:cy2, cx1:cx2]

        res = self.model.predict(crop, conf=0.25, verbose=False,
                                 device=self.device, imgsz=448)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            self.state.lifter_found = False
            return self.state

        # keypoints back to full-frame coords
        kps_all = res.keypoints.data.cpu().numpy().copy()   # (n, 17, 3)
        kps_all[:, :, 0] += cx1
        kps_all[:, :, 1] += cy1

        # --- pick the lifter: shoulders under the bar span; lying beats standing
        bar_x1 = bar_state.cx - bar_state.bar_w / 2
        bar_x2 = bar_state.cx + bar_state.bar_w / 2
        best, best_score, best_lying = None, -1e9, False
        for kps in kps_all:
            sh = [kps[i] for i in (5, 6) if kps[i][2] > KP_CONF]
            if not sh:
                continue
            sx = float(np.mean([p[0] for p in sh]))
            sy = float(np.mean([p[1] for p in sh]))
            if not (bar_x1 - 0.1 * bar_state.bar_w < sx < bar_x2 + 0.1 * bar_state.bar_w):
                continue
            hips = [kps[i] for i in (11, 12) if kps[i][2] > KP_CONF]
            lying = True
            if hips:
                hx = float(np.mean([p[0] for p in hips]))
                hy = float(np.mean([p[1] for p in hips]))
                torso = math.hypot(sx - hx, sy - hy)
                lying = torso < 1e-6 or abs(sy - hy) < 0.6 * torso
            score = -abs(sy - bar_state.cy)
            if (lying and not best_lying) or (lying == best_lying and score > best_score):
                best, best_score, best_lying = kps, score, lying

        st = self.state
        if best is None:
            st.lifter_found = False
            return st
        st.lifter_found = True
        st.lying = best_lying
        st.shoulders = [tuple(best[i][:2]) for i in (5, 6) if best[i][2] > KP_CONF]
        st.wrists = [tuple(best[i][:2]) for i in (9, 10) if best[i][2] > WRIST_CONF]

        # anatomical reach gate: a wrist farther from the shoulders than an arm
        # can reach is a mislabeled joint (knees, other people) - drop it
        hips = [best[i] for i in (11, 12) if best[i][2] > KP_CONF]
        if st.shoulders and hips:
            smx = float(np.mean([p[0] for p in st.shoulders]))
            smy = float(np.mean([p[1] for p in st.shoulders]))
            hmx = float(np.mean([p[0] for p in hips]))
            hmy = float(np.mean([p[1] for p in hips]))
            torso = math.hypot(smx - hmx, smy - hmy)
            if torso > 1e-6:
                st.wrists = [(wx, wy) for (wx, wy) in st.wrists
                             if math.hypot(wx - smx, wy - smy) <= 1.35 * torso]

        # --- hand evidence vs the bar axis
        bar_len = max(bar_state.bar_w, 1.0)
        a = math.radians(bar_state.tilt) if bar_state.tilt is not None else 0.0
        ax, ay = math.cos(a), math.sin(a)
        on = off = 0
        for (wx, wy) in st.wrists:
            rx, ry = wx - bar_state.cx, wy - bar_state.cy
            along = rx * ax + ry * ay
            perp = abs(-rx * ay + ry * ax)
            if perp < GRIP_FRAC * bar_len and abs(along) < bar_len * 0.55:
                on += 1
            elif perp > OFF_FRAC * bar_len:
                off += 1
            # in between: ambiguous -> no evidence either way
        st.hands_on = on
        self._off_hist.append(off)
        if len(self._off_hist) == DEBOUNCE and len(set(self._off_hist)) == 1:
            st.hands_off = self._off_hist[0]
        return st
