"""End-to-end video processing (iteration 2): detect + OBB + pose -> track ->
triggers -> annotated output. Used by the CLI (run_video.py) and web UI (app.py).
"""
from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np

from .tracker import BarTracker
from .triggers import TriggerEngine
from .pose import PoseTracker

COL_BAR = (60, 200, 255)
COL_PLATE = (80, 220, 80)
COL_TRAIL = (40, 140, 255)   # BGR orange
COL_WARN = (0, 200, 255)
COL_ALARM = (40, 40, 230)
COL_POSE = (255, 120, 200)


class GuardianPipeline:
    def __init__(self, weights: str, obb_weights: str | None = None,
                 pose_weights: str = "yolov8n-pose.pt", conf: float = 0.35,
                 device=None, use_pose: bool = True):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.obb = None
        if obb_weights and os.path.exists(obb_weights):
            self.obb = YOLO(obb_weights)
        self.pose = PoseTracker(pose_weights, every_n=2, device=device) if use_pose else None
        self.conf = conf
        self.device = device

    def detect(self, frame):
        res = self.model.predict(frame, conf=self.conf, verbose=False,
                                 device=self.device, imgsz=640)[0]
        out = []
        for b in res.boxes:
            out.append((int(b.cls.item()), float(b.conf.item()),
                        *[float(v) for v in b.xyxy[0].tolist()]))
        return out

    def detect_obb_angle(self, frame):
        """Bar angle (deg, [-90, 90], + = right side lower) from the OBB model."""
        if self.obb is None:
            return None
        res = self.obb.predict(frame, conf=0.30, verbose=False,
                               device=self.device, imgsz=640)[0]
        if res.obb is None or len(res.obb) == 0:
            return None
        i = int(res.obb.conf.argmax().item())
        x, y, w, h, r = [float(v) for v in res.obb.xywhr[i].tolist()]
        ang = math.degrees(r)
        if w < h:                       # ensure angle refers to the LONG axis
            ang += 90.0
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        return ang

    def process(self, video_path: str, out_path: str | None = None,
                events_path: str | None = None, frame_cb=None, log=print,
                max_seconds: float | None = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if out_path:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        tracker = BarTracker(fps=fps)
        engine = TriggerEngine(fps=fps, log=log)
        idx = 0
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_seconds and idx / fps > max_seconds:
                    break
                dets = self.detect(frame)
                obb_angle = self.detect_obb_angle(frame)
                st = tracker.update(frame, dets, idx, obb_angle=obb_angle)
                pose_st = self.pose.update(frame, st) if self.pose else None
                engine.update(st, pose_st)
                vis = draw_overlay(frame, st, tracker, engine, pose_st)
                if writer is not None:
                    writer.write(vis)
                if frame_cb is not None and frame_cb(vis, st, engine) is False:
                    break
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if events_path:
            with open(events_path, "w") as f:
                for ev in engine.events:
                    f.write(json.dumps(ev) + "\n")
        wall = time.time() - t0
        log(f"processed {idx} frames in {wall:.1f}s ({idx / max(wall, 1e-6):.1f} fps)")
        return engine.events


def _content_bounds(frame):
    """x-range of actual video content (excludes pillarbox black bars)."""
    W = frame.shape[1]
    small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
    col_std = small.std(axis=0) + small.mean(axis=0) * 0.1
    live = np.where(col_std > 8)[0]
    if len(live) < 4:
        return 0, W
    scale = W / 160.0
    return int(live[0] * scale), int(min(W, (live[-1] + 1) * scale))


def draw_overlay(frame, st, tracker: BarTracker, engine: TriggerEngine, pose_st=None):
    vis = frame.copy()
    H, W = vis.shape[:2]

    pts = list(tracker.trail)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue                             # gap marker: don't bridge it
        cv2.line(vis, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), COL_TRAIL, 2)

    if st.detected:
        x1 = int(st.cx - st.bar_w / 2); y1 = int(st.cy - st.bar_h / 2)
        x2 = int(st.cx + st.bar_w / 2); y2 = int(st.cy + st.bar_h / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COL_BAR, 2)
        for (pcx, pcy, pw, ph, _) in st.plates:
            cv2.rectangle(vis, (int(pcx - pw / 2), int(pcy - ph / 2)),
                          (int(pcx + pw / 2), int(pcy + ph / 2)), COL_PLATE, 2)
        # bar axis line from fused tilt
        if st.tilt is not None:
            a = math.radians(st.tilt)
            dx, dy = math.cos(a) * st.bar_w / 2, math.sin(a) * st.bar_w / 2
            cv2.line(vis, (int(st.cx - dx), int(st.cy - dy)),
                     (int(st.cx + dx), int(st.cy + dy)),
                     (255, 255, 255) if st.tilt_reliable else (140, 140, 140), 2)

    if pose_st is not None and pose_st.lifter_found:
        for (wx, wy) in pose_st.wrists:
            cv2.circle(vis, (int(wx), int(wy)), 7, COL_POSE, -1)

    lvl = engine.level
    col = {"OK": (60, 180, 60), "WARN": COL_WARN, "ALARM": COL_ALARM}[lvl]
    cv2.rectangle(vis, (12, 12), (12 + 360, 140), (25, 25, 25), -1)
    cv2.rectangle(vis, (12, 12), (12 + 360, 140), col, 2)
    label = {"OK": "MONITORING", "WARN": "WARNING", "ALARM": "!! ALARM !!"}[lvl]
    cv2.putText(vis, label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    tilt_s = (f"{st.tilt:+.1f}deg({st.tilt_src})" if st.tilt is not None else "--")
    cv2.putText(vis, f"v: {st.vy:+.2f} m/s   tilt: {tilt_s}",
                (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if pose_st is not None and pose_st.lifter_found:
        cv2.putText(vis, f"hands on:{pose_st.hands_on} off:{pose_st.hands_off}",
                    (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_POSE, 1)
    if engine.rack_y is not None:
        # clip the line to the actual video content (pillarboxed uploads have
        # dead black side bars - extrapolating across them looks absurd)
        cx0, cx1 = _content_bounds(frame)
        ppm = st.px_per_m or 1.0
        ya = engine._safety_y_at(float(cx0), ppm)
        yb = engine._safety_y_at(float(cx1), ppm)
        if ya is not None and yb is not None:
            cv2.line(vis, (cx0, int(ya)), (cx1, int(yb)), (200, 200, 60), 1)
            cv2.putText(vis, "safety line", (max(cx0, cx1 - 150), int(yb) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 60), 1)
    reason = ""
    for ev in reversed(engine.events):
        if ev["level"] in ("WARN", "ALARM"):
            reason = f'{ev["reason"]} @ {ev["t"]:.1f}s'
            break
    cv2.putText(vis, reason, (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    if lvl == "ALARM" and (st.frame_idx // 8) % 2 == 0:
        cv2.rectangle(vis, (0, 0), (W - 1, H - 1), COL_ALARM, 10)
    return vis
