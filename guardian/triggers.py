"""Set/rep state machine, per-lifter baselines, and triggers T1-T7 (iteration 2).

Consumes BarState (+ optional PoseState) per frame, emits WARN/ALARM/RECOVERY/INFO.
All thresholds in meters/seconds/degrees; px converted via BarState.px_per_m.

Iteration-2 changes:
  T4  tilt measured as deviation from a per-set tilt baseline (camera-angle proof)
  T5  plate slide v2: plate-to-bar-end distance vs per-set baseline + detached plate
      + plate-count drop, with a settling period
  T6  bar on body while <2 hands gripping (pose)
  T7  elbow lockout failure (pose): incomplete lockout + stalled bar -> WARN,
      descent without recovery -> ALARM
  +   bar-lost-mid-rep WARN / reappeared-low ALARM
  +   recovery requires an upward crossing into the original rack zone
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

from .tracker import BarState

# ---- tunables ---------------------------------------------------------------
V_EPS = 0.05           # m/s "motionless"
V_UP = 0.08            # m/s counts as rising
SINK_MARGIN_M = 0.06
T1_NO_RISE_S = 1.5
T2_HOLD_S = 5.0
T3_DROP_MS = 1.5
T3_BASELINE_X = 3.0
T4_DEV_SUST = 12.0     # deg deviation from set tilt baseline, sustained
T4_SUST_S = 1.0
T4_DEV_INSTANT = 20.0   # measurements are reliability-gated + same-source-ref
T5_SETTLE_S = 2.0
T5_GROW_FRAC = 0.08    # plate->bar-end distance growth (bar lengths)
T5_SUST_FRAMES = 4
T5_MISS_WARN_S = 0.7   # plate missing on an in-frame end
T5_MISS_ALARM_S = 1.5
T6_HANDSOFF_S = 1.2
T6_ALARM_ENABLED = False   # WARN-only until pose is fine-tuned on lifter data:
                           # COCO-pretrained keypoints on lying lifters are not
                           # alarm-grade evidence (proven by false alarms)
T7_STALL_S = 2.0
T7_TOP_MARGIN_M = 0.05    # safety line this far below the unrack lockout height
REP_MIN_TRAVEL_M = 0.15
ALARM_COOLDOWN_S = 8.0
LOST_DEPTH_M = 0.25   # bar-lost only counts when meaningfully deep in a rep


@dataclass
class Rep:
    start_t: float
    bottom_y: float = 0.0
    bottom_t: float = 0.0
    hold_s: float = 0.0
    max_desc: float = 0.0
    top_y: float | None = None      # highest bar point of the rep (min cy)
    top_cx: float | None = None     # bar x at that highest point
    done: bool = False


@dataclass
class Baseline:
    bottom_y: float | None = None
    hold_s: float = 1.0
    max_desc: float = 0.6
    top_y: float | None = None      # median highest point of completed reps
    top_cx: float | None = None     # median bar x at rep tops (line anchor)
    n_reps: int = 0


class TriggerEngine:
    def __init__(self, fps: float, log=print):
        self.fps = fps
        self.log = log
        self.state = "IDLE"
        self.level = "OK"
        self.events: list[dict] = []
        self.baseline = Baseline()
        self.reps: list[Rep] = []
        self.cur_rep: Rep | None = None
        self.phase = "top"
        self.rack_y: float | None = None
        self.rack_cx: float | None = None
        self.set_t0 = 0.0
        self.start_y_hist: list[float] = []
        self.start_cx_hist: list[float] = []
        self.cy_hist: deque[tuple[float, float]] = deque(maxlen=90)   # (t, cy_m below rack)
        # tilt (per-source refs: obb and plate-line can disagree by a constant offset)
        self.tilt_ref: dict[str, float] = {}
        self.tilt_calib: dict[str, list[float]] = {}
        self.tilt_since: float | None = None
        # T1/T2
        self.sink_since: float | None = None
        self.still_since: float | None = None
        # T3
        self.drop_frames = 0
        # T5
        self.end_base: dict[int, list[float]] = {0: [], 1: []}   # side -> dist samples
        self.end_grow: dict[int, int] = {0: 0, 1: 0}
        self.plate_seen: dict[int, float] = {0: 0.0, 1: 0.0}     # accumulated seen-time
        self.plate_miss_since: dict[int, float | None] = {0: None, 1: None}
        # T6
        self.handsoff_since: float | None = None
        # T7
        self.stall_since: float | None = None
        self.lockout_warned = False
        self.lockout_pending_desc = False
        self.lockout_still_since: float | None = None
        self.asc_min_y: float | None = None
        self.asc_min_cx: float | None = None
        # safety line: marked ONCE at the unrack lockout (highest bar position
        # between leaving the hooks and the first descent), fixed thereafter
        self.safety_y0: float | None = None
        self.safety_x0: float | None = None
        self.pre_top_y: float | None = None
        self.pre_top_cx: float | None = None
        # lost bar
        self.lost_since: float | None = None
        self.lost_warned = False
        self.lost_warn_t = -1e9
        self.last_depth = 0.0
        self.last_alarm_t = -1e9
        self.missing_frames = 0
        self.handling = False
        self.last_lying_t = -1e9
        self.tilt_dips = 0

    # ------------------------------------------------------------------ events
    def _emit(self, t, level, reason, detail):
        # a positively-detected STANDING person under the bar = bar handling
        # (unracking plates, walking the bar) -> not an emergency
        if level in ("WARN", "ALARM") and self.handling:
            return
        ev = {"t": round(t, 2), "level": level, "reason": reason, "detail": detail}
        self.events.append(ev)
        self.log(f"[{level}] t={t:6.2f}s  {reason}  |  {detail}")
        if level == "ALARM":
            self.level = "ALARM"
            self.last_alarm_t = t
        elif level == "WARN" and self.level != "ALARM":
            self.level = "WARN"
        elif level == "RECOVERY":
            self.level = "OK"

    def _alarm_ok(self, t):
        return t - self.last_alarm_t > ALARM_COOLDOWN_S

    # ------------------------------------------------------------------ update
    def update(self, st: BarState, pose=None):
        t = st.t
        # "A nearby person never cancels an alarm" (doc). Suppress only for clear
        # teardown: a standing person at the bar AND no lying lifter seen under
        # the bar for 8s. A spotter leaning over a pinned lifter never silences.
        if pose is not None and pose.lifter_found and pose.lying:
            self.last_lying_t = t
        standing = bool(pose is not None and pose.lifter_found and not pose.lying)
        # handling suppression additionally requires the bar near rack height:
        # a bar deep at chest level is NEVER suppressible - pose must not hold
        # veto power over bar-based alarms in a potential emergency
        near_rack_now = (self.rack_y is None or st.px_per_m == 0 or
                         (st.cy_smooth - self.rack_y) / (st.px_per_m or 1) < 0.15)
        self.handling = standing and (t - self.last_lying_t > 8.0) and near_rack_now
        if st.cut:
            self._reset_set("camera cut")

        if not st.detected:
            self.missing_frames += 1
            # bar lost suddenly while deep in a rep -> suspicious (Tuomas case)
            if (self.state == "ACTIVE" and self.last_depth > LOST_DEPTH_M
                    and self.phase in ("descent", "bottom", "ascent")
                    and not self.lost_warned and self.missing_frames >= 3
                    and t - self.lost_warn_t > 5.0):
                self.lost_warned = True
                self.lost_since = t
                self.lost_warn_t = t
                self._emit(t, "WARN", "bar_lost_midrep",
                           f"bar vanished at depth {self.last_depth:.2f}m")
            if self.missing_frames > self.fps * 3 and self.state == "ACTIVE":
                self._reset_set("bar lost >3s")
            return
        self.missing_frames = 0
        ppm = st.px_per_m or 1.0

        # reappeared low/tilted after a mid-rep loss -> alarm
        if self.lost_warned and self.lost_since is not None:
            if t - self.lost_since < 3.0:
                dev = self._tilt_dev(st)
                depth_now = ((st.cy_smooth - self.rack_y) / ppm) if self.rack_y else 0
                if (depth_now > 0.20 or (dev is not None and abs(dev) > T4_DEV_SUST)) \
                        and self._alarm_ok(t):
                    self._emit(t, "ALARM", "bar_reappeared_low",
                               f"after loss: depth={depth_now:.2f}m dev={dev}")
            self.lost_warned = False
            self.lost_since = None

        # --- set activation & rack calibration
        if self.state == "IDLE":
            self.start_y_hist.append(st.cy)
            self.start_cx_hist.append(st.cx)
            if len(self.start_y_hist) >= int(self.fps * 0.7):
                n = int(self.fps * 0.7)
                self.rack_y = statistics.median(self.start_y_hist[:n])
                self.rack_cx = statistics.median(self.start_cx_hist[:n])
                self.state = "ACTIVE"
                self.set_t0 = t
                self._emit(t, "INFO", "set_active",
                           f"rack_y={self.rack_y:.0f}px scale={ppm:.0f}px/m")
            return

        depth_m = (st.cy_smooth - self.rack_y) / ppm
        self.last_depth = depth_m
        self.cy_hist.append((t, depth_m))

        # --- tilt baseline calibration: the reference "level" is whatever this
        # camera sees when the bar is guaranteed level for this lifter - racked,
        # or at the top of a rep. Per source (obb / plate-line offsets differ).
        # Each source self-calibrates from its first 15 reliable samples taken
        # at those safe moments; tilt deviation is always vs the matching ref.
        if (st.tilt is not None and st.tilt_reliable
                and st.tilt_src not in self.tilt_ref
                and self.level == "OK"
                and (self.phase == "top" or self._near_rack(st, ppm)
                     or t - self.set_t0 <= 1.5)):
            buf = self.tilt_calib.setdefault(st.tilt_src, [])
            buf.append(st.tilt)
            if len(buf) >= 10:
                self.tilt_ref[st.tilt_src] = statistics.median(buf)
                self._emit(t, "INFO", "tilt_ref",
                           f"{st.tilt_src}:{self.tilt_ref[st.tilt_src]:+.1f}deg "
                           f"(relative baseline)")

        # --- track the unrack lockout: highest bar position before first descent
        if self.safety_y0 is None and self.phase == "top" and not self.reps:
            if self.pre_top_y is None or st.cy_smooth < self.pre_top_y:
                self.pre_top_y = st.cy_smooth
                self.pre_top_cx = st.cx

        # --- rep phase machine
        if self.phase in ("top", "ascent") and st.vy > V_EPS and depth_m > 0.05:
            # first descent of the set: freeze the safety line at the lockout
            if self.safety_y0 is None and self.pre_top_y is not None:
                self.safety_y0 = self.pre_top_y
                self.safety_x0 = self.pre_top_cx
                self._emit(t, "INFO", "safety_line",
                           f"marked at unrack lockout y={self.safety_y0:.0f}px (fixed)")
            self.phase = "descent"
            self.cur_rep = Rep(start_t=t, bottom_y=st.cy_smooth)
        elif self.phase == "descent":
            if self.cur_rep:
                self.cur_rep.bottom_y = max(self.cur_rep.bottom_y, st.cy_smooth)
                self.cur_rep.max_desc = max(self.cur_rep.max_desc, st.vy)
            if abs(st.vy) <= V_EPS:
                self.phase = "bottom"
                if self.cur_rep:
                    self.cur_rep.bottom_t = t
        elif self.phase == "bottom":
            if self.cur_rep:
                self.cur_rep.bottom_y = max(self.cur_rep.bottom_y, st.cy_smooth)
                self.cur_rep.hold_s = t - self.cur_rep.bottom_t
            if st.vy < -V_UP:
                self.phase = "ascent"
                self.asc_min_y = st.cy_smooth
                self.asc_min_cx = st.cx
            elif st.vy > V_EPS:
                self.phase = "descent"
        elif self.phase == "ascent":
            if self.asc_min_y is None or st.cy_smooth < self.asc_min_y:
                self.asc_min_y = st.cy_smooth
                self.asc_min_cx = st.cx
            if self._near_rack(st, ppm):
                self._complete_rep(t, pose)

        # --- T3 sudden drop
        thr = max(T3_DROP_MS, T3_BASELINE_X * self.baseline.max_desc)
        if st.vy_raw > thr and depth_m > 0.10:
            self.drop_frames += 1
        else:
            self.drop_frames = 0
        if self.drop_frames >= 2 and self._alarm_ok(t):
            self._emit(t, "ALARM", "sudden_drop",
                       f"v={st.vy_raw:.2f}m/s thr={thr:.2f} depth={depth_m:.2f}m")

        # --- T4 tilt (deviation from set baseline)
        dev = self._tilt_dev(st)
        if dev is not None:
            if abs(dev) >= T4_DEV_INSTANT and self._alarm_ok(t):
                ref = self.tilt_ref.get(st.tilt_src, 0.0)
                self._emit(t, "ALARM", "severe_tilt",
                           f"dev={dev:+.1f}deg (instant, ref {ref:+.1f})")
            elif abs(dev) >= T4_DEV_SUST:
                self.tilt_dips = 0
                if self.tilt_since is None:
                    self.tilt_since = t
                    self._emit(t, "WARN", "tilt", f"dev={dev:+.1f}deg")
                elif t - self.tilt_since >= T4_SUST_S and self._alarm_ok(t):
                    self._emit(t, "ALARM", "severe_tilt",
                               f"dev={dev:+.1f}deg sustained {t - self.tilt_since:.1f}s")
            elif abs(dev) < T4_DEV_SUST - 3.0:
                # hysteresis + dip counter: single-frame oscillation can't reset
                self.tilt_dips += 1
                if self.tilt_dips >= 3:
                    self.tilt_since = None
                    self.tilt_dips = 0

        # --- T5 plate slide v2
        self._t5(st, t, depth_m, ppm)

        # --- T1 bar sinking below bottom baseline (with hysteresis: once sinking,
        #     only a real rise clears the timer, not pixel jitter at the margin)
        if self.baseline.bottom_y is not None:
            sink_px = SINK_MARGIN_M * ppm
            below_enter = st.cy_smooth > self.baseline.bottom_y + sink_px
            below_stay = st.cy_smooth > self.baseline.bottom_y + 0.4 * sink_px
            if self.sink_since is None:
                if below_enter:
                    self.sink_since = t
                    self._emit(t, "WARN", "sinking",
                               f"{(st.cy_smooth - self.baseline.bottom_y) / ppm:.2f}m below baseline")
            else:
                rising = st.vy < -V_UP
                if rising or not below_stay:
                    self.sink_since = None
                elif t - self.sink_since >= T1_NO_RISE_S and self._alarm_ok(t):
                    self._emit(t, "ALARM", "failed_press_sink",
                               f"below baseline {t - self.sink_since:.1f}s, no ascent")

        # --- T2 stuck at bottom
        in_bottom = (self.baseline.bottom_y is not None and
                     abs(st.cy_smooth - self.baseline.bottom_y) <= SINK_MARGIN_M * ppm) or \
                    (self.baseline.bottom_y is None and depth_m > 0.25)
        if in_bottom and abs(st.vy) <= V_EPS:
            if self.still_since is None:
                self.still_since = t
            hold_thr = max(T2_HOLD_S, 2.0 * self.baseline.hold_s)
            held = t - self.still_since
            if hold_thr * 0.6 <= held < hold_thr and self.level == "OK":
                self._emit(t, "WARN", "long_hold", f"{held:.1f}s at bottom (thr {hold_thr:.1f}s)")
            if held >= hold_thr and self._alarm_ok(t):
                self._emit(t, "ALARM", "stuck_at_bottom",
                           f"motionless {held:.1f}s (thr {hold_thr:.1f}s)")
        else:
            self.still_since = None

        # --- T6 bar on body, not held (pose). POSITIVE evidence only: a wrist
        # confidently seen away from the bar. An unseen wrist is never "off",
        # so bad pose output alone cannot raise this alarm.
        if pose is not None and pose.lifter_found and in_bottom and abs(st.vy) <= 2 * V_EPS:
            # WARN needs the same positive-evidence standard as the alarm:
            # both wrists confidently off, or one clearly off while the other
            # is clearly on (a single unknown-partner wrist is just noise)
            strong_off = pose.hands_off >= 2 or (pose.hands_off == 1 and pose.hands_on == 1)
            if strong_off:
                if self.handsoff_since is None:
                    self.handsoff_since = t
                    self._emit(t, "WARN", "hands_off",
                               f"{pose.hands_off} hand(s) confidently off bar at chest depth")
                elif (T6_ALARM_ENABLED
                      and t - self.handsoff_since >= T6_HANDSOFF_S
                      and pose.hands_off >= 2 and pose.hands_on == 0
                      and self._alarm_ok(t)):
                    self._emit(t, "ALARM", "bar_on_body_not_held",
                               f"off={pose.hands_off} on={pose.hands_on} "
                               f"for {t - self.handsoff_since:.1f}s")
            else:
                self.handsoff_since = None
        else:
            self.handsoff_since = None

        # --- T7 elbow lockout failure (pose)
        self._t7(st, t, depth_m, pose)

        # --- recovery: upward crossing back into original rack zone
        if self.level in ("WARN", "ALARM") and self._near_rack(st, ppm) \
                and abs(st.vy) < V_EPS and self._rose_recently():
            self._emit(t, "RECOVERY", "bar_racked", "bar returned up to rack height")

    # ------------------------------------------------------------------ T5 v3
    def _t5(self, st: BarState, t, depth_m, ppm):
        if t - self.set_t0 < T5_SETTLE_S or st.bar_w <= 1:
            return
        # stable geometry: bar ends from the CALIBRATED bar length (running
        # median), never the instantaneous box - occlusion shrinks the box and
        # would teleport the computed ends. Frames where the detected box
        # deviates >15% from calibration are occlusion frames: skip T5 entirely.
        cal_len = ppm * 2.2   # BAR_LENGTH_M
        if cal_len <= 1 or abs(st.bar_w - cal_len) > 0.15 * cal_len:
            return
        bar_len = cal_len
        a = math.radians(st.tilt) if st.tilt is not None else 0.0
        ax, ay = math.cos(a), math.sin(a)
        ends = {0: (st.cx - ax * bar_len / 2, st.cy - ay * bar_len / 2),
                1: (st.cx + ax * bar_len / 2, st.cy + ay * bar_len / 2)}
        present = {0: False, 1: False}
        for (pcx, pcy, pw, ph, _) in st.plates:
            side = 0 if pcx < st.cx else 1
            present[side] = True
            ex, ey = ends[side]
            d = math.hypot(pcx - ex, pcy - ey) / bar_len
            base = self.end_base[side]
            if len(base) < int(self.fps * 1.5):
                base.append(d)
                continue
            ref = statistics.median(base)
            if d > ref + T5_GROW_FRAC:
                self.end_grow[side] += 1
                if self.end_grow[side] >= T5_SUST_FRAMES and depth_m > 0.05 \
                        and self._alarm_ok(t):
                    self._emit(t, "ALARM", "plate_slide",
                               f"side{side} end-dist {d:.2f} vs ref {ref:.2f} bar-lengths")
                    self.end_grow[side] = 0
            else:
                self.end_grow[side] = 0
                if len(base) < int(self.fps * 4):
                    base.append(d)

        # plate-count drop: a side that had a plate loses it while its end is in frame
        dtf = 1.0 / self.fps
        for side in (0, 1):
            ex, ey = ends[side]
            in_frame = 0.02 * st.frame_w < ex < 0.98 * st.frame_w and \
                       0.02 * st.frame_h < ey < 0.98 * st.frame_h
            if present[side]:
                self.plate_seen[side] += dtf
                self.plate_miss_since[side] = None
            elif self.plate_seen[side] > 2.0 and in_frame:
                if self.plate_miss_since[side] is None:
                    self.plate_miss_since[side] = t
                miss = t - self.plate_miss_since[side]
                if T5_MISS_WARN_S <= miss < T5_MISS_ALARM_S and self.level == "OK":
                    self._emit(t, "WARN", "plate_missing",
                               f"side{side} plate gone {miss:.1f}s (end in frame)")
                elif miss >= T5_MISS_ALARM_S and self._alarm_ok(t):
                    self._emit(t, "ALARM", "plate_detached",
                               f"side{side} plate gone {miss:.1f}s (end in frame)")
                    self.plate_miss_since[side] = None

    # ------------------------------------------------------------------ T7
    def _ref_slope(self) -> float:
        """Slope (dy/dx) of the camera's 'level bar' line, from the tilt ref."""
        ref = self.tilt_ref.get("obb", self.tilt_ref.get("plates"))
        return math.tan(math.radians(ref)) if ref is not None else 0.0

    def _safety_y_at(self, x: float, ppm) -> float | None:
        """Safety line height at bar position x. The line runs parallel to the
        level-bar orientation of this camera (slope from the tilt reference),
        anchored at the unrack lockout - marked once, fixed for the stream."""
        if self.safety_y0 is None:
            return None
        base_y = self.safety_y0 + T7_TOP_MARGIN_M * ppm
        anchor_x = self.safety_x0 if self.safety_x0 is not None else x
        return base_y + self._ref_slope() * (x - anchor_x)

    def safety_line(self, ppm, frame_w: int):
        """((x0,y0),(x1,y1)) for drawing, or None."""
        y0 = self._safety_y_at(0.0, ppm)
        y1 = self._safety_y_at(float(frame_w), ppm)
        if y0 is None or y1 is None:
            return None
        return (0, int(y0)), (frame_w, int(y1))

    def _t7(self, st: BarState, t, depth_m, pose):
        ppm = st.px_per_m or 1.0
        safety = self._safety_y_at(st.cx, ppm)

        # sustained stall during ascent while still below the safety line.
        # (mere deceleration is normal near lockout - it must be a real stall)
        if safety is not None and self.phase == "ascent" \
                and abs(st.vy) <= V_EPS and st.cy_smooth > safety:
            if self.stall_since is None:
                self.stall_since = t
            elif t - self.stall_since >= T7_STALL_S and not self.lockout_warned:
                self.lockout_warned = True
                short = (st.cy_smooth - safety) / ppm
                self._emit(t, "WARN", "incomplete_rep",
                           f"stalled {t - self.stall_since:.1f}s, "
                           f"{short:.2f}m below safety line")
        else:
            self.stall_since = None

        if self.lockout_warned:
            if st.vy > V_EPS:                      # coming back down after the stall
                self.lockout_pending_desc = True
            if self.lockout_pending_desc and abs(st.vy) <= V_EPS and depth_m > 0.10:
                if self.lockout_still_since is None:
                    self.lockout_still_since = t
                elif t - self.lockout_still_since >= T1_NO_RISE_S and self._alarm_ok(t):
                    self._emit(t, "ALARM", "failed_press_incomplete",
                               "descent after incomplete rep, no recovery")
                    self.lockout_warned = False
                    self.lockout_pending_desc = False
                    self.lockout_still_since = None
            elif self.lockout_pending_desc and st.vy < -V_UP:
                self.lockout_still_since = None    # rising again
            if st.vy < -V_UP and self._near_rack(st, st.px_per_m or 1):
                self.lockout_warned = False        # made it up after all
                self.lockout_pending_desc = False
                self.lockout_still_since = None

    # ------------------------------------------------------------------ helpers
    def _tilt_dev(self, st: BarState):
        """Tilt deviation vs the calibrated reference of the SAME source.
        No cross-source fallback: comparing an obb angle against a plate-line
        reference reintroduces the constant-offset flapping."""
        if st.tilt is None or not st.tilt_reliable:
            return None
        ref = self.tilt_ref.get(st.tilt_src)
        return None if ref is None else st.tilt - ref

    def _near_rack(self, st: BarState, ppm: float) -> bool:
        return self.rack_y is not None and st.cy_smooth < self.rack_y + 0.08 * ppm

    def _rose_recently(self) -> bool:
        """True if the bar was meaningfully below rack in the last 2s (real re-rack)."""
        if not self.cy_hist:
            return False
        t_now = self.cy_hist[-1][0]
        return any(d > 0.15 for (tt, d) in self.cy_hist if t_now - tt <= 2.0)

    def _complete_rep(self, t, pose=None):
        r = self.cur_rep
        self.phase = "top"
        self.cur_rep = None
        if r is None or self.rack_y is None:
            return
        r.top_y = self.asc_min_y
        r.top_cx = self.asc_min_cx
        self.asc_min_y = None
        self.asc_min_cx = None
        r.done = True
        self.reps.append(r)
        done = [x for x in self.reps if x.done][:3]
        if done:
            self.baseline.bottom_y = statistics.median([x.bottom_y for x in done])
            self.baseline.hold_s = max(0.3, statistics.median([x.hold_s for x in done]))
            self.baseline.max_desc = max(0.3, max(x.max_desc for x in done))
            tops = [x.top_y for x in done if x.top_y is not None]
            if len(tops) >= 3:
                self.baseline.top_y = statistics.median(tops)
                cxs = [x.top_cx for x in done if x.top_cx is not None]
                if cxs:
                    self.baseline.top_cx = statistics.median(cxs)
            self.baseline.n_reps = len(done)
        self._emit(t, "INFO", "rep_done",
                   f"rep#{len(self.reps)} bottom_y={r.bottom_y:.0f}px hold={r.hold_s:.1f}s"
                   + (f" top_y={r.top_y:.0f}px" if r.top_y is not None else ""))

    def _reset_set(self, why: str):
        if self.state != "IDLE":
            self._emit(self.events[-1]["t"] if self.events else 0.0, "INFO", "set_reset", why)
        events, last_alarm = self.events, self.last_alarm_t
        self.__init__(self.fps, self.log)
        self.events = events               # keep history + alarm cooldown across resets
        self.last_alarm_t = last_alarm
