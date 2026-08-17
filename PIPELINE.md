# GuardianAI — MVP Pipeline Design

Scope: detect a failed bench press from a fixed camera, bar-only signals (no pose model in MVP).
Reference failure footage: fail-1-red (failed press → bar on chest), fail-4 (sustained one-sided tilt),
fail-drop (sudden one-sided drop, ~44° tilt).

## Pipeline

```
video ──► DETECT   YOLO: barbell + bar_plates, per frame
      ──► TRACK    pick active bar, Kalman w/ innovation gate, camera-cut guard
      ──► SIGNALS  per frame: center (x,y) · vertical velocity · tilt · confidence
      ──► STATE    idle → set active → per-rep (descent/bottom/ascent) → racked
      ──► BASELINE median bottom-y, bottom-hold, descent speed of first 3 reps
      ──► TRIGGERS rules T1–T4, WARN → ALARM escalation
      ──► ALARM    JSON/console log + red overlay banner + saved event clip
```

## Key design decisions (tweaks vs. original GuardianAI doc)

1. **No pose model in MVP** — distress cues deferred to v2; compensated with longer
   thresholds + WARN→ALARM escalation.
2. **Tilt = angle of line between the two plate-box centers.** Single plate visible →
   no tilt update that frame (hold last value briefly, never fabricate).
3. **Auto-calibration:** px→m scale from bar length (≈2.2 m); rack zone = stable bar
   height at set start; bottom zone from lifter's own reps. Manual zone marking remains
   an install-time upgrade.
4. **Camera-cut guard** for edited eval footage: global frame-change ⇒ tracker reset;
   no trigger may fire across a reset.
5. **Kalman innovation gate** (doc's "mode switch"): large innovation ⇒ trust the raw
   measurement so free-fall is never smoothed away.

## Triggers (bar-only)

| # | Name | Condition | Latency target |
|---|------|-----------|----------------|
| T1 | Failed press / bar sinking | bar drops below lifter's bottom baseline by margin (sinking into chest; preset depth if first rep fails) AND no upward motion within 1.5 s — raw (unsmoothed) velocity | ~1.5–2 s |
| T2 | Stuck at bottom | in normal bottom zone (not below), motionless for max(5 s, 2× median bottom hold) | threshold |
| T3 | Sudden drop | descent >3× baseline max (or >1.5 m/s), ends at/below chest; raw measurement, not across a cut | 1–2 s |
| T4 | Severe tilt | \|tilt\| >12° sustained 1 s, or >25° instant, while unracked | 1–2 s |
| T5 | Plates sliding off | plate center projected onto bar axis drifts outward >~10% of bar length within ~1 s while unracked, or plate box separates from bar box (MVP fires even on deliberate self-rescue slide; v2 pose downgrades it) | 1–2 s |
| R | Recovery | bar re-racked or steady ascent ⇒ cancel WARN, log, keep watching 60 s | — |

Mapping to reference clips: Mizkif = T1/T2 · Knudsen = T4 · Tuomas = T3+T4+T5.

## MVP acceptance criteria

- All 3 fail clips raise the correct alarm within the doc's 10–12 s human-alert budget.
- Zero alarms across full normal-set videos (vid007/vid009 sources), incl. pause reps.

## Build order (iteration 1 — DONE)

1. Train detector (YOLOv8s, 701 images) — val: P 0.91 / R 0.78 / mAP50 0.89.
2. Tracker + trigger engine + overlay (guardian/ package).
3. CLI (run_video.py) + web UI (app.py, port 5001).
4. Result: 3/3 fail clips alarmed (plate_slide 16.8s / severe_tilt 15.9s / severe_tilt 25.4s late).

## Iteration 2 — planned

**Pose model enters (pretrained YOLOv8-pose or RTMPose, no annotation needed).**
Lifter = person whose torso lies under the bar span. Pose every 2nd-3rd frame for fps.

| # | Name | Condition | Level |
|---|------|-----------|-------|
| T6 | Bar on body, not held | unracked AND chest-depth AND hands_on < 2 (wrist-to-bar-axis > grip radius ~7% bar len, debounced 0.5 s) for 1-1.5 s | ALARM |
| T7 | Press stall (grind) | ascent → bar stationary mid-height AND worse-arm elbow angle < ~150° (no lockout) for > 2 s | WARN |
| T7b | Failed press after stall | from T7 WARN: bar descends, elbows bent, no re-ascent within 1.5 s of chest zone | ALARM |

T7 recovery (bar completes ascent to lockout) → INFO, but tighten T1/T2 thresholds
for the rest of the set (lifter is at their limit).

**Plate-slide v2:** plate-to-bar-END distance growth (1 s window) · detached plate
(center exits expanded bar box or velocity diverges) = instant ALARM · plate-count
drop while that end is in frame = WARN→ALARM.

**Tilt robustness:**
- Train YOLOv8s-OBB from the saved CVAT XMLs (rotated barbell boxes already exist —
  zero new annotation). OBB angle becomes primary tilt source; plate-pair line is
  cross-check; plates focus on slide + scale.
- Fallback chain w/ tilt Kalman: OBB angle > plate-pair > single-plate + AABB aspect
  (|tilt| ≈ asin((box_h − bar_thickness)/box_len), sign from plate y-offset).
- Out-of-plane rotation: apparent bar length << calibrated length while "level"
  ⇒ tilt_unreliable flag (never fabricate an angle; T4 fires on measured values only).

**Cleanups:** "bar lost mid-rep" WARN (Tuomas gap) · recovery requires original rack
zone of the same set (kills bogus RECOVERY after resets) · annotate motion-blurred
failure frames (Tuomas drop) into the dataset.

**Order:** OBB retrain → pose + T6/T7 → plate-slide v2 → tilt fusion + cleanups →
re-run 3 fail clips + false-alarm sweep over normal videos.
