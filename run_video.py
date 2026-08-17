"""CLI: process a video -> annotated mp4 + events jsonl.

Usage: python run_video.py <input.mp4> [--weights runs/v1/weights/best.pt]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from guardian.pipeline import GuardianPipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    base = os.path.dirname(__file__)
    det_v2 = os.path.join(base, "runs_det", "v2", "weights", "best.pt")
    det_v1 = os.path.join(base, "runs", "v1", "weights", "best.pt")
    ap.add_argument("--weights", default=det_v2 if os.path.exists(det_v2) else det_v1)
    ap.add_argument("--obb", default=os.path.join(base, "runs_obb", "v1", "weights", "best.pt"))
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, f"{stem}_tracked.mp4")
    events = os.path.join(outdir, f"{stem}_events.jsonl")

    pipe = GuardianPipeline(args.weights, obb_weights=args.obb,
                            use_pose=not args.no_pose, conf=args.conf)
    pipe.process(args.video, out_path=out, events_path=events)
    print(f"output: {out}\nevents: {events}")
