"""Build a YOLO-OBB dataset (barbell only) from the saved CVAT XMLs.

Uses the rotation angle of each barbell box to write 4-corner oriented labels.
Images are taken from the existing detection dataset (same files, same split).
"""
import math
import os
import shutil
import xml.etree.ElementTree as ET

BASE = r"C:\Users\abdul\Claude\Projects\GuardianAI"
DET = os.path.join(BASE, "dataset")
OBB = os.path.join(BASE, "dataset_obb")
SCRATCH = (r"C:\Users\abdul\AppData\Local\Temp\claude\C--Users-abdul-Claude-Projects"
           r"\0d452096-3a58-4faf-93dc-1dca0b1bc1e2\scratchpad")

# (xml path, rename_prefix or None if names already carry vidXXX, frozen-box filter)
SOURCES = [
    # task_2471480 (dense vid007 + vid008) export no longer on disk - OBB trains without it
    (r"C:\Users\abdul\Downloads\task_2471448_annotations_2026_07_31_09_23_33_cvat for images 1.1\annotations.xml", None, None),
    (os.path.join(SCRATCH, "job_4302693", "annotations.xml"), None, None),
    (r"C:\Users\abdul\Downloads\fail-1-red\annotations.xml", "vid011", None),
    (r"C:\Users\abdul\Downloads\fail-4\annotations.xml", "vid012", None),
    (r"C:\Users\abdul\Downloads\fail-drop\annotations.xml", "vid013", None),
    (r"C:\Users\abdul\Downloads\new\v1\annotations.xml", "vid014", None),
    (r"C:\Users\abdul\Downloads\new\v2\annotations.xml", "vid015", None),
    (r"C:\Users\abdul\Downloads\new\v3\annotations.xml", "vid016", None),
    (r"C:\Users\abdul\Downloads\new\v4\annotations.xml", "vid017", ("519.02", "375.09", "57.20")),
]


def find_split(stem):
    for sub in ("train", "val"):
        for ext in (".jpg", ".png"):
            if os.path.exists(os.path.join(DET, "images", sub, stem + ext)):
                return sub, ext
    return None, None


def corners(xtl, ytl, xbr, ybr, rot_deg):
    cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
    w, h = xbr - xtl, ybr - ytl
    a = math.radians(rot_deg)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * math.cos(a) - dy * math.sin(a),
                    cy + dx * math.sin(a) + dy * math.cos(a)))
    return pts


def main():
    for sub in ("train", "val"):
        os.makedirs(os.path.join(OBB, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(OBB, "labels", sub), exist_ok=True)

    n = {"train": 0, "val": 0}
    skipped_no_img = skipped_frozen = 0
    for xml_path, prefix, frozen in SOURCES:
        root = ET.parse(xml_path).getroot()
        for img in root.findall(".//image"):
            W, H = float(img.get("width")), float(img.get("height"))
            name = img.get("name")
            if prefix:
                num = os.path.splitext(name)[0].split("_")[-1]
                stem = f"{prefix}_{num}"
            else:
                stem = os.path.splitext(name)[0]
            split, ext = find_split(stem)
            if split is None:
                skipped_no_img += 1
                continue
            lines = []
            for b in img.findall("box"):
                if b.get("label") != "barbell":
                    continue
                if frozen and (b.get("xtl"), b.get("ytl"), b.get("rotation")) == frozen:
                    skipped_frozen += 1
                    continue
                xtl, ytl, xbr, ybr = (float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                pts = corners(xtl, ytl, xbr, ybr, float(b.get("rotation") or 0))
                flat = []
                for x, y in pts:
                    flat += [min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0)]
                lines.append("0 " + " ".join(f"{v:.6f}" for v in flat))
            if not lines:
                continue
            src = os.path.join(DET, "images", split, stem + ext)
            dst = os.path.join(OBB, "images", split, stem + ext)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            with open(os.path.join(OBB, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n[split] += 1

    with open(os.path.join(OBB, "data.yaml"), "w") as f:
        f.write("path: " + OBB.replace("\\", "/") +
                "\ntrain: images/train\nval: images/val\nnames:\n  0: barbell\n")
    print(f"OBB dataset: train={n['train']} val={n['val']} "
          f"| no-image skips={skipped_no_img} frozen skips={skipped_frozen}")


if __name__ == "__main__":
    main()
