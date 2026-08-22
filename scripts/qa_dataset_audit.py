#!/usr/bin/env python3
"""
Comprehensive Quality Assurance (QA) & Missing Label Audit Engine
==================================================================
Identifies missing annotations, temporal sequence gaps, model-label disagreements,
and anomalous bounding boxes in surveillance datasets.
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def check_geometry_and_bounds(labels_dir):
    """Checks for corrupt, NaN, negative, or degenerate bounding boxes."""
    issues = []
    for txt_path in labels_dir.glob("*.txt"):
        if txt_path.stat().st_size == 0:
            continue
        try:
            with open(txt_path, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        issues.append((txt_path.name, f"Malformed line {line_idx+1}: '{line.strip()}'"))
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                        issues.append((txt_path.name, f"Center out of bounds (xc={xc:.3f}, yc={yc:.3f})"))
                    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        issues.append((txt_path.name, f"Invalid dimensions (w={w:.3f}, h={h:.3f})"))
                    if (w * h) < 0.0002:
                        issues.append((txt_path.name, f"Extremely tiny box (area={w*h:.6f})"))
        except Exception as e:
            issues.append((txt_path.name, f"Read error: {e}"))
    return issues


def find_temporal_gaps(frames_dir, labels_dir, max_gap_size=5):
    """
    Finds suspicious temporal holes in video sequences:
    e.g. Frame 10 has weapon, Frame 11-13 has NO weapon, Frame 14 has weapon.
    """
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    labels_map = {p.stem: p for p in labels_dir.glob("*.txt")}

    # Group by session
    session_frames = {}
    for img in frames:
        sess, num_str = img.stem.split("_")
        session_frames.setdefault(sess, []).append((int(num_str), img))

    gaps = []
    for sess, sorted_tuples in session_frames.items():
        sorted_tuples.sort(key=lambda x: x[0])
        box_statuses = []

        for idx, img in sorted_tuples:
            has_box = False
            if img.stem in labels_map:
                lbl = labels_map[img.stem]
                if lbl.stat().st_size > 0:
                    has_box = True
            box_statuses.append((idx, img.name, has_box))

        # Scan for gaps (True -> [False, False] -> True)
        n = len(box_statuses)
        for i in range(n):
            if not box_statuses[i][2]:
                continue  # Start from a labeled frame
            for j in range(i + 2, min(n, i + 2 + max_gap_size)):
                if box_statuses[j][2]:
                    # All frames between i and j are empty -> suspicious gap
                    middle_empty = [box_statuses[k] for k in range(i + 1, j) if not box_statuses[k][2]]
                    if len(middle_empty) == (j - i - 1) and len(middle_empty) > 0:
                        gaps.append({
                            "session": sess,
                            "before": box_statuses[i][1],
                            "after": box_statuses[j][1],
                            "gap_frames": [m[1] for m in middle_empty]
                        })
                    break

    return gaps


def mine_model_label_disagreements(frames_dir, labels_dir, model_path="runs/firearms/weights/best.pt", conf_thresh=0.50):
    """
    Runs model inference against ground-truth labels.
    Flags frames where the model is highly confident a weapon exists, but the label is empty/0-box.
    """
    if YOLO is None:
        print("[WARN] YOLO not available for disagreement mining.")
        return []

    m_path = Path(model_path)
    if not m_path.exists():
        fallback_models = ["yolo11n.pt", "yolov8n.pt", "yolov8m-world.pt"]
        for fb in fallback_models:
            if Path(fb).exists():
                m_path = Path(fb)
                break

    if not m_path.exists():
        print("[WARN] No model available to mine disagreements.")
        return []

    print(f"[INFO] Mining model-vs-label disagreements using '{m_path}' (conf={conf_thresh})...")
    model = YOLO(str(m_path))

    disagreements = []
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    labels_map = {p.stem: p for p in labels_dir.glob("*.txt")}

    for img_p in frames:
        lbl_p = labels_map.get(img_p.stem)
        has_gt_box = (lbl_p is not None and lbl_p.stat().st_size > 0)

        # We specifically look for False Negative candidates: Model sees weapon, but GT is empty
        if not has_gt_box:
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue

            results = model.predict(source=frame, conf=conf_thresh, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                top_box = results[0].boxes[0]
                conf = float(top_box.conf[0].item())
                cls_id = int(top_box.cls[0].item())
                cls_name = model.names.get(cls_id, f"cls_{cls_id}")

                disagreements.append({
                    "frame": img_p.name,
                    "model_conf": round(conf, 3),
                    "model_class": cls_name,
                    "issue": "Model detected high-confidence weapon, but ground truth label is EMPTY"
                })

    return disagreements


def render_html_qa_report(dataset_root, geometry_issues, temporal_gaps, disagreements, output_html_path):
    """Generates an interactive HTML dashboard for rapid human verification."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Quality Assurance & Missing Label Audit</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
        h1 {{ color: #38bdf8; }}
        h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-top: 32px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155; }}
        .badge-warn {{ background: #f59e0b; color: #000; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .badge-err {{ background: #ef4444; color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .badge-ok {{ background: #10b981; color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
        code {{ background: #334155; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>Dataset QA & Missing Label Audit Report</h1>
    <div class="card">
        <p><strong>Dataset Root:</strong> <code>{dataset_root}</code></p>
        <p><strong>Geometry Anomalies:</strong> {len(geometry_issues)} | <strong>Temporal Sequence Gaps:</strong> {len(temporal_gaps)} | <strong>Model-GT Disagreements:</strong> {len(disagreements)}</p>
    </div>

    <h2>1. Temporal Sequence Gaps (Potential Missed Weapons)</h2>
    <div class="card">
        <p>A temporal gap occurs when adjacent frames have labeled weapons, but intermediate frames were left blank.</p>
        <table>
            <tr><th>Session</th><th>Before Frame (Weapon)</th><th>Suspicious Empty Frames</th><th>After Frame (Weapon)</th></tr>
    """
    for g in temporal_gaps:
        html += f"<tr><td><code>{g['session']}</code></td><td>{g['before']}</td><td><span class='badge-warn'>{', '.join(g['gap_frames'])}</span></td><td>{g['after']}</td></tr>"

    if not temporal_gaps:
        html += "<tr><td colspan='4'><span class='badge-ok'>No suspicious temporal gaps detected.</span></td></tr>"

    html += """
        </table>
    </div>

    <h2>2. Model Disagreements (Potential False Negative Labels)</h2>
    <div class="card">
        <p>Frames where YOLO detects an object with high confidence, but the label file has 0 boxes.</p>
        <table>
            <tr><th>Frame Name</th><th>Model Confidence</th><th>Detected Class</th><th>Diagnostic</th></tr>
    """
    for d in disagreements:
        html += f"<tr><td><code>{d['frame']}</code></td><td><strong>{d['model_conf']}</strong></td><td>{d['model_class']}</td><td><span class='badge-err'>{d['issue']}</span></td></tr>"

    if not disagreements:
        html += "<tr><td colspan='4'><span class='badge-ok'>No high-confidence model omissions found.</span></td></tr>"

    html += f"""
        </table>
    </div>

    <h2>3. Geometry & Bounding Box Sanity</h2>
    <div class="card">
        <table>
            <tr><th>File</th><th>Issue</th></tr>
    """
    for fname, issue in geometry_issues:
        html += f"<tr><td><code>{fname}</code></td><td><span class='badge-err'>{issue}</span></td></tr>"

    if not geometry_issues:
        html += "<tr><td colspan='2'><span class='badge-ok'>All bounding box coordinates and formats are strictly valid.</span></td></tr>"

    html += """
        </table>
    </div>
</body>
</html>
    """

    with open(output_html_path, "w") as f:
        f.write(html)
    print(f"[INFO] Saved interactive QA report to: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Quality Assurance & Missing Label Audit Engine")
    parser.add_argument("--dataset-root", type=str, default="/Users/pratyushbharadwaj/Downloads/frames")
    parser.add_argument("--model", type=str, default="runs/firearms/weights/best.pt")
    parser.add_argument("--output-report", type=str, default="outputs/dataset_qa_report.html")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    frames_dir = root / "frames"
    labels_dir = root / "labels"

    print("=" * 70)
    print("RUNNING DATASET QUALITY ASSURANCE & MISSING LABEL AUDIT")
    print("=" * 70)

    # 1. Geometry check
    print("[1/3] Checking bounding box geometry and coordinate bounds...")
    geo_issues = check_geometry_and_bounds(labels_dir)
    print(f"      -> Found {len(geo_issues)} geometry issues.")

    # 2. Temporal sequence gaps
    print("[2/3] Analyzing video frame continuity for temporal gaps...")
    temporal_gaps = find_temporal_gaps(frames_dir, labels_dir)
    print(f"      -> Found {len(temporal_gaps)} suspicious temporal gaps.")

    # 3. Model disagreements
    print("[3/3] Mining model-vs-ground-truth disagreements...")
    disagreements = mine_model_label_disagreements(frames_dir, labels_dir, model_path=args.model)
    print(f"      -> Found {len(disagreements)} potential missing label candidates.")

    # Render Report
    out_rep = Path(args.output_report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    render_html_qa_report(root, geo_issues, temporal_gaps, disagreements, out_rep)

    print("\n" + "=" * 70)
    print("QA AUDIT COMPLETE")
    print(f"Report: file://{out_rep.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
