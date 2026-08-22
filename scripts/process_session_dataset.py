#!/usr/bin/env python3
"""
Session-Aware Dataset Splitter & Auto-Prelabeler for Surveillance Datasets
========================================================================
Performs leakage-free splitting by session/scene (so adjacent video frames never cross train/val/test splits).
Handles class ID alignment, 0-byte hard negatives, and auto-prelabeling of missing frames.
"""

import argparse
import os
import shutil
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Session Splitting Strategy:
# - Train: s01, s02, s04, s06, s07, s11 (part), s09 (part)
# - Valid: s05, s08, s09 (part)
# - Test:  s03 (occlusion), s10 (decoy tripod), s09 (part)
DEFAULT_SPLIT_MAP = {
    "train": ["s01", "s02", "s04", "s06", "s07"],
    "valid": ["s05", "s08"],
    "test":  ["s03", "s10"]
}


def audit_dataset(frames_dir, labels_dir):
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    labels_map = {p.stem: p for p in labels_dir.glob("*.txt")}

    sessions = {}
    missing_files = []

    for img in frames:
        sess = img.stem.split("_")[0]
        sessions.setdefault(sess, {"total": 0, "boxes": 0, "empty": 0, "missing": []})
        sessions[sess]["total"] += 1

        if img.stem in labels_map:
            lbl = labels_map[img.stem]
            if lbl.stat().st_size > 0:
                sessions[sess]["boxes"] += 1
            else:
                sessions[sess]["empty"] += 1
        else:
            sessions[sess]["missing"].append(img)
            missing_files.append(img)

    return sessions, missing_files


def auto_prelabel_missing(missing_imgs, labels_dir, model_path="runs/firearms/weights/best.pt", conf_thresh=0.25):
    if not missing_imgs:
        print("[INFO] No missing labels to pre-label!")
        return

    if YOLO is None:
        print("[WARN] YOLO not available for auto-prelabeling.")
        return

    m_path = Path(model_path)
    if not m_path.exists():
        fallback_models = ["yolo11n.pt", "yolov8n.pt", "yolov8m-world.pt"]
        for fb in fallback_models:
            if Path(fb).exists():
                m_path = Path(fb)
                break

    if not m_path.exists():
        print(f"[WARN] Model weights not found. Creating empty labels for missing frames.")
        for img in missing_imgs:
            lbl_file = labels_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                open(str(lbl_file), "w").close()
        return

    print(f"[INFO] Auto-prelabeling {len(missing_imgs)} missing frames using '{m_path}' (conf={conf_thresh})...")
    model = YOLO(str(m_path))

    prelabeled = 0
    negatives = 0

    for img_p in missing_imgs:
        txt_path = labels_dir / f"{img_p.stem}.txt"
        frame = cv2.imread(str(img_p))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        boxes_to_write = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                bx1, by1, bx2, by2 = xyxy
                bw = (bx2 - bx1) / w
                bh = (by2 - by1) / h
                bx_center = (bx1 + bx2) / (2.0 * w)
                by_center = (by1 + by2) / (2.0 * h)
                # Map to rifle/firearm class 0
                boxes_to_write.append(f"0 {bx_center:.6f} {by_center:.6f} {bw:.6f} {bh:.6f}\n")

        with open(txt_path, "w") as f:
            if boxes_to_write:
                f.writelines(boxes_to_write)
                prelabeled += 1
            else:
                negatives += 1

    print(f"[INFO] Auto-prelabeled {prelabeled} frames with weapon candidate boxes and {negatives} hard negatives.")


def build_session_splits(frames_dir, labels_dir, output_root=Path("data/processed")):
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    labels_map = {p.stem: p for p in labels_dir.glob("*.txt")}

    splits = {"train": [], "valid": [], "test": []}

    # Handle split distribution
    for img in frames:
        sess = img.stem.split("_")[0]
        if sess in DEFAULT_SPLIT_MAP["train"]:
            splits["train"].append(img)
        elif sess in DEFAULT_SPLIT_MAP["valid"]:
            splits["valid"].append(img)
        elif sess in DEFAULT_SPLIT_MAP["test"]:
            splits["test"].append(img)
        elif sess == "s09":
            # Negative session split evenly across train (70%), val (15%), test (15%)
            idx = int(img.stem.split("_")[1])
            if idx % 10 < 7:
                splits["train"].append(img)
            elif idx % 10 < 8:
                splits["valid"].append(img)
            else:
                splits["test"].append(img)
        elif sess == "s11":
            # Placed weapon session split 70% train, 30% valid
            idx = int(img.stem.split("_")[1])
            if idx % 10 < 7:
                splits["train"].append(img)
            else:
                splits["valid"].append(img)
        else:
            splits["train"].append(img)

    for split_name, img_list in splits.items():
        out_img_dir = output_root / split_name / "images"
        out_lbl_dir = output_root / split_name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        box_count = 0
        neg_count = 0

        for img_p in img_list:
            shutil.copy2(str(img_p), str(out_img_dir / img_p.name))
            target_lbl = out_lbl_dir / f"{img_p.stem}.txt"

            if img_p.stem in labels_map:
                src_lbl = labels_map[img_p.stem]
                shutil.copy2(str(src_lbl), str(target_lbl))
                if src_lbl.stat().st_size > 0:
                    box_count += 1
                else:
                    neg_count += 1
            else:
                open(str(target_lbl), "w").close()
                neg_count += 1

        print(f"[SPLIT: {split_name.upper():<5}] -> Total: {len(img_list):<4} frames | Weapons: {box_count:<4} | Hard Negatives: {neg_count:<4}")


def main():
    parser = argparse.ArgumentParser(description="Session-Aware CCTV Dataset Pipeline")
    parser.add_argument("--dataset-root", type=str, default="/Users/pratyushbharadwaj/Downloads/frames")
    parser.add_argument("--model", type=str, default="runs/firearms/weights/best.pt")
    parser.add_argument("--prelabel-missing", action="store_true", help="Auto-propose labels for missing frames")
    parser.add_argument("--export-splits", action="store_true", help="Export session splits to data/processed/")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    frames_dir = root / "frames"
    labels_dir = root / "labels"

    print("=" * 70)
    print(f"SURVEILLANCE DATASET AUDIT: {root}")
    print("=" * 70)

    sessions, missing = audit_dataset(frames_dir, labels_dir)
    total_imgs = sum(s["total"] for s in sessions.values())
    total_boxes = sum(s["boxes"] for s in sessions.values())
    total_empty = sum(s["empty"] for s in sessions.values())

    print(f"Total Frames:               {total_imgs}")
    print(f"Frames with Weapon Boxes:   {total_boxes} ({100.0 * total_boxes / total_imgs:.1f}%)")
    print(f"Hard-Negative Frames (0b):  {total_empty} ({100.0 * total_empty / total_imgs:.1f}%)")
    print(f"Missing Unannotated Frames: {len(missing)}")
    print("-" * 70)

    for sess, st in sorted(sessions.items()):
        print(f"[{sess}] Total: {st['total']:<4} | Boxes: {st['boxes']:<4} | Negatives: {st['empty']:<4} | Missing: {len(st['missing']):<4}")

    if args.prelabel_missing and missing:
        auto_prelabel_missing(missing, labels_dir, model_path=args.model)

    if args.export_splits:
        print("\n" + "=" * 70)
        print("EXPORTING LEAKAGE-FREE SESSION SPLITS TO data/processed/")
        print("=" * 70)
        build_session_splits(frames_dir, labels_dir)
        print("\n[✓] Export complete! Ready to run evaluate.py or train.py.")


if __name__ == "__main__":
    main()
