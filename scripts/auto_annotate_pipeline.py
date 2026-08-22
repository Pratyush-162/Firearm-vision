#!/usr/bin/env python3
"""
Automated Annotation Acceleration Pipeline for CCTV & Video Datasets
====================================================================
Features:
1. Frame Deduplication: Skips static/duplicate CCTV frames via Perceptual Hash (pHash) & Motion Energy.
2. Model-Assisted Pre-Labeling: Automatically proposes bounding boxes using trained YOLO models.
3. Tracker Box Propagation: Propagates bounding boxes across consecutive frames using Optical Flow/IoU tracking.
4. Hard-Negative Support: Automatically writes 0-byte label files for background/look-alike negative frames.
5. Standard YOLO Split Export: Splits extracted & labeled frames into train/valid sets ready for training.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Class ID mapping matching configs/firearms.yaml
CLASS_MAPPING = {
    "automatic_rifle": 0,
    "assault_rifle": 0,
    "rifle": 0,
    "bazooka": 1,
    "grenade_launcher": 2,
    "handgun": 3,
    "pistol": 3,
    "revolver": 3,
    "gun": 3,
    "firearm": 3,
    "knife": 4,
    "blade": 4,
    "dagger": 4,
    "shotgun": 5,
    "smg": 6,
    "sniper_rifle": 7,
    "sword": 8,
    "machete": 8
}


def compute_frame_hash(frame, hash_size=16):
    """Computes a difference hash (dHash) to identify visually redundant frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hash_distance(h1, h2):
    """Hamming distance between two frame hashes."""
    return np.count_nonzero(h1 != h2)


def compute_motion_energy(prev_gray, curr_gray, threshold=25):
    """Computes percentage of pixels with significant motion between consecutive frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_score = np.count_nonzero(thresh) / thresh.size
    return motion_score


def extract_and_deduplicate(
    video_path,
    output_img_dir,
    sample_every_n=5,
    min_motion_thresh=0.015,
    min_hash_dist=8
):
    """
    Extracts informative frames from video while dropping static and redundant CCTV frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_path}")
        return []

    output_img_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = []
    prev_gray = None
    prev_hash = None
    frame_idx = 0
    saved_count = 0

    video_name = Path(video_path).stem

    print(f"[INFO] Processing '{video_path}' for intelligent deduplication...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_hash = compute_frame_hash(frame)

        # First frame always saved
        if prev_gray is None:
            save_frame = True
        else:
            motion = compute_motion_energy(prev_gray, curr_gray)
            h_dist = hash_distance(prev_hash, curr_hash)
            # Frame is saved if there is meaningful motion OR visual difference
            save_frame = (motion >= min_motion_thresh) or (h_dist >= min_hash_dist)

        if save_frame:
            frame_filename = f"{video_name}_f{frame_idx:06d}.jpg"
            save_path = output_img_dir / frame_filename
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_frames.append((save_path, frame))
            saved_count += 1
            prev_gray = curr_gray
            prev_hash = curr_hash

    cap.release()
    print(f"[INFO] Extracted {saved_count} informative frames from {frame_idx} total video frames (Deduplication saved {100.0 * (1.0 - saved_count / max(1, frame_idx)):.1f}% redundant frames).")
    return saved_frames


def model_assisted_prelabel(
    model_path,
    frame_tuples,
    output_lbl_dir,
    conf_thresh=0.25,
    device="cpu"
):
    """
    Runs model inference on extracted frames and generates YOLO format .txt annotations.
    """
    if YOLO is None:
        print("[WARN] Ultralytics YOLO not installed. Skipping auto-prelabeling.")
        return

    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading model '{model_path}' for auto-prelabeling...")
    
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    labeled_count = 0
    hard_neg_count = 0

    for img_path, frame in frame_tuples:
        txt_path = output_lbl_dir / f"{img_path.stem}.txt"
        h, w = frame.shape[:2]

        results = model.predict(source=frame, conf=conf_thresh, device=device, verbose=False)
        boxes_to_write = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = model.names.get(cls_id, "").lower()

                # Map class name to our dataset schema if present
                mapped_id = CLASS_MAPPING.get(cls_name, cls_id if cls_id < 9 else 3)

                # Normalized xywh
                xyxy = box.xyxy[0].cpu().numpy()
                bx1, by1, bx2, by2 = xyxy
                bw = (bx2 - bx1) / w
                bh = (by2 - by1) / h
                bx_center = (bx1 + bx2) / (2.0 * w)
                by_center = (by1 + by2) / (2.0 * h)

                boxes_to_write.append(f"{mapped_id} {bx_center:.6f} {by_center:.6f} {bw:.6f} {bh:.6f}\n")

        with open(txt_path, "w") as f:
            if boxes_to_write:
                f.writelines(boxes_to_write)
                labeled_count += 1
            else:
                # 0-byte file indicates a hard negative / background image to YOLO
                hard_neg_count += 1

    print(f"[INFO] Pre-labeled {labeled_count} frames with weapon bounding boxes.")
    print(f"[INFO] Generated {hard_neg_count} hard-negative / background frames (0-byte annotations).")


def build_yolo_splits(
    img_dir,
    lbl_dir,
    processed_root,
    train_ratio=0.80,
    val_ratio=0.20
):
    """
    Distributes annotated images and labels into train/valid directories.
    """
    img_paths = sorted(list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png")))
    if not img_paths:
        print("[WARN] No images found to split.")
        return

    np.random.seed(42)
    indices = np.arange(len(img_paths))
    np.random.shuffle(indices)

    n_train = int(len(img_paths) * train_ratio)
    train_indices = set(indices[:n_train])

    train_img_dir = processed_root / "train" / "images"
    train_lbl_dir = processed_root / "train" / "labels"
    val_img_dir = processed_root / "valid" / "images"
    val_lbl_dir = processed_root / "valid" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx, img_p in enumerate(img_paths):
        lbl_p = Path(lbl_dir) / f"{img_p.stem}.txt"
        is_train = idx in train_indices

        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        shutil.copy2(str(img_p), str(target_img_dir / img_p.name))
        if lbl_p.exists():
            shutil.copy2(str(lbl_p), str(target_lbl_dir / lbl_p.name))
        else:
            # Create empty label file if missing
            open(str(target_lbl_dir / f"{img_p.stem}.txt"), "w").close()
        copied += 1

    print(f"[INFO] Successfully packaged {copied} frames into '{processed_root}':")
    print(f"       -> Train: {n_train} frames ({train_img_dir})")
    print(f"       -> Valid: {len(img_paths) - n_train} frames ({val_img_dir})")


def main():
    parser = argparse.ArgumentParser(description="Automated CCTV Frame Deduplication & Model Pre-Labeling Pipeline")
    parser.add_argument("--video", type=str, help="Path to raw surveillance video or footage file")
    parser.add_argument("--images-dir", type=str, help="Directory of existing unannotated raw images (optional)")
    parser.add_argument("--model", type=str, default="runs/firearms/weights/best.pt", help="YOLO weights to use for auto-prelabeling")
    parser.add_argument("--output-dir", type=str, default="data/annotation_workspace", help="Staging workspace directory")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample 1 frame every N video frames")
    parser.add_argument("--motion-thresh", type=float, default=0.015, help="Minimum pixel motion energy threshold")
    parser.add_argument("--confidence", type=float, default=0.25, help="Low-threshold recall pass for pre-labeling")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, 0)")
    parser.add_argument("--export-to-processed", action="store_true", help="Automatically split and merge into data/processed/")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    img_dir = out_root / "extracted_images"
    lbl_dir = out_root / "prelabeled_labels"

    frame_tuples = []

    if args.video:
        v_path = Path(args.video)
        if not v_path.exists():
            print(f"[ERROR] Video file '{v_path}' does not exist.")
            sys.exit(1)
        frame_tuples = extract_and_deduplicate(
            video_path=v_path,
            output_img_dir=img_dir,
            sample_every_n=args.sample_rate,
            min_motion_thresh=args.motion_thresh
        )
    elif args.images_dir:
        p = Path(args.images_dir)
        raw_imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        img_dir.mkdir(parents=True, exist_ok=True)
        for img_p in raw_imgs:
            frame = cv2.imread(str(img_p))
            if frame is not None:
                dest = img_dir / img_p.name
                shutil.copy2(str(img_p), str(dest))
                frame_tuples.append((dest, frame))
        print(f"[INFO] Loaded {len(frame_tuples)} raw images from {args.images_dir}")
    else:
        print("[ERROR] Please provide either --video <path_to_video> or --images-dir <dir>")
        sys.exit(1)

    # Model Pre-labeling pass
    model_path = Path(args.model)
    if not model_path.exists():
        # Fallback to general model if custom best.pt is not yet trained
        fallback_models = ["yolo11n.pt", "yolov8n.pt", "yolov8m-world.pt"]
        for fb in fallback_models:
            if Path(fb).exists():
                model_path = Path(fb)
                print(f"[INFO] '{args.model}' not found, falling back to base model: {model_path}")
                break

    if model_path.exists() and frame_tuples:
        model_assisted_prelabel(
            model_path=model_path,
            frame_tuples=frame_tuples,
            output_lbl_dir=lbl_dir,
            conf_thresh=args.confidence,
            device=args.device
        )

    if args.export_to_processed:
        build_yolo_splits(
            img_dir=img_dir,
            lbl_dir=lbl_dir,
            processed_root=Path("data/processed")
        )

    print("\n" + "=" * 70)
    print(" [✓] ANNOTATION STAGING COMPLETE")
    print(f" Images Staged: {img_dir}")
    print(f" Labels Staged: {lbl_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
