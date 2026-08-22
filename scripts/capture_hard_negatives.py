#!/usr/bin/env python3
"""
Live Hard Negative Mining Tool
==============================
Capture false positives from your webcam (e.g. phones, tripods, umbrellas)
and instantly export them as 0-byte hard negative labels for retraining.
"""

import argparse
import os
import time
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Please install ultralytics to run the hard negative miner.")


def main():
    parser = argparse.ArgumentParser(description="Live Hard Negative Mining for False Positive Suppression")
    parser.add_argument("--model", type=str, default="runs/detect/runs/firearms-8/weights/best.pt", help="Path to your custom model weights")
    parser.add_argument("--source", type=str, default="0", help="Camera source (0, 1, or RTSP url)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold to show detections")
    parser.add_argument("--output-dir", type=str, default="data/hard_negatives", help="Directory to save mined frames")
    parser.add_argument("--prefix", type=str, default="hn", help="Prefix for saved filenames")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] Model weights not found at {model_path}")

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video source {source}")

    print("=" * 60)
    print("HARD NEGATIVE MINING INTERFACE")
    print("=" * 60)
    print("Instructions:")
    print("1. Hold non-weapon objects (phones, cups, tools, tripods) in front of the camera.")
    print("2. If the model FALSELY detects it as a weapon, press 'SPACE' to capture.")
    print("3. Press 'q' or 'ESC' to quit.")
    print(f"Frames will be saved to: {out_dir}")
    print("=" * 60)

    # Count existing files to avoid overwriting
    existing_files = list(img_dir.glob(f"{args.prefix}_*.jpg"))
    counter = len(existing_files)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(source)
            continue

        # Run inference
        results = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        
        # Draw detections
        display_frame = frame.copy()
        num_detections = 0
        if results.boxes is not None:
            num_detections = len(results.boxes)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names.get(cls, str(cls))} {conf:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # UI Overlay
        cv2.putText(display_frame, f"Detections: {num_detections} | Captured: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to save as Hard Negative | Q to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Hard Negative Miner", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == 32:  # SPACE
            filename = f"{args.prefix}_{counter:05d}"
            img_path = img_dir / f"{filename}.jpg"
            lbl_path = lbl_dir / f"{filename}.txt"

            # Save frame
            cv2.imwrite(str(img_path), frame)
            # Save 0-byte label file (Hard Negative)
            with open(lbl_path, 'w') as f:
                pass
            
            print(f"[SAVED] Captured False Positive! Saved as {filename} (0-byte label)")
            
            # Flash effect
            flash = np.ones_like(display_frame) * 255
            cv2.imshow("Hard Negative Miner", cv2.addWeighted(display_frame, 0.5, flash, 0.5, 0))
            cv2.waitKey(50)
            
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Mining session complete. Total captured this session: {counter - len(existing_files)}")
    print(f"[INFO] To include these in your next training run, copy them to your dataset or merge using a script.")

if __name__ == "__main__":
    import numpy as np  # Needed for flash effect
    main()
