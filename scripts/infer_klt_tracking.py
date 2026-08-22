import argparse
import time
from pathlib import Path
import cv2

from src.lightweight_tracker import LightweightTrackerEngine


def parse_source(value):
    return int(value) if value.isdigit() else value


def load_haar_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV Haar cascade face detector failed to load.")
    return detector


def detect_haar_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    detections = []
    for (x, y, w, h) in rects:
        detections.append({
            "box": [x, y, x + w, y + h],
            "label": "Face",
            "confidence": 0.95,
            "is_weapon": False
        })
    return detections


def main():
    parser = argparse.ArgumentParser(description="KLT Pyramid Optical Flow Face Tracking Demo")
    parser.add_argument("--source", default="0", help="Webcam index (0) or path to video file")
    parser.add_argument("--skip-frames", type=int, default=5, help="Number of frames between heavy face detections")
    parser.add_argument("--show-keypoints", action="store_true", help="Draw Shi-Tomasi feature points on faces")
    parser.add_argument("--record", action="store_true", help="Record output video to outputs/klt_tracking.mp4")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.record:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "klt_tracking.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[INFO] Recording to {out_path}")

    detector = load_haar_detector()
    tracker_engine = LightweightTrackerEngine(dist_thresh=180.0, max_missed=15)

    frame_count = 0
    start_time = time.time()
    last_frame_type = "INIT"
    process_ms = 0.0

    print(f"[INFO] Started KLT Optical Flow Tracking (Detection every {args.skip-frames} frames). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        is_detection_frame = (frame_count % args.skip_frames == 0) or len(tracker_engine.tracks) == 0

        if is_detection_frame:
            last_frame_type = "DETECTION (Heavy)"
            detections = detect_haar_faces(frame, detector)
            tracker_engine.update_heavy_detections(frame, detections)
            tracked_objects = tracker_engine.step_visual_tracking(frame)
        else:
            last_frame_type = "KLT TRACKING (Fast <0.5ms)"
            tracked_objects = tracker_engine.step_visual_tracking(frame)

        t1 = time.time()
        process_ms = (t1 - t0) * 1000.0

        # Draw bounding boxes & annotations
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["box"]
            track_id = obj["track_id"]
            label = obj["label"]
            conf = obj["confidence"]

            color = (0, 255, 120) if is_detection_frame else (255, 190, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"#{track_id} {label} ({conf:.2f})",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Draw Shi-Tomasi feature keypoints
            for kpt in obj.get("keypoints", []):
                cv2.circle(frame, (kpt[0], kpt[1]), 2, (0, 255, 0), -1)

        # Draw HUD stats
        frame_count += 1
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

        hud_color = (0, 255, 0) if "KLT" in last_frame_type else (0, 200, 255)
        cv2.putText(frame, f"Mode: {last_frame_type} ({process_ms:.2f} ms)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.1f} | Active Tracks: {len(tracked_objects)}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Firearm Vision — KLT Optical Flow Face Tracking", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] KLT tracking demo stopped.")


if __name__ == "__main__":
    main()
