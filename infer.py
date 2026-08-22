import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.temporal import Candidate, TemporalConfirm


def parse_source(value):
    return int(value) if value.isdigit() else value


def load_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("OpenCV's built-in face detector could not be loaded")
    return detector


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))


def draw_or_blur_faces(frame, faces, blur):
    for x, y, width, height in faces:
        if blur:
            region = frame[y:y + height, x:x + width]
            if region.size:
                frame[y:y + height, x:x + width] = cv2.GaussianBlur(region, (31, 31), 0)
        else:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 190, 0), 2)
            cv2.putText(frame, "face", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 190, 0), 2)


def main():
    p = argparse.ArgumentParser(description="Local visual-object alert tool; all detections require human review.")
    p.add_argument("--model", required=True, help="Path to trained best.pt")
    p.add_argument("--source", default="0", help="0 for webcam or a video path")
    p.add_argument("--confidence", type=float, default=.55)
    p.add_argument("--frames", type=int, default=5); p.add_argument("--required", type=int, default=3)
    p.add_argument("--faces", action="store_true", help="Draw face boxes; does not identify or recognize anyone")
    p.add_argument("--blur-faces", action="store_true", help="Blur detected faces in the live view and any recording")
    p.add_argument("--record", action="store_true", help="Save annotated webcam/video output")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()
    if not Path(args.model).exists(): raise SystemExit(f"Missing model: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model); cap = cv2.VideoCapture(parse_source(args.source))
    face_detector = load_face_detector() if args.faces or args.blur_faces else None
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) if args.record else None
    gate = TemporalConfirm(args.frames, args.required); frame_number = 0; alert_open = False
    with (args.output_dir / "alerts.csv").open("w", newline="") as output:
        log = csv.DictWriter(output, fieldnames=["time_seconds", "frame", "label", "confidence", "status"]); log.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok: break
            result = model(frame, conf=args.confidence, verbose=False)[0]; candidates = []
            if result.boxes is not None:
                for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    label = result.names[int(cls)]; candidates.append(Candidate(label, score, tuple(box)))
            confirmed = False
            for candidate in candidates:
                x1, y1, x2, y2 = map(int, candidate.box)
                confirmed = gate.confirmed(candidate) or confirmed
                color = (0, 0, 255) if confirmed else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{candidate.label} {candidate.confidence:.2f}", (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            if confirmed and not alert_open:
                top = max(candidates, key=lambda c: c.confidence)
                log.writerow({"time_seconds": round(frame_number / fps, 2), "frame": frame_number, "label": top.label, "confidence": round(top.confidence, 3), "status": "human_review_required"})
            alert_open = confirmed
            if confirmed:
                cv2.rectangle(frame, (0, 0), (width, 38), (0, 0, 180), -1); cv2.putText(frame, "POSSIBLE OBJECT DETECTED - HUMAN REVIEW REQUIRED", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            if face_detector is not None:
                draw_or_blur_faces(frame, detect_faces(frame, face_detector), args.blur_faces)
            cv2.imshow("Firearm Vision (press q to quit)", frame)
            if writer: writer.write(frame)
            frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()
    if writer: writer.release()
    print(f"Saved {args.output_dir / 'alerts.csv'}")


if __name__ == "__main__": main()
