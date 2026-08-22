import argparse
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description="Capture consented calibration images from the intended webcam. Press s to save; q to quit.")
    p.add_argument("--source", default="0"); p.add_argument("--output-dir", type=Path, default=Path("data/office_calibration"))
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source); n = 0
    if not cap.isOpened(): raise SystemExit("Cannot open camera")
    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.putText(frame, "s: save  q: quit", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2); cv2.imshow("Calibration capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(str(args.output_dir / f"office_{n:04d}.jpg"), frame); n += 1
        elif key == ord("q"): break
    cap.release(); cv2.destroyAllWindows(); print(f"Saved {n} images")


if __name__ == "__main__": main()
