# Firearm Vision — local webcam detection prototype

This is a local Cursor project for detecting visual object categories in video and webcam footage. It preserves every class in the Kaggle **Weapon Detection Dataset**:

```text
automatic_rifle, bazooka, grenade_launcher, handgun, knife,
shotgun, smg, sniper_rifle, sword
```

It does **not** identify people, infer intent, or take action. The program only writes a **human-review-required** alert after an object persists over multiple frames.

## Reality check on accuracy

No project can guarantee high accuracy before testing against a held-out set recorded with your actual camera. The provided public dataset is small and largely sourced from internet images, not your office webcam. This project therefore includes a calibration collection workflow and a real test report; use those results, not a training-score claim, to decide whether it is usable for a controlled demo.

## 1. Open in Cursor and install

Open this directory in Cursor. In the Cursor terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon: keep `--device mps` in commands below. Intel Mac: use `--device cpu`. NVIDIA: use `--device 0`.

## 2. Download and import Kaggle data

Download the Kaggle **Weapon Detection Dataset** ZIP, unzip it, and place its extracted folder anywhere under `data/incoming/`.

```bash
unzip ~/Downloads/weapon-detection-test.zip -d data/incoming/kaggle_weapon
python scripts/import_kaggle_yolo.py --input-dir data/incoming/kaggle_weapon
```

The importer copies data into the required YOLO structure without overwriting existing data:

```text
data/processed/
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

If the Kaggle folder already includes train/valid/test directories, the importer keeps them. Otherwise it creates a reproducible 70/15/15 split. Never put visually adjacent video frames in different splits.

## 3. Train

```bash
python train.py --data configs/firearms.yaml --epochs 100 --imgsz 960 --device mps
```

Training first downloads the general YOLO base model. Your trained model will be:

```text
runs/firearms/weights/best.pt
```

## 4. Evaluate before webcam use

```bash
python evaluate.py --model runs/firearms/weights/best.pt --device mps
```

Read `outputs/test_metrics.json` and the generated plots. Check per-class results, especially `automatic_rifle`, `shotgun`, `smg`, and `sniper_rifle`. A high overall score can hide poor long-firearm performance.

## 5. Improve it for your office webcam

Public data alone is not enough. In a private, consented, controlled setting, collect images of your **harmless model long firearm** and normal look-alike objects from the exact webcam:

```bash
python scripts/collect_calibration.py --source 0
```

Press `s` repeatedly to save images; press `q` to stop. Create a matching mixture of:

- model long firearm at different angles, distances, light levels, and partial occlusions;
- empty hands, phones, bottles, tools, umbrellas, cameras, and bags;
- no people and normal office backgrounds.

Label the firearm images as the closest visual category such as `automatic_rifle` or `smg`; label normal negative images with **no boxes**. Keep an entire set of sessions unseen until final evaluation. Import this labeled calibration data using the same YOLO folder convention, then retrain.

## 6. Run webcam inference

```bash
python infer.py --model runs/firearms/weights/best.pt --source 0 --record
```

Press `q` to stop. It saves timestamped potential-object alerts to `outputs/alerts.csv`, and `--record` writes `outputs/annotated.mp4`.

Defaults require a compatible detection across 3 of the last 5 frames before alerting. You may change those conservative temporal filters:

```bash
python infer.py --model runs/firearms/weights/best.pt --source 0 --confidence 0.65 --frames 7 --required 4
```

Tune the threshold only using a validation set; report results on the untouched test set. Do not use this prototype as an autonomous safety, enforcement, or targeting system.

## Optional face detection and privacy blur

Face detection only draws a box around faces; it does not recognize, identify, profile, or store identities. Add face boxes to the live view:

```bash
python infer.py --model runs/firearms/weights/best.pt --source 0 --faces
```

For recorded demos, prefer privacy blur:

```bash
python infer.py --model runs/firearms/weights/best.pt --source videos/test.mp4 --blur-faces --record
```

## 7. Run RTSP Web Frontend Dashboard

You can stream RTSP camera URLs, webcams, or video files through a web browser dashboard:

```bash
python app.py --model runs/firearms/weights/best.pt --source rtsp://admin:password@192.168.1.100:554/stream
```

Open `http://localhost:8000` in your web browser to:
- Connect dynamically to any RTSP camera URL, webcam `0`, or video file (`videos/test.mp4`).
- Monitor real-time MJPEG video stream with bounding box overlays and FPS metrics.
- Adjust confidence threshold and temporal verification parameters live.
- Toggle privacy face blurring and audio alerts.
- View real-time alert logs and historical detections (`HUMAN REVIEW REQUIRED`).

