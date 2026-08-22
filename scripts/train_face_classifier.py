import os
import shutil
import random
import json
from pathlib import Path
from ultralytics import YOLO


STATUS_FILE = Path("data/training_status.json")


def update_status(status="running", progress=0, epoch=0, total_epochs=60, loss=0.0, message=""):
    """Writes real-time training progress and logs to data/training_status.json for web UI terminal."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                current = json.load(f)
        except Exception:
            pass
    logs = current.get("logs", [])
    if message:
        logs.append(message)
        if len(logs) > 60:
            logs = logs[-60:]
    data = {
        "status": status,
        "progress": progress,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": round(loss, 4),
        "message": message,
        "logs": logs
    }
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def prepare_classification_dataset(src_dir="data/enrolled_faces", out_dir="data/processed_faces", split_ratio=0.8):
    src_path = Path(src_dir)
    out_path = Path(out_dir)

    if out_path.exists():
        shutil.rmtree(out_path)

    train_dir = out_path / "train"
    val_dir = out_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    person_dirs = [d for d in src_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    total_images = 0
    for p_dir in person_dirs:
        raw_name = p_dir.name.replace("_", " ").title()
        clean_class = raw_name.split()[0]

        images = [f for f in p_dir.glob("*") if f.suffix.lower() in valid_exts]
        if not images:
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * split_ratio))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:] if len(images) > 1 else images

        (train_dir / clean_class).mkdir(parents=True, exist_ok=True)
        (val_dir / clean_class).mkdir(parents=True, exist_ok=True)

        for img_f in train_imgs:
            shutil.copy(img_f, train_dir / clean_class / img_f.name)
            total_images += 1

        for img_f in val_imgs:
            shutil.copy(img_f, val_dir / clean_class / img_f.name)

    # Add 'Unknown' class with negative background samples
    unknown_train = train_dir / "Unknown"
    unknown_val = val_dir / "Unknown"
    unknown_train.mkdir(parents=True, exist_ok=True)
    unknown_val.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    for i in range(25):
        noise = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(unknown_train / f"unknown_{i}.jpg"), noise)
        if i % 5 == 0:
            cv2.imwrite(str(unknown_val / f"unknown_{i}.jpg"), noise)

    msg = f"[DATASET] Prepared dataset with {total_images} enrolled photos across {[d.name for d in (train_dir.iterdir()) if d.is_dir()]}"
    print(msg)
    update_status(status="running", progress=5, message=msg)
    return out_path


def train_classifier():
    # Clear previous status logs
    if STATUS_FILE.exists():
        try:
            STATUS_FILE.unlink()
        except Exception:
            pass

    update_status(status="running", progress=2, message="📁 Preparing multi-angle classification dataset...")
    dataset_path = prepare_classification_dataset()

    print("[TRAIN] Initializing YOLOv8 Medium Person Classification Model (yolov8m-cls.pt)...")
    update_status(status="running", progress=8, message="🧠 Loading YOLOv8 Medium Classification Neural Network (yolov8m-cls.pt)...")
    model = YOLO("yolov8m-cls.pt")

    def on_epoch_end(trainer):
        ep = trainer.epoch + 1
        pct = int((ep / trainer.epochs) * 90) + 8
        try:
            if hasattr(trainer, "loss") and trainer.loss is not None:
                loss_val = float(trainer.loss.item() if hasattr(trainer.loss, "item") else trainer.loss)
            elif trainer.loss_items is not None:
                if isinstance(trainer.loss_items, (list, tuple)):
                    loss_val = float(trainer.loss_items[0])
                else:
                    loss_val = float(trainer.loss_items.item() if hasattr(trainer.loss_items, "item") else trainer.loss_items)
            else:
                loss_val = 0.0
        except Exception:
            loss_val = 0.0

        msg = f"⚡ [EPOCH {ep}/{trainer.epochs}] Loss: {loss_val:.4f} | Device: Apple Silicon MPS GPU"
        print(msg)
        update_status(status="running", progress=pct, epoch=ep, total_epochs=trainer.epochs, loss=loss_val, message=msg)

    model.add_callback("on_train_epoch_end", on_epoch_end)

    print("[TRAIN] Fine-tuning High-Precision Person Classifier on enrolled facial profiles...")
    results = model.train(
        data=str(dataset_path),
        epochs=60,
        imgsz=224,
        batch=8,
        workers=2,
        degrees=30.0,
        fliplr=0.5,
        scale=0.25,
        perspective=0.001,
        erasing=0.10,
        device="mps" if torch_mps_available() else "cpu",
        project="runs/faces",
        name="person_classifier",
        exist_ok=True,
        verbose=True
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_weights = Path("runs/classify/runs/faces/person_classifier/weights/best.pt")
    target_weights = models_dir / "face_classifier.pt"

    if best_weights.exists():
        shutil.copy(best_weights, target_weights)
        msg = f"✅ SUCCESS: Exported fine-tuned Person Classifier to {target_weights}"
        print(msg)
        update_status(status="completed", progress=100, epoch=60, total_epochs=60, loss=0.0, message=msg)
        return True

    msg = "❌ ERROR: Could not find best.pt weights"
    update_status(status="failed", progress=0, message=msg)
    return False


def torch_mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    train_classifier()
