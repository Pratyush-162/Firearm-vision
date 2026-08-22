import os
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Models to test
models = {
    "Subh775_Firearm": ("Subh775/Firearm_Detection_Yolov8n", "best.pt"),
    "Hadi959_Weapon": ("Hadi959/weapon-detection-yolov8", "best.pt")
}

img_path = "/Users/pratyushbharadwaj/.gemini/antigravity/brain/bb3c46be-7b6f-42be-9740-778035287a1a/.user_uploaded/media_1787139043707.png"

for name, (repo, filename) in models.items():
    print(f"\n--- Testing {name} ---")
    try:
        model_path = hf_hub_download(repo_id=repo, filename=filename)
        model = YOLO(model_path)
        results = model(img_path, conf=0.1)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls] if hasattr(model, 'names') else str(cls)
                print(f"Detected: {label} ({conf:.2f})")
    except Exception as e:
        print(f"Failed to download or run {name}: {e}")
