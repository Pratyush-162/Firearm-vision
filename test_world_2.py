import cv2
from ultralytics import YOLO

model = YOLO("yolov8m-world.pt")
classes = [
    "assault rifle", "gun", "rifle", "pistol", "shotgun", 
    "machine gun", "explosive", "grenade", 
    "knife", "sword", "machete", "baseball bat"
]
model.set_classes(classes)
img_path = "/Users/pratyushbharadwaj/.gemini/antigravity/brain/bb3c46be-7b6f-42be-9740-778035287a1a/.user_uploaded/media_1787139043707.png"

results = model(img_path, conf=0.01)
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls]
        print(f"Detected: {name} ({conf:.2f})")
