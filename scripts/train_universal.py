from ultralytics import YOLO

def train():
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    print("Starting training on super_dataset.yaml...")
    results = model.train(
        data="data/merged/super_dataset.yaml",
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        device="mps",
        project="runs",
        name="universal-weapon"
    )
    
    print("Training complete!")

if __name__ == "__main__":
    train()
