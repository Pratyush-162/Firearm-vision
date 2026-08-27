from ultralytics import YOLO

def main():
    print("[INFO] Loading your custom model: best.pt")
    model = YOLO("best.pt")
    
    print("[INFO] Starting fine-tuning training on your new dataset...")
    print("[INFO] This will run for 20 epochs. You can stop it early if you are happy with the accuracy.")
    
    # Train the model
    results = model.train(
        data="weapon detection/data.yaml", 
        epochs=20,          # 20 epochs is a good amount for fine-tuning
        imgsz=640,          # Standard image size
        batch=16,           # Good default batch size
        device="mps",       # Force Apple Silicon GPU
        project="runs",     # Save results in the runs folder
        name="fine_tuned"   # Save as runs/fine_tuned/weights/best.pt
    )
    
    print("\n[SUCCESS] Training complete!")
    print("Your new and improved model is saved at: runs/fine_tuned/weights/best.pt")

if __name__ == "__main__":
    main()
