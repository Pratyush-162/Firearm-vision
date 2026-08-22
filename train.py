import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Fine-tune a local detector on annotated visual-object data.")
    p.add_argument("--data", default="configs/firearms.yaml")
    p.add_argument("--base-model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision (keep False on MPS to prevent nan loss)")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="firearms")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Missing dataset config file: {args.data}")

    # Prevent 'nan' gradient explosion on Apple Silicon MPS by disabling AMP mixed precision on MPS
    use_amp = args.amp if args.device != "mps" else False

    print(f"[INFO] Starting YOLO training on device='{args.device}', imgsz={args.imgsz}, batch={args.batch}, amp={use_amp}")
    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=use_amp,
        patience=15,
        cos_lr=True,
        fliplr=0.5,
        mosaic=0.2,
        workers=4
    )
    print(f"[SUCCESS] Training complete. Best weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
