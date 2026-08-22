import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Evaluate only on a held-out test split.")
    p.add_argument("--model", required=True); p.add_argument("--data", default="configs/firearms.yaml")
    p.add_argument("--imgsz", type=int, default=960); p.add_argument("--device", default="mps"); p.add_argument("--output", type=Path, default=Path("outputs/test_metrics.json"))
    args = p.parse_args(); metrics = YOLO(args.model).val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, plots=True)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(metrics.results_dict, indent=2))
    print(json.dumps(metrics.results_dict, indent=2))


if __name__ == "__main__": main()
