import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import LABELS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_for(image, all_labels):
    direct = image.with_suffix(".txt")
    if direct.exists(): return direct
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"; alternative = Path(*parts).with_suffix(".txt")
        if alternative.exists(): return alternative
    return all_labels.get(image.stem)


def source_split(path):
    names = {part.lower() for part in path.parts}
    if names & {"val", "valid", "validation"}: return "valid"
    if names & {"test", "testing"}: return "test"
    if names & {"train", "training"}: return "train"
    return None


def main():
    p = argparse.ArgumentParser(description="Normalize a downloaded YOLO dataset into this project's expected layout.")
    p.add_argument("--input-dir", type=Path, required=True, help="Unzipped Kaggle dataset folder")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed")); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); random.seed(args.seed)
    if any(args.output_dir.glob("*/images/*")):
        raise SystemExit(f"{args.output_dir} already has data. Use a new output directory to avoid overwriting it.")
    images = sorted(x for x in args.input_dir.rglob("*") if x.suffix.lower() in IMAGE_EXTENSIONS)
    labels = {x.stem: x for x in args.input_dir.rglob("*.txt")}
    if not images: raise SystemExit("No images found. Unzip the Kaggle download first.")
    counts = {"train": 0, "valid": 0, "test": 0}
    for image in images:
        split = source_split(image) or random.choices(["train", "valid", "test"], weights=[.7, .15, .15])[0]
        label = label_for(image, labels)
        if label:
            for line in label.read_text().splitlines():
                if not line.strip(): continue
                values = line.split()
                if len(values) != 5 or not values[0].isdigit() or not 0 <= int(values[0]) < len(LABELS):
                    raise SystemExit(f"Unexpected YOLO label in {label}: {line}")
        destination = args.output_dir / split
        (destination / "images").mkdir(parents=True, exist_ok=True); (destination / "labels").mkdir(parents=True, exist_ok=True)
        name = f"{counts[split]:07d}{image.suffix.lower()}"
        shutil.copy2(image, destination / "images" / name)
        shutil.copy2(label, destination / "labels" / Path(name).with_suffix(".txt")) if label else (destination / "labels" / Path(name).with_suffix(".txt")).touch()
        counts[split] += 1
    print("Imported:", counts)


if __name__ == "__main__": main()
