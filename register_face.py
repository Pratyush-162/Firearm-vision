import argparse
import pickle
from pathlib import Path

import face_recognition


def build_registry(source_dir, db_path):
    registry = {}
    for person_dir in sorted(Path(source_dir).iterdir()):
        if not person_dir.is_dir():
            continue
        encodings = []
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = face_recognition.load_image_file(str(img_path))
            faces = face_recognition.face_encodings(img)
            if faces:
                encodings.append(faces[0])
            else:
                print(f"  no face found in {img_path.name}, skipping")
        if encodings:
            registry[person_dir.name] = encodings
            print(f"Registered {person_dir.name}: {len(encodings)} samples")
        else:
            print(f"Skipped {person_dir.name}: no usable photos")

    with open(db_path, "wb") as f:
        pickle.dump(registry, f)
    print(f"\nSaved registry to {db_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="registered_faces", help="Folder containing one subfolder per person")
    p.add_argument("--output", default="registry.pkl")
    args = p.parse_args()
    build_registry(args.source, args.output)
