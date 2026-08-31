#!/usr/bin/env python3
"""
One-time setup script to pre-generate and cache face embeddings.
Run once: python setup_face_cache.py

This eliminates the ~30-60 second delay on first startup and any subsequent
app restart. After this, face recognition loads instantly from cache.
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face_recognizer import FaceIdentificationEngine

def main():
    print("[*] Setting up face recognition cache...")
    print("[*] This will compute embeddings for all enrolled faces (one-time only)")
    
    engine = FaceIdentificationEngine(enrolled_dir="data/enrolled_faces")
    
    if engine.app is None:
        print("[ERROR] InsightFace not initialized. Check your installation.")
        return False
    
    if engine.matrix_names:
        print(f"[✓] Successfully cached {len(engine.matrix_names)} subjects.")
        print(f"[✓] Next startup will load instantly from cache!")
        return True
    else:
        print("[WARN] No enrolled faces found. The app will still work in weapon-only mode.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
