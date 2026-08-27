"""
State-of-the-Art Face Recognition Engine
=========================================
Uses InsightFace buffalo_l model pack (auto-downloads ~300MB on first run):
  - SCRFD-10G:      High-accuracy face detector with 5-point landmark alignment
  - ArcFace w600k_r50: 512-dimensional face embedding model (trained on 600K identities)

Centroid + Exemplar matching with SIMD-vectorized cosine similarity.
Thread-safe implementation for real-time video stream processing.

Models are cached at ~/.insightface/models/buffalo_l/
"""

import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARN] insightface package not installed. Run: pip install insightface onnxruntime")


class FaceIdentificationEngine:
    """
    Production-grade face identification engine using InsightFace buffalo_l.

    Architecture:
      Detection:    SCRFD-10G with 5-point landmark alignment (99.80% WiderFace Easy)
      Recognition:  ArcFace w600k_r50 (512-d embeddings, 99.83% LFW, 98.30% CFP-FP)
      Matching:     Centroid + Exemplar cosine similarity with Lowe's ratio test

    Models auto-download to ~/.insightface/models/buffalo_l/ on first initialization.
    """

    # ── Recognition thresholds (tuned for ArcFace w600k_r50) ──────────────────
    MATCH_THRESHOLD = 0.32        # Minimum cosine similarity for positive match
    RATIO_TEST_MARGIN = 0.02      # Minimum gap between 1st and 2nd best match
    BLUR_QUALITY_GATE = 8.0       # Minimum Laplacian variance for usable face
    DET_THRESHOLD = 0.50          # Face detection confidence threshold

    def __init__(self, enrolled_dir="data/enrolled_faces"):
        self.lock = threading.Lock()
        self.enrolled_dir = Path(enrolled_dir)
        self.enrolled_dir.mkdir(parents=True, exist_ok=True)

        # Model references (backward compatibility with stream_manager)
        self.app = None
        self.det_model = None
        self.rec_model = None

        # Enrolled face database
        self.enrolled_profiles = {}
        self.matrix_names = []
        self.centroid_matrix = np.empty((0, 512), dtype=np.float32)
        self.exemplar_matrix = np.empty((0, 512), dtype=np.float32)
        self.exemplar_bounds = []

        if not INSIGHTFACE_AVAILABLE:
            return

        # Initialize InsightFace with buffalo_l (SCRFD-10G + ArcFace w600k_r50)
        self._init_insightface()

        # Build face embedding database from enrolled photos
        self.reload_enrolled_faces()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_insightface(self):
        """Initialize InsightFace FaceAnalysis with auto-download of model pack."""
        model_packs = ["buffalo_l", "antelopev2", "buffalo_s"]

        for pack_name in model_packs:
            try:
                print(f"[INFO] Initializing InsightFace model pack: {pack_name} ...")
                self.app = FaceAnalysis(
                    name=pack_name,
                    providers=["CPUExecutionProvider"]
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))

                # Extract individual model handles for backward compatibility
                if hasattr(self.app, "det_model") and self.app.det_model is not None:
                    self.det_model = self.app.det_model

                if hasattr(self.app, "models"):
                    for model in self.app.models.values():
                        if hasattr(model, "taskname") and model.taskname == "recognition":
                            self.rec_model = model
                            break

                det_name = type(self.det_model).__name__ if self.det_model else "None"
                rec_name = type(self.rec_model).__name__ if self.rec_model else "None"
                print(f"[INFO] ✓ InsightFace {pack_name} ready | Detector: {det_name} | Recognizer: {rec_name}")
                return

            except Exception as e:
                print(f"[WARN] Could not initialize {pack_name}: {e}")
                self.app = None
                continue

        print("[ERROR] All InsightFace model packs failed to initialize. Face recognition disabled.")

    # ── High-Level API ────────────────────────────────────────────────────────

    def detect_and_identify(self, frame, max_faces=10, det_thresh=None):
        """
        Detect all faces in frame and identify them against enrolled profiles.

        Returns:
            list[dict]: Each dict contains:
                "box":        [x1, y1, x2, y2]
                "label":      "Person: Name" or "Unknown"
                "confidence": float
                "landmark":   5-point keypoints or None
                "is_matched": bool
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        thresh = det_thresh or self.DET_THRESHOLD
        results = []

        with self.lock:
            try:
                faces = self.app.get(frame)

                # Sort by face area (largest first) and limit
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )[:max_faces]

                for face in faces:
                    if face.det_score < thresh:
                        continue

                    x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                    kps = face.kps if hasattr(face, "kps") else None
                    embedding = face.embedding if hasattr(face, "embedding") else None

                    # Quality gate: reject blurry / low-resolution faces
                    h_f, w_f = frame.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_f, x2), min(h_f, y2)
                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                        continue

                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

                    if lap_var < self.BLUR_QUALITY_GATE:
                        results.append({
                            "box": [x1, y1, x2, y2],
                            "label": "Unknown",
                            "confidence": float(face.det_score),
                            "landmark": kps,
                            "is_matched": False
                        })
                        continue

                    # Match embedding against enrolled database
                    if embedding is not None and len(self.matrix_names) > 0:
                        label, score, is_matched = self._match_embedding(embedding)
                    else:
                        label, score, is_matched = "Unknown", float(face.det_score), False

                    results.append({
                        "box": [x1, y1, x2, y2],
                        "label": label,
                        "confidence": max(float(face.det_score), score),
                        "landmark": kps,
                        "is_matched": is_matched
                    })

            except Exception as e:
                print(f"[WARN] Face detection/identification error: {e}")

        return results

    def detect_faces_only(self, frame, max_faces=5, det_thresh=0.30):
        """
        Detect face bounding boxes only (no recognition). Thread-safe.
        Used for enrollment capture and privacy blurring.

        Returns:
            list: [[x1, y1, x2, y2], ...]
        """
        if self.det_model is None or frame is None or frame.size == 0:
            return []

        with self.lock:
            try:
                bboxes, _ = self.det_model.detect(frame, max_num=max_faces)
                if bboxes is not None:
                    return [bbox[:4].astype(int).tolist() for bbox in bboxes if bbox[4] >= det_thresh]
            except Exception:
                pass
        return []

    def identify_face(self, frame, box_xyxy, landmark=None):
        """
        Match a single face region against enrolled profiles.
        Backward-compatible interface.

        Returns:
            tuple: (label: str, confidence: float, is_matched: bool)
        """
        if self.app is None or not self.enrolled_profiles:
            return "Unknown", 0.95, False

        with self.lock:
            try:
                x1, y1, x2, y2 = map(int, box_xyxy)
                h_frame, w_frame = frame.shape[:2]

                # Pad around face crop for better alignment
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = int(bw * 0.25), int(bh * 0.25)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w_frame, x2 + pad_w)
                cy2 = min(h_frame, y2 + pad_h)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                    return "Unknown", 0.95, False

                # Blur quality gate
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                if lap_var < self.BLUR_QUALITY_GATE:
                    return "Unknown", 0.45, False

                # Run InsightFace on the padded crop
                faces = self.app.get(crop)
                if not faces:
                    return "Unknown", 0.50, False

                # Use the largest detected face in the crop
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                if face.embedding is None:
                    return "Unknown", 0.50, False

                return self._match_embedding(face.embedding)

            except Exception:
                return "Unknown", 0.50, False

    # ── Matching Engine ───────────────────────────────────────────────────────

    def _match_embedding(self, embedding):
        """
        Match a 512-d face embedding against enrolled profiles using
        centroid + exemplar cosine similarity with Lowe's ratio test.

        Returns:
            tuple: (label, best_score, is_matched)
        """
        if len(self.matrix_names) == 0:
            return "Unknown", 0.95, False

        feat = np.array(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        # SIMD-vectorized cosine similarity via BLAS matrix multiplication
        c_scores = np.dot(self.centroid_matrix, feat)

        if self.exemplar_matrix.shape[0] > 0:
            e_scores = np.dot(self.exemplar_matrix, feat)
        else:
            e_scores = np.array([])

        # Combined score: 60% centroid similarity + 40% best exemplar similarity
        combined_scores = np.zeros_like(c_scores)
        for idx in range(len(self.matrix_names)):
            start, end = self.exemplar_bounds[idx]
            if start < end:
                max_e_score = np.max(e_scores[start:end])
            else:
                max_e_score = c_scores[idx]
            combined_scores[idx] = 0.6 * c_scores[idx] + 0.4 * max_e_score

        best_idx = int(np.argmax(combined_scores))
        best_score = float(combined_scores[best_idx])
        best_name = self.matrix_names[best_idx]

        # Lowe's ratio test: gap between best and second-best must exceed margin
        if len(combined_scores) > 1:
            temp_scores = combined_scores.copy()
            temp_scores[best_idx] = -1.0
            second_best_score = float(np.max(temp_scores))
        else:
            second_best_score = -1.0

        ratio_test_passed = (second_best_score < 0) or (best_score - second_best_score >= self.RATIO_TEST_MARGIN)

        if best_name and best_score >= self.MATCH_THRESHOLD and ratio_test_passed:
            print(f"[RECOGNITION ✓] {best_name} | ArcFace Cosine: {best_score:.3f}")
            return f"Person: {best_name}", best_score, True

        return "Unknown", 0.90, False

    # ── Enrollment Database ───────────────────────────────────────────────────

    def reload_enrolled_faces(self):
        """
        Scan data/enrolled_faces/ and build centroid + exemplar 512-d ArcFace
        embedding database using full InsightFace pipeline (detect → align → embed).
        Now includes ultra-fast Pickle caching for instant server reboots.
        """
        if self.app is None:
            return

        import pickle

        with self.lock:
            cache_file = self.enrolled_dir / "arcface_cache.pkl"
            valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = [p for p in self.enrolled_dir.rglob("*") if p.suffix.lower() in valid_extensions]

            # Check if cache is valid (newer than all images)
            try:
                if cache_file.exists():
                    cache_mtime = cache_file.stat().st_mtime
                    # Get newest modified image
                    latest_img_mtime = max((p.stat().st_mtime for p in image_paths), default=0)
                    
                    if cache_mtime >= latest_img_mtime:
                        with open(cache_file, "rb") as f:
                            data = pickle.load(f)
                        self.enrolled_profiles = data["enrolled_profiles"]
                        self.matrix_names = data["matrix_names"]
                        self.centroid_matrix = data["centroid_matrix"]
                        self.exemplar_matrix = data["exemplar_matrix"]
                        self.exemplar_bounds = data["exemplar_bounds"]
                        print(f"[INFO] ⚡ Loaded ArcFace Database from fast cache! ({len(self.matrix_names)} subjects)")
                        return
            except Exception as e:
                print(f"[WARN] Cache load failed, rebuilding database... ({e})")

            self.enrolled_profiles.clear()

            temp_dict = {}
            count = 0

            print("[INFO] Rebuilding ArcFace database from raw images... (This takes a few seconds)")

            for img_path in image_paths:
                # Extract person name from directory structure
                if img_path.parent != self.enrolled_dir:
                    raw_name = img_path.parent.name.replace("_", " ").title()
                else:
                    raw_name = img_path.stem.replace("_", " ").title()

                clean_name = re.sub(r"[\d\_]+$", "", raw_name).strip()
                if not clean_name:
                    clean_name = raw_name

                # Skip auto-captured unknowns
                if clean_name.lower().startswith("unknown") or clean_name.lower().startswith("associate"):
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                feat = self._extract_embedding_from_image(img)
                if feat is not None:
                    if clean_name not in temp_dict:
                        temp_dict[clean_name] = []
                    temp_dict[clean_name].append(feat)
                    count += 1

            # Build SIMD-vectorized centroid + exemplar matrices
            self.matrix_names = []
            centroid_list = []
            exemplar_list = []
            self.exemplar_bounds = []

            for name, feats in temp_dict.items():
                if feats:
                    mean_vec = np.mean(feats, axis=0)
                    norm = np.linalg.norm(mean_vec)
                    if norm > 0:
                        mean_vec = mean_vec / norm

                    self.enrolled_profiles[name] = {
                        "centroid": mean_vec,
                        "exemplars": feats
                    }

                    self.matrix_names.append(name)
                    centroid_list.append(mean_vec)
                    start_idx = len(exemplar_list)
                    exemplar_list.extend(feats)
                    end_idx = len(exemplar_list)
                    self.exemplar_bounds.append((start_idx, end_idx))

            if centroid_list:
                self.centroid_matrix = np.array(centroid_list, dtype=np.float32)
                self.exemplar_matrix = np.array(exemplar_list, dtype=np.float32)
            else:
                self.centroid_matrix = np.empty((0, 512), dtype=np.float32)
                self.exemplar_matrix = np.empty((0, 512), dtype=np.float32)

            print(
                f"[INFO] ✓ ArcFace Database: {count} photos across "
                f"{len(self.enrolled_profiles)} subjects: {list(self.enrolled_profiles.keys())}"
            )

            # Save the new cache!
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({
                        "enrolled_profiles": self.enrolled_profiles,
                        "matrix_names": self.matrix_names,
                        "centroid_matrix": self.centroid_matrix,
                        "exemplar_matrix": self.exemplar_matrix,
                        "exemplar_bounds": self.exemplar_bounds
                    }, f)
                print("[INFO] ⚡ ArcFace database cached successfully for instant reboots!")
            except Exception as e:
                print(f"[WARN] Failed to write cache: {e}")

    def _extract_embedding_from_image(self, img):
        """
        Extract a normalized 512-d ArcFace embedding from an image.
        Uses InsightFace full pipeline: detect → 5-point align → embed.
        Falls back to center-crop + direct recognition model if detection fails.
        """
        if self.app is None or img is None or img.size == 0:
            return None

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return None

        try:
            faces = self.app.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                if face.embedding is not None:
                    feat = face.embedding.flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        return feat / norm

            # Fallback: direct recognition model on resized crop
            if self.rec_model is not None:
                aligned = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                feat = self.rec_model.get_feat(rgb)
                if feat is not None:
                    feat = feat.flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        return feat / norm

        except Exception:
            pass

        return None

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll_person(self, name, image_or_images):
        """
        Register a new person into data/enrolled_faces/{person_name}/.
        Saves face crop images and rebuilds the embedding database.
        """
        clean_folder = name.lower().strip().replace(" ", "_")
        person_dir = self.enrolled_dir / clean_folder
        person_dir.mkdir(parents=True, exist_ok=True)

        images = image_or_images if isinstance(image_or_images, (list, tuple)) else [image_or_images]
        saved_count = 0

        for idx, item in enumerate(images, 1):
            if isinstance(item, (str, Path)):
                img = cv2.imread(str(item))
            else:
                img = item

            if img is not None:
                target_path = person_dir / f"{clean_folder}_{idx}.jpg"
                cv2.imwrite(str(target_path), img)
                saved_count += 1

        if saved_count > 0:
            print(f"[SUCCESS] Enrolled {saved_count} face photos for '{name}' in {person_dir}")
            self.reload_enrolled_faces()
            return True
        return False
