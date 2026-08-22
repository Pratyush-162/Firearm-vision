import os
import re
import threading
from pathlib import Path
import cv2
import numpy as np
import insightface
from insightface.utils import face_align


class FaceIdentificationEngine:
    """
    Ultimate High-Precision Face Recognition Engine using InsightFace 5-Point Landmark Alignment
    + ArcFace (w600k_r50 ResNet50) 512-dimensional Embeddings & Centroid-Exemplar Matching.
    Thread-safe implementation preventing C++ ONNX Runtime memory double-free crashes.
    """
    def __init__(
        self,
        det_path="models/det_10g.onnx",
        rec_path="models/w600k_r50.onnx",
        enrolled_dir="data/enrolled_faces"
    ):
        self.lock = threading.Lock()
        self.det_path = Path(det_path)
        self.rec_path = Path(rec_path)
        self.enrolled_dir = Path(enrolled_dir)
        self.enrolled_dir.mkdir(parents=True, exist_ok=True)

        self.det_model = None
        self.rec_model = None
        self.yolo_cls = None
        self.enrolled_profiles = {}  # {name: {'centroid': vec, 'exemplars': [vec1, vec2, ...]}}

        # 0. Load Fine-Tuned Person Classifier Model if present
        self.reload_classifier()

        # 1. Initialize InsightFace 5-Point Landmark Detector
        if self.det_path.exists():
            try:
                self.det_model = insightface.model_zoo.get_model(str(self.det_path), providers=["CPUExecutionProvider"])
                self.det_model.prepare(ctx_id=0, input_size=(640, 640))
                print(f"[INFO] Initialized InsightFace 5-Landmark Face Detector from {self.det_path}")
            except Exception as e:
                print(f"[WARN] Could not initialize 5-landmark detector: {e}")

        # 2. Initialize InsightFace ArcFace w600k_r50 Model
        if self.rec_path.exists():
            try:
                self.rec_model = insightface.model_zoo.get_model(str(self.rec_path), providers=["CPUExecutionProvider"])
                self.rec_model.prepare(ctx_id=0)
                print(f"[INFO] Initialized InsightFace ArcFace (w600k_r50 ResNet50) Engine from {self.rec_path}")
            except Exception as e:
                print(f"[WARN] Could not initialize ArcFace recognizer: {e}")

        self.reload_enrolled_faces()

    def reload_classifier(self):
        """Loads or reloads the fine-tuned YOLOv8 person classification model from models/face_classifier.pt."""
        cls_path = Path("models/face_classifier.pt")
        if cls_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo_cls = YOLO(str(cls_path))
                print(f"[INFO] Successfully loaded Fine-Tuned Person Classifier Model from {cls_path}")
            except Exception as e:
                print(f"[WARN] Could not load fine-tuned person classifier: {e}")
        else:
            self.yolo_cls = None

    def reload_enrolled_faces(self):
        """Scans data/enrolled_faces/ and builds Centroid + Exemplar 512-d ArcFace embedding database."""
        if not self.rec_model:
            return

        with self.lock:
            self.enrolled_profiles.clear()
            valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = [p for p in self.enrolled_dir.rglob("*") if p.suffix.lower() in valid_extensions]

            temp_dict = {}
            count = 0

            for img_path in image_paths:
                if img_path.parent != self.enrolled_dir:
                    raw_name = img_path.parent.name.replace("_", " ").title()
                else:
                    raw_name = img_path.stem.replace("_", " ").title()

                clean_name = re.sub(r"[\d\_]+$", "", raw_name).strip()
                if not clean_name:
                    clean_name = raw_name

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                feat = self._extract_aligned_feature(img)
                if feat is not None:
                    if clean_name not in temp_dict:
                        temp_dict[clean_name] = []
                    temp_dict[clean_name].append(feat)
                    count += 1

            # Compute mean centroid feature vector for each subject
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

            print(f"[INFO] Built ArcFace Centroid-Exemplar Database: {count} photos across {len(self.enrolled_profiles)} enrolled subjects: {list(self.enrolled_profiles.keys())}")

    def _extract_aligned_feature(self, img_or_crop):
        """
        Extracts 512-d ArcFace embedding vector using 5-point landmark alignment if possible.
        """
        if not self.rec_model or img_or_crop is None or img_or_crop.size == 0:
            return None

        h, w = img_or_crop.shape[:2]
        if h < 12 or w < 12:
            return None

        try:
            aligned_crop = None

            # Attempt 5-point landmark alignment
            if self.det_model is not None and (h >= 40 and w >= 40):
                try:
                    bboxes, kpss = self.det_model.detect(img_or_crop, max_num=1, metric="default")
                    if kpss is not None and len(kpss) > 0:
                        aligned_crop = face_align.norm_crop(img_or_crop, landmark=kpss[0])
                except Exception:
                    pass

            if aligned_crop is None:
                aligned_crop = cv2.resize(img_or_crop, (112, 112), interpolation=cv2.INTER_CUBIC)

            rgb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB)
            feat = self.rec_model.get_feat(rgb)
            if feat is not None:
                feat = feat.flatten()
                norm = np.linalg.norm(feat)
                if norm > 0:
                    return feat / norm
        except Exception:
            pass

        return None

    def identify_face(self, frame, box_xyxy, landmark=None):
        """
        Matches a detected face box against enrolled subject profiles using ArcFace Centroid + Exemplar Cosine Similarity.
        Thread-safe execution preventing ONNX Runtime C++ malloc double-free crashes.
        """
        if not self.rec_model or not self.enrolled_profiles:
            return "Unknown", 0.95, False

        with self.lock:
            x1, y1, x2, y2 = map(int, box_xyxy)
            h_frame, w_frame = frame.shape[:2]

            aligned_crop = None
            if landmark is not None:
                try:
                    aligned_crop = face_align.norm_crop(frame, landmark=landmark)
                except Exception:
                    pass

            if aligned_crop is None:
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
                cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                cx2, cy2 = min(w_frame, x2 + pad_w), min(h_frame, y2 + pad_h)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                    return "Unknown", 0.95, False

                aligned_crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)

            # Blur Quality Gate: Measure Laplacian variance on crop
            gray_crop = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY) if len(aligned_crop.shape) == 3 else aligned_crop
            lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            if lap_var < 18.0:
                # Blurry/low-quality face frame: Flag as low-confidence unknown so locked identity is not demoted
                return "Unknown", 0.45, False

            rgb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB)
            feat = self.rec_model.get_feat(rgb)
            if feat is None:
                return "Unknown", 0.50, False

            feat = feat.flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            best_score = -1.0
            second_best_score = -1.0
            best_name = None

            # Dual-score matching: 0.6 * Centroid_Sim + 0.4 * Max_Exemplar_Sim
            for name, profile in self.enrolled_profiles.items():
                c_score = float(np.dot(profile["centroid"], feat))
                max_e_score = max([float(np.dot(ex, feat)) for ex in profile["exemplars"]]) if profile["exemplars"] else c_score
                combined_score = 0.6 * c_score + 0.4 * max_e_score

                if combined_score > best_score:
                    if best_name and best_name != name:
                        second_best_score = best_score
                    best_score = combined_score
                    best_name = name
                elif combined_score > second_best_score and name != best_name:
                    second_best_score = combined_score

            # High-Sensitivity Thresholding: Requires ArcFace Cosine Similarity >= 0.45 to match enrolled person!
            ratio_test_passed = (second_best_score < 0) or (best_score - second_best_score >= 0.04)
            if best_name and best_score >= 0.45 and ratio_test_passed:
                print(f"[RECOGNITION MATCH] Person: {best_name} | ArcFace Sim: {best_score:.3f}")
                return f"Person: {best_name}", best_score, True

            return "Unknown", 0.90, False

    def enroll_person(self, name, image_or_images):
        """
        Registers a new person profile into data/enrolled_faces/{person_name}/.
        Saves facial photos from multiple angles for robust recognition.
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
