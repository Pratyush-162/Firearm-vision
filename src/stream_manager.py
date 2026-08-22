import csv
import os
import time
import threading
from collections import deque
from pathlib import Path
import cv2
import numpy as np

class OwlV2Wrapper:
    def __init__(self, model_id="google/owlv2-large-patch14-ensemble"):
        print(f"[INFO] Initializing HuggingFace OWLv2: {model_id} ...")
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        import torch
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Moving OWLv2 to device: {self.device}")
        self.model.to(self.device)
        self.texts = [["firearm", "gun", "rifle", "assault rifle", "pistol"]]
        self.names = {i: t for i, t in enumerate(self.texts[0])}

    def __call__(self, img, conf=0.1, **kwargs):
        import torch
        from PIL import Image
        import cv2

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        inputs = self.processor(text=self.texts, images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=conf)
        
        class MockBox:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = [xyxy]
                self.conf = [conf]
                self.cls = [cls]
                
        class MockResult:
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names
                
        boxes = []
        for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            b = [round(i, 2) for i in box.tolist()]
            boxes.append(MockBox(b, score.item(), label.item()))
            
        return [MockResult(boxes, self.names)]

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.temporal import Candidate, TemporalConfirm
from src.face_recognizer import FaceIdentificationEngine

# Dictionary mapping weapon labels to normalized display categories
WEAPON_CLASSES = {
    # Long Firearms & Rifles
    "automatic_rifle": "Firearm (Automatic Rifle)",
    "assault_rifle": "Firearm (Assault Rifle)",
    "rifle": "Firearm (Rifle)",
    "shotgun": "Firearm (Shotgun)",
    "sniper_rifle": "Firearm (Sniper Rifle)",
    "smg": "Firearm (SMG)",
    "bazooka": "Explosive Weapon (Bazooka)",
    "grenade_launcher": "Explosive Weapon (Grenade Launcher)",

    # Handguns & Pistols
    "handgun": "Firearm (Handgun)",
    "pistol": "Firearm (Pistol)",
    "revolver": "Firearm (Revolver)",
    "glock": "Firearm (Glock)",
    "gun": "Firearm (Gun)",
    "firearm": "Firearm",
    
    # Edged & Sharp Weapons
    "knife": "Edged Weapon (Knife)",
    "blade": "Edged Weapon (Blade)",
    "sword": "Edged Weapon (Sword)",
    "machete": "Edged Weapon (Machete)",
    "dagger": "Edged Weapon (Dagger)",
    "pocket_knife": "Edged Weapon (Pocket Knife)",
    "combat_knife": "Edged Weapon (Combat Knife)",
    "scissors": "Sharp Object (Scissors)",
    "axe": "Bladed Weapon (Axe)",

    # Blunt & Impact Weapons
    "bat": "Blunt Weapon (Bat)",
    "baseball_bat": "Blunt Weapon (Baseball Bat)",
    "pipe": "Blunt Object (Pipe)",
    "crowbar": "Blunt Tool (Crowbar)",

    # Generic & Open-Vocabulary
    "weapon": "Weapon",
    "dangerous_object": "Dangerous Object"
}

# Standard non-weapon COCO classes
NON_WEAPON_CLASSES = {
    "face": "Face",
    "person": "Face",
    "human": "Face",
    "man": "Face",
    "woman": "Face",
    "cell phone": "Cell Phone",
    "phone": "Phone",
    "bottle": "Bottle",
    "cup": "Cup",
    "chair": "Chair",
    "couch": "Couch",
    "laptop": "Laptop",
    "tv": "TV",
    "book": "Book",
    "car": "Car",
    "dog": "Dog",
    "cat": "Cat",
    "backpack": "Backpack",
    "handbag": "Handbag",
    "umbrella": "Umbrella",
    "tie": "Tie",
    "suitcase": "Suitcase",
    "clock": "Clock",
    "vase": "Vase"
}


def parse_source(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value_str = value.strip()
        if value_str.isdigit():
            return int(value_str)
        return value_str
    return value


def apply_clahe_enhancement(frame):
    """Enhance low-light and low-contrast CCTV frames using LAB color space + CLAHE."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        return frame


def apply_sharpening_filter(frame):
    """Unsharp Masking filter for CCTV footage to sharpen blurry metallic weapon edges."""
    try:
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.5)
        sharpened = cv2.addWeighted(frame, 1.4, gaussian, -0.4, 0)
        return sharpened
    except Exception as e:
        return frame


def non_max_suppression_candidates(candidates, overlap_thresh=0.25, iomin_thresh=0.35):
    """Enhanced NMS checking both IoU and IoMin (containment) to prevent multiple overlapping boxes."""
    if not candidates:
        return []

    boxes = np.array([c.box for c in candidates], dtype=np.float32)
    scores = np.array([c.confidence for c in candidates], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(1e-6, union)
        iomin = inter / np.maximum(1e-6, np.minimum(areas[i], areas[order[1:]]))

        # Suppress candidate if IoU > overlap_thresh OR IoMin > iomin_thresh (nested containment)
        inds = np.where((iou <= overlap_thresh) & (iomin <= iomin_thresh))[0]
        order = order[inds + 1]

    return [candidates[k] for k in keep]


class RTSPStreamManager:
    """
    Asynchronous multi-threaded stream manager supporting RTSP URLs, webcams, and video files.
    Decouples frame capture (30-60 FPS) from YOLO inference so live streaming is 100% real-time.
    Supports target mode switching: 'weapons_only', 'faces_only', or 'all'.
    """
    def __init__(
        self,
        model_path="runs/firearms/weights/best.pt",
        source="0",
        confidence=0.35,
        frames=5,
        required=2,
        faces=False,
        blur_faces=False,
        cctv_mode=True,
        clahe_enhance=False,
        sharpness_boost=False,
        tile_inference=False,
        imgsz=960,
        detect_mode="weapons_only",  # 'weapons_only', 'faces_only' (or 'persons_only'), or 'all'
        weapon_filter="all_weapons",
        output_dir=Path("outputs")
    ):
        self.model_path = model_path
        self.source = parse_source(source)
        self.source_str = str(source)
        self.confidence = confidence
        self.frames_window = frames
        self.required_matches = required
        self.faces = faces
        self.blur_faces = blur_faces
        
        # Target Detection Mode
        self.detect_mode = detect_mode
        
        # CCTV & Performance settings
        self.cctv_mode = cctv_mode
        self.clahe_enhance = clahe_enhance
        self.sharpness_boost = sharpness_boost
        self.tile_inference = tile_inference
        self.imgsz = imgsz
        self.weapon_filter = weapon_filter
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from src.face_tracker import SmoothObjectTracker
        from src.lightweight_tracker import LightweightTrackerEngine
        self.tracker = SmoothObjectTracker(max_missed_frames=30, iou_threshold=0.20)
        self.klt_tracker = LightweightTrackerEngine(dist_thresh=200.0, max_missed=15)

        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.status_message = "Initializing"
        self._needs_reconnect = True
        self._pending_source = self.source
        self._pending_source_str = self.source_str

        # Frame buffers
        self.cap = None
        self.latest_raw_frame = None
        self.latest_jpeg = None
        
        # Performance & Stats
        self.fps = 0.0
        self.infer_fps = 0.0
        self.frame_count = 0
        self.active_detections = []
        self.is_confirmed_alert = False
        self.alert_history = deque(maxlen=100)
        self._alert_open = False

        # Worker Threads & Alert CSV
        self._capture_thread = None
        self._inference_thread = None
        self.alert_csv_path = self.output_dir / "alerts.csv"
        self._init_alerts_csv()

        # Load models (Custom best.pt + YOLO 360° Person Detector + RetinaFace 5-Landmark Detector + ArcFace Person Identifier)
        self.custom_model = self._load_custom_weapon_model(model_path)
        self.person_model = self._load_person_model()
        self.pose_model = self._load_pose_model()
        self.face_detector = self._load_face_detector()
        self.face_recognizer = FaceIdentificationEngine()
        self.gate = TemporalConfirm(frames=self.frames_window, required=self.required_matches)

    def _load_person_model(self):
        if not YOLO:
            return None
        person_paths = [
            Path("yolov8n.pt"),
            Path("yolo11n.pt")
        ]
        for p in person_paths:
            if p.exists():
                print(f"[INFO] Loading YOLO 360-Degree Person Detector from {p}")
                try:
                    return YOLO(str(p))
                except Exception as e:
                    print(f"[WARN] Could not load person model {p}: {e}")
        return None

    def _load_pose_model(self):
        if not YOLO:
            return None
        pose_paths = [
            Path("yolov8n-pose.pt"),
            Path("yolo11n-pose.pt")
        ]
        for p in pose_paths:
            if p.exists():
                print(f"[INFO] Loading 360-Degree Pose Head Anchor model from {p}")
                try:
                    return YOLO(str(p))
                except Exception as e:
                    print(f"[WARN] Could not load pose model {p}: {e}")
        return None

    def _load_custom_weapon_model(self, path):
        if path and "owlv2" in path.lower():
            return OwlV2Wrapper()

        if not YOLO:
            print("[WARN] ultralytics package not available.")
            return None

        custom_paths = [
            Path(path) if path else None,
            Path("runs/hf-yolov8n-best.pt")
        ]
        custom_paths = [p for p in custom_paths if p is not None]
        for cp in custom_paths:
            if cp.exists():
                print(f"[INFO] Loading weapon model from {cp}")
                try:
                    return YOLO(str(cp))
                except Exception as e:
                    print(f"[WARN] Could not load model {cp}: {e}")
        return None

    def _load_face_detector(self):
        try:
            if Path("yolov8n-face.pt").exists() and YOLO:
                print("[INFO] Loading ultra-fast 60 FPS YOLO face model yolov8n-face.pt")
                return YOLO("yolov8n-face.pt")
            elif Path("yolov8l-face.pt").exists() and YOLO:
                print("[INFO] Loading high-capacity YOLO face model yolov8l-face.pt")
                return YOLO("yolov8l-face.pt")
        except Exception as e:
            print(f"[WARN] Failed to load YOLO face model: {e}")

        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                return detector
        except Exception as e:
            print(f"[WARN] Failed to load OpenCV face detector: {e}")
        return None

    def _init_alerts_csv(self):
        if not self.alert_csv_path.exists():
            with self.alert_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["time_seconds", "frame", "label", "confidence", "status"])
                writer.writeheader()

    def start(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self):
        self.running = False
        self._needs_reconnect = False
        if self._capture_thread and self._capture_thread.is_alive():
            try:
                self._capture_thread.join(timeout=1.0)
            except Exception:
                pass
        self.connected = False

    def connect(self, new_source):
        with self.lock:
            self.source_str = str(new_source)
            self.source = parse_source(new_source)
            self._pending_source = self.source
            self._pending_source_str = self.source_str
            self._needs_reconnect = True
            self.connected = False
            self.status_message = f"Connecting to {self.source_str}..."
            self.gate = TemporalConfirm(frames=self.frames_window, required=self.required_matches)
            self._alert_open = False

    def update_settings(
        self,
        confidence=None,
        frames=None,
        required=None,
        faces=None,
        blur_faces=None,
        cctv_mode=None,
        clahe_enhance=None,
        sharpness_boost=None,
        tile_inference=None,
        imgsz=None,
        detect_mode=None,
        weapon_filter=None
    ):
        with self.lock:
            if confidence is not None:
                self.confidence = float(confidence)
                # Instantly vanish any existing tracked bounding boxes that fall below the new threshold
                if hasattr(self, 'tracker') and self.tracker:
                    self.tracker.tracks = [t for t in self.tracker.tracks if getattr(t, 'confidence', 0.0) >= self.confidence]
            if frames is not None or required is not None:
                if frames is not None:
                    self.frames_window = int(frames)
                if required is not None:
                    self.required_matches = int(required)
                self.gate = TemporalConfirm(frames=self.frames_window, required=self.required_matches)
            if faces is not None:
                self.faces = bool(faces)
            if blur_faces is not None:
                self.blur_faces = bool(blur_faces)
            if cctv_mode is not None:
                self.cctv_mode = bool(cctv_mode)
            if clahe_enhance is not None:
                self.clahe_enhance = bool(clahe_enhance)
            if sharpness_boost is not None:
                self.sharpness_boost = bool(sharpness_boost)
            if tile_inference is not None:
                self.tile_inference = bool(tile_inference)
            if imgsz is not None:
                self.imgsz = int(imgsz)
            if detect_mode is not None:
                if self.detect_mode != str(detect_mode):
                    self.detect_mode = str(detect_mode)
                    if hasattr(self, 'tracker') and self.tracker:
                        self.tracker.tracks = []
                    self.active_detections = []
            if weapon_filter is not None:
                self.weapon_filter = str(weapon_filter)

    def _detect_and_blur_faces(self, frame):
        """Applies Gaussian Blur for face privacy if blur_faces is enabled."""
        if not self.blur_faces or not self.face_detector:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        for (x, y, w, h) in faces:
            region = frame[y:y+h, x:x+w]
            if region.size > 0:
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(region, (31, 31), 0)

    def _classify_detection(self, raw_label):
        lbl = raw_label.lower().replace(" ", "_").strip()
        
        # Check known weapon dictionary
        if lbl in WEAPON_CLASSES:
            return WEAPON_CLASSES[lbl], True
            
        # Check partial keyword weapon matches
        if any(kw in lbl for kw in ["gun", "rifle", "pistol", "revolver", "firearm", "shotgun", "glock", "launcher"]):
            return f"Firearm ({raw_label.replace('_', ' ').title()})", True
        if any(kw in lbl for kw in ["knife", "blade", "machete", "sword", "dagger", "scissors"]):
            return f"Edged Weapon ({raw_label.replace('_', ' ').title()})", True
        if any(kw in lbl for kw in ["bat", "pipe", "club", "crowbar"]):
            return f"Blunt Weapon ({raw_label.replace('_', ' ').title()})", True
            
        # Non-weapon COCO classes
        if lbl in NON_WEAPON_CLASSES:
            return NON_WEAPON_CLASSES[lbl], False

        # Default fallback
        return raw_label.replace("_", " ").title(), True

    def _run_single_pass_inference(self, img, offset_x=0, offset_y=0, conf=0.35, imgsz=960):
        cands = []
        mode = self.detect_mode

        # 1. Custom Fine-Tuned Model (best.pt) - Active in 'weapons_only' and 'all' modes
        if mode in ("weapons_only", "all") and self.custom_model:
            try:
                results = self.custom_model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    pose_kpts = []
                    if self.pose_model:
                        try:
                            p_res = self.pose_model(img, conf=0.30, verbose=False)[0]
                            if p_res.keypoints is not None:
                                for kpts in p_res.keypoints:
                                    kd = kpts.data[0].cpu().numpy()
                                    # Collect wrists (indices 9 and 10)
                                    for kidx in [9, 10]:
                                        if len(kd) > kidx and kd[kidx][2] > 0.3:
                                            pose_kpts.append((kd[kidx][0] + offset_x, kd[kidx][1] + offset_y))
                        except Exception:
                            pass

                    for box, score, cls in zip(
                        results.boxes.xyxy.cpu().tolist(),
                        results.boxes.conf.cpu().tolist(),
                        results.boxes.cls.cpu().tolist()
                    ):
                        x1, y1, x2, y2 = box
                        abs_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
                        raw_label = results.names[int(cls)]
                        norm_label, is_weapon = self._classify_detection(raw_label)
                        
                        if not is_weapon:
                            continue  # Prevent weapon model non-weapon classes from polluting the person tracker
                            
                        is_brandished = False
                        if is_weapon and pose_kpts:
                            margin = 60
                            bx1, by1, bx2, by2 = abs_box
                            for px, py in pose_kpts:
                                if bx1 - margin <= px <= bx2 + margin and by1 - margin <= py <= by2 + margin:
                                    is_brandished = True
                                    break
                                    
                        cands.append((norm_label, score, abs_box, is_weapon, is_brandished))
            except Exception as e:
                pass
        
        return cands

    def _run_person_inference(self, img, imgsz=960):
        cands = []
        mode = self.detect_mode
        if mode not in ("faces_only", "persons_only", "all"):
            return cands
            
        face_cands = []
        if self.face_recognizer and self.face_recognizer.det_model:
            try:
                bboxes, kpss = self.face_recognizer.det_model.detect(img, max_num=10, metric="default")
                if bboxes is not None and len(bboxes) > 0:
                    for idx, bbox in enumerate(bboxes):
                        x1, y1, x2, y2, score = bbox[:5]
                        if score >= 0.25:  # Cleaned high-precision face score threshold
                            abs_box = (x1, y1, x2, y2)
                            kps = kpss[idx] if (kpss is not None and idx < len(kpss)) else None
                            face_lbl, face_score, _ = self.face_recognizer.identify_face(img, (x1, y1, x2, y2), landmark=kps)
                            face_cands.append((face_lbl, max(float(score), face_score), abs_box))
            except Exception as e:
                pass

        person_boxes = []
        if self.person_model:
            try:
                p_results = self.person_model(img, conf=0.55, imgsz=imgsz, verbose=False)[0]
                if p_results.boxes is not None:
                    for box, score, cls in zip(
                        p_results.boxes.xyxy.cpu().tolist(),
                        p_results.boxes.conf.cpu().tolist(),
                        p_results.boxes.cls.cpu().tolist()
                    ):
                        if int(cls) == 0:  # Person class
                            px1, py1, px2, py2 = box
                            ph = py2 - py1
                            hy2 = py1 + ph * 0.60  # Head & Upper-body crop (60% to handle seated people)
                            abs_pbox = (px1, py1, px2, hy2)
                            person_boxes.append((abs_pbox, float(score)))
            except Exception:
                pass

        # Global Optimal Face-to-Person Association (Resolves Proximity & Occlusion Stealing)
        matched_face_indices = set()
        if person_boxes and face_cands:
            pairs = []
            for p_idx, (p_box, p_score) in enumerate(person_boxes):
                px1, py1, px2, py2 = p_box
                p_area = max(1.0, (px2 - px1) * (py2 - py1))
                p_cx = (px1 + px2) / 2.0
                p_w = max(1.0, px2 - px1)

                for f_idx, (f_lbl, f_score, f_box) in enumerate(face_cands):
                    fx1, fy1, fx2, fy2 = f_box
                    ix1, iy1 = max(px1, fx1), max(py1, fy1)
                    ix2, iy2 = min(px2, fx2), min(py2, fy2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                    if inter > 0:
                        f_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                        containment = inter / f_area  # Fraction of face inside person upper-body (0.0 to 1.0)
                        union = p_area + f_area - inter
                        iou = inter / max(1.0, union)
                        f_cx = (fx1 + fx2) / 2.0
                        center_score = max(0.0, 1.0 - (abs(p_cx - f_cx) / p_w))

                        composite_score = 0.6 * containment + 0.2 * iou + 0.2 * center_score
                        if composite_score >= 0.15:
                            pairs.append((composite_score, p_idx, f_idx))

            # Global best-match-first assignment (1-to-1 bijection)
            pairs.sort(key=lambda x: x[0], reverse=True)
            assigned_persons = set()
            assigned_faces = {}

            for score, p_idx, f_idx in pairs:
                if p_idx in assigned_persons or f_idx in assigned_faces.values():
                    continue
                assigned_persons.add(p_idx)
                assigned_faces[p_idx] = f_idx

            for p_idx, (p_box, p_score) in enumerate(person_boxes):
                matched_identity = None
                matched_score = p_score
                face_seen = False

                if p_idx in assigned_faces:
                    f_idx = assigned_faces[p_idx]
                    f_lbl, f_score, _ = face_cands[f_idx]
                    face_seen = True
                    matched_face_indices.add(f_idx)
                    matched_identity = f_lbl
                    matched_score = max(p_score, f_score)

                if face_seen:
                    cands.append((matched_identity, matched_score, p_box, False, False))
                else:
                    cands.append(("Identifying...", p_score, p_box, False, False))

        elif person_boxes:
            # No face candidates detected on frame -> pass raw person detections as Identifying...
            for p_box, p_score in person_boxes:
                cands.append(("Identifying...", p_score, p_box, False, False))

        for f_idx, (f_lbl, f_score, f_box) in enumerate(face_cands):
            if f_idx not in matched_face_indices and f_lbl.startswith("Person: "):
                cands.append((f_lbl, f_score, f_box, False, False))

        return cands

    def _capture_loop(self):
        """Thread 1: Dedicated 30-60 FPS Video Stream Capture & Renderer."""
        last_time = time.time()
        fps_smoothing = 0.95
        read_fail_count = 0

        while self.running:
            with self.lock:
                needs_reconnect = self._needs_reconnect
                if needs_reconnect:
                    self._needs_reconnect = False
                    current_source = self._pending_source
                    source_name = self._pending_source_str
                    self.source = current_source
                    self.source_str = source_name
                else:
                    current_source = self.source
                    source_name = self.source_str

                use_cctv = self.cctv_mode
                use_clahe = self.clahe_enhance
                use_sharpness = self.sharpness_boost
                use_tiling = self.tile_inference
                inference_size = self.imgsz
                current_mode = self.detect_mode
                current_dets = list(self.active_detections)
                is_confirmed = self.is_confirmed_alert

            # Safely release previous VideoCapture strictly on the capture thread
            if needs_reconnect and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(0.05)

            if self.cap is None or not self.cap.isOpened():
                print(f"[INFO] Opening stream source: {source_name}")
                if isinstance(current_source, str) and current_source.startswith("rtsp://"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
                    cap = cv2.VideoCapture(current_source, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        print("[INFO] RTSP TCP failed, trying UDP transport...")
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;5000000"
                        cap = cv2.VideoCapture(current_source, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(current_source)
                elif isinstance(current_source, int) or (isinstance(current_source, str) and current_source.isdigit()):
                    idx = int(current_source)
                    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                    if not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(idx)
                    if not cap.isOpened() and idx == 0:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
                        if not cap.isOpened():
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = cv2.VideoCapture(1)
                else:
                    cap = cv2.VideoCapture(current_source)

                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    with self.lock:
                        self.connected = False
                        if isinstance(current_source, str) and current_source.startswith("rtsp://"):
                            err_msg = f"RTSP Unreachable: {source_name}\nCheck Camera IP (554), Credentials & Local Network"
                        elif isinstance(current_source, int) or (isinstance(current_source, str) and current_source.isdigit()):
                            err_msg = f"Camera Access Error (Src: {source_name})\nGrant Terminal Camera Permission in macOS Settings"
                        else:
                            err_msg = f"File Not Found or Unreadable:\n{source_name}"
                        self.status_message = f"Failed to connect to {source_name}"
                        self.latest_jpeg = self._create_placeholder_jpeg(err_msg)
                    time.sleep(1.0)
                    continue
                else:
                    self.cap = cap
                    read_fail_count = 0
                    with self.lock:
                        self.connected = True
                        self.status_message = "Connected"

            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None or frame.size == 0:
                read_fail_count += 1
                if read_fail_count < 10:
                    time.sleep(0.05)
                    continue
                read_fail_count = 0
                with self.lock:
                    if isinstance(current_source, str) and not current_source.startswith("rtsp://") and Path(current_source).exists():
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        continue
                    else:
                        self.connected = False
                        self.status_message = "Stream interrupted..."
                        if self.cap:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                time.sleep(1.0)
                continue

            read_fail_count = 0

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                self.fps = self.fps * fps_smoothing + (1.0 / dt) * (1.0 - fps_smoothing)

            self.frame_count += 1
            height, width = frame.shape[:2]

            with self.lock:
                self.latest_raw_frame = frame.copy()

            # Render active detection boxes with zero-latency 60+ FPS smooth room-wide tracking & identity lock
            smoothed_dets = self.tracker.step_frame(frame) if hasattr(self, 'tracker') else None
            display_dets = smoothed_dets if smoothed_dets else current_dets

            for det in display_dets:
                x1, y1, x2, y2 = det["box"]
                is_weapon = det.get("is_weapon", False)
                is_cand_confirmed = det.get("confirmed", False)
                raw_lbl = det.get("label", "")

                if raw_lbl == "Identifying...":
                    continue

                # Clean display name: strip 'Person: ' prefix and '#track_id'
                if raw_lbl.startswith("Person: "):
                    clean_lbl = raw_lbl.replace("Person: ", "").strip()
                    color = (0, 230, 118)  # Bright Green box for Identified Person
                elif raw_lbl in ("Unknown", "Identifying..."):
                    clean_lbl = raw_lbl
                    color = (255, 200, 0)  # Amber/Yellow box for Unconfirmed
                elif is_weapon:
                    clean_lbl = raw_lbl
                    color = (0, 0, 255) if is_cand_confirmed else (0, 140, 255)
                else:
                    clean_lbl = raw_lbl
                    color = (255, 200, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Only draw floating labels on the bounding box for weapons or unconfirmed tracks
                if is_weapon or not raw_lbl.startswith("Person: "):
                    text_str = f"{clean_lbl} {det['confidence']:.2f}"
                    (w_text, h_text), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + w_text + 6, max(22, y1)), color, -1)
                    cv2.putText(
                        frame,
                        text_str,
                        (x1 + 3, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255) if is_weapon else (0, 0, 0),
                        2
                    )

            # Face privacy blur option
            if self.faces or self.blur_faces:
                self._detect_and_blur_faces(frame)

            # Render Red Confirmation Alert Banner overlay
            if is_confirmed and current_mode not in ("faces_only", "persons_only"):
                any_brandished = any(det.get("is_brandished", False) for det in display_dets if det.get("confirmed", False))
                
                if any_brandished:
                    cv2.rectangle(frame, (0, 0), (width, 38), (0, 0, 180), -1)
                    banner_text = "HIGH THREAT: BRANDISHED WEAPON DETECTED"
                else:
                    cv2.rectangle(frame, (0, 0), (width, 38), (0, 140, 255), -1)
                    banner_text = "LOW THREAT: UNATTENDED WEAPON DETECTED"

                cv2.putText(
                    frame,
                    banner_text,
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # High-tech FPS & Mode Status Badge
            mode_tag = "TARGET: WEAPONS ONLY" if current_mode == "weapons_only" else ("TARGET: FACES ONLY" if current_mode in ("faces_only", "persons_only") else "TARGET: ALL")
            mode_desc = f"FAST ({inference_size}p) | {mode_tag}"
            if use_tiling: mode_desc += " + TILE-4X"
            if use_sharpness: mode_desc += " + SHARP"
            if use_clahe: mode_desc += " + CLAHE"
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f} | Mode: {mode_desc} | Src: {source_name}",
                (12, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 128),
                1,
                cv2.LINE_AA
            )

            # Encode frame to JPEG
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

            time.sleep(0.001)

        # Clean teardown strictly within the capture thread
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _inference_loop(self):
        """Thread 2: Asynchronous Background YOLO Inference Engine."""
        while self.running:
            with self.lock:
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()
                conf_thresh = self.confidence
                inference_size = self.imgsz
                use_sharpness = self.sharpness_boost
                use_clahe = self.clahe_enhance
                use_tiling = self.tile_inference
                current_filter = self.weapon_filter
                current_mode = self.detect_mode

            if frame is None or not self.connected:
                time.sleep(0.05)
                continue

            # Preprocessing Pipeline
            proc_frame = frame
            if use_sharpness:
                proc_frame = apply_sharpening_filter(proc_frame)
            if use_clahe:
                proc_frame = apply_clahe_enhancement(proc_frame)

            height, width = proc_frame.shape[:2]
            raw_detections = []

            # Execute YOLO inference (Multi-Tile for CCTV weapons or Fast Single-Pass for Faces)
            if use_tiling and current_mode != "faces_only" and width >= 480 and height >= 360:
                pad = 40
                mid_x, mid_y = width // 2, height // 2
                slices = [
                    (proc_frame, 0, 0),
                    (proc_frame[0:mid_y + pad, 0:mid_x + pad], 0, 0),
                    (proc_frame[0:mid_y + pad, max(0, mid_x - pad):width], max(0, mid_x - pad), 0),
                    (proc_frame[max(0, mid_y - pad):height, 0:mid_x + pad], 0, max(0, mid_y - pad)),
                    (proc_frame[max(0, mid_y - pad):height, max(0, mid_x - pad):width], max(0, mid_x - pad), max(0, mid_y - pad))
                ]
                for crop_img, off_x, off_y in slices:
                    cands = self._run_single_pass_inference(crop_img, offset_x=off_x, offset_y=off_y, conf=conf_thresh, imgsz=inference_size)
                    raw_detections.extend(cands)
            else:
                raw_detections = self._run_single_pass_inference(proc_frame, offset_x=0, offset_y=0, conf=conf_thresh, imgsz=inference_size)

            # --- RUN PERSON/FACE INFERENCE EXACTLY ONCE ON FULL FRAME ---
            person_detections = self._run_person_inference(proc_frame, imgsz=inference_size)
            raw_detections.extend(person_detections)
            # ------------------------------------------------------------

            weapon_candidates = []
            non_weapon_candidates = []

            for norm_label, score, box, is_weapon, is_brandished in raw_detections:
                cand = Candidate(norm_label, score, box, is_brandished)
                if is_weapon:
                    # Enforce minimum 0.45 confidence threshold for weapons to prune low-conf tile artifacts
                    if score < max(conf_thresh, 0.45):
                        continue
                    if current_filter == "firearms_only" and "Firearm" not in norm_label and "Explosive" not in norm_label:
                        continue
                    weapon_candidates.append(cand)
                else:
                    if current_mode != "weapons_only":
                        non_weapon_candidates.append(cand)

            # Apply strict Non-Maximum Suppression to prevent duplicate face/weapon boxes
            merged_non_weapon_candidates = non_max_suppression_candidates(non_weapon_candidates, overlap_thresh=0.20)
            merged_weapon_candidates = non_max_suppression_candidates(weapon_candidates, overlap_thresh=0.30)

            # Suppress weapon detections if they heavily overlap with a face/person box
            filtered_weapon_candidates = []
            for w_cand in merged_weapon_candidates:
                wb = w_cand.box
                is_spurious = False
                for nw_cand in merged_non_weapon_candidates:
                    fb = nw_cand.box
                    # Compute IoU between weapon box and face/person box
                    xA, yA = max(wb[0], fb[0]), max(wb[1], fb[1])
                    xB, yB = min(wb[2], fb[2]), min(wb[3], fb[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    w_area = (wb[2] - wb[0]) * (wb[3] - wb[1])
                    # If more than 25% of the weapon box is inside a face box, it's a hallucination
                    if inter / float(w_area + 1e-6) > 0.25:
                        is_spurious = True
                        break
                if not is_spurious:
                    filtered_weapon_candidates.append(w_cand)

            merged_weapon_candidates = filtered_weapon_candidates

            det_list = []
            confirmed = False

            # Non-weapons (Face / Identified Persons)
            if current_mode in ("faces_only", "persons_only", "all"):
                for candidate in merged_non_weapon_candidates:
                    x1, y1, x2, y2 = map(int, candidate.box)
                    
                    # Strict validation for unconfirmed person detections to prevent false tracks
                    # (e.g. equipment, tables) from entering the tracker.
                    is_valid = True
                    if candidate.label in ("Identifying...", "Unknown", "Face", "Person"):
                        w = x2 - x1
                        h = max(1, y2 - y1)
                        # Reject if box is excessively wide (not person-like) or too small
                        if w > h * 1.5 or w < 30 or h < 30:
                            is_valid = False
                            
                    if not is_valid:
                        continue
                        
                    det_list.append({
                        "label": candidate.label,
                        "confidence": round(candidate.confidence, 3),
                        "box": [x1, y1, x2, y2],
                        "confirmed": False,
                        "is_weapon": False
                    })

            # Weapons - Included in 'weapons_only' and 'all' modes
            if current_mode in ("weapons_only", "all"):
                for candidate in merged_weapon_candidates:
                    x1, y1, x2, y2 = map(int, candidate.box)
                    is_cand_confirmed = self.gate.confirmed(candidate)
                    confirmed = is_cand_confirmed or confirmed
                    
                    display_label = candidate.label
                    if candidate.is_brandished:
                        display_label = f"Brandished {candidate.label}"
                    else:
                        display_label = f"Unattended {candidate.label}"

                    det_list.append({
                        "label": display_label,
                        "confidence": round(candidate.confidence, 3),
                        "box": [x1, y1, x2, y2],
                        "confirmed": is_cand_confirmed,
                        "is_weapon": True,
                        "is_brandished": candidate.is_brandished
                    })

            self.active_detections = det_list
            
            # Alert Confirmation Logic
            if confirmed and not self._alert_open and current_mode not in ("faces_only", "persons_only") and merged_weapon_candidates:
                top = max(merged_weapon_candidates, key=lambda c: c.confidence)
                timestamp_sec = round(self.frame_count / (self.fps or 30.0), 2)
                alert_item = {
                    "time_seconds": timestamp_sec,
                    "frame": self.frame_count,
                    "label": top.label,
                    "confidence": round(top.confidence, 3),
                    "status": "human_review_required",
                    "timestamp": time.strftime("%H:%M:%S")
                }
                self.alert_history.appendleft(alert_item)
                
                try:
                    with self.alert_csv_path.open("a", newline="") as output:
                        log = csv.DictWriter(output, fieldnames=["time_seconds", "frame", "label", "confidence", "status"])
                        log.writerow({
                            "time_seconds": alert_item["time_seconds"],
                            "frame": alert_item["frame"],
                            "label": alert_item["label"],
                            "confidence": alert_item["confidence"],
                            "status": alert_item["status"]
                        })
                except Exception as e:
                    print(f"[ERROR] Failed to write alert log: {e}")

            with self.lock:
                self._alert_open = confirmed
                self.is_confirmed_alert = confirmed
                self.active_detections = det_list
                if hasattr(self, 'tracker'):
                    self.tracker.update_detections(det_list, frame=frame)
                if hasattr(self, 'klt_tracker') and frame is not None:
                    self.klt_tracker.update_heavy_detections(frame, det_list)

            time.sleep(0.01)

    def _create_placeholder_jpeg(self, text):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (620, 460), (30, 30, 40), 2)
        lines = text.split("\n")
        y = 210
        for line in lines:
            cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
            y += 35
        ok, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes() if ok else b""

    def generate_mjpeg_stream(self):
        while self.running:
            with self.lock:
                frame_bytes = self.latest_jpeg

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.03)

    def enroll_person_burst(self, name, num_samples=30, delay_sec=0.35):
        """Captures 30 multi-angle facial crop photos from live stream over 10.5 seconds to allow ample time to turn head."""
        snapshots = []
        for _ in range(num_samples):
            with self.lock:
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

            if frame is not None:
                h_f, w_f = frame.shape[:2]
                face_boxes = []
                
                # 1. Run YOLO Face Detector directly on frame
                if self.face_detector and hasattr(self.face_detector, "__call__"):
                    try:
                        f_results = self.face_detector(frame, conf=0.15, verbose=False)[0]
                        if f_results.boxes is not None:
                            for b in f_results.boxes.xyxy.cpu().tolist():
                                face_boxes.append(b)
                    except Exception as e:
                        pass
                        
                if face_boxes:
                    # Pick largest face box
                    fx1, fy1, fx2, fy2 = map(int, max(face_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
                    bw, bh = fx2 - fx1, fy2 - fy1
                    pad_w, pad_h = int(bw * 0.25), int(bh * 0.25)
                    cx1, cy1 = max(0, fx1 - pad_w), max(0, fy1 - pad_h)
                    cx2, cy2 = min(w_f, fx2 + pad_w), min(h_f, fy2 + pad_h)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0 and crop.shape[0] >= 20 and crop.shape[1] >= 20:
                        snapshots.append(crop)

            time.sleep(delay_sec)

        if not snapshots:
            return False

        return self.face_recognizer.enroll_person(name, snapshots)

    def get_status(self):
        with self.lock:
            active_dets = self.active_detections
            if hasattr(self, 'tracker') and self.tracker.tracks:
                active_dets = []
                for track in self.tracker.tracks:
                    if track.missed_frames <= 15:
                        active_dets.append({
                            "track_id": track.track_id,
                            "label": track.label,
                            "confidence": round(float(track.confidence), 3),
                            "box": track.get_int_box(),
                            "is_weapon": track.is_weapon
                        })

            return {
                "connected": self.connected,
                "status_message": self.status_message,
                "source": self.source_str,
                "fps": round(self.fps, 1),
                "frame_count": self.frame_count,
                "confidence_threshold": self.confidence,
                "frames_window": self.frames_window,
                "required_matches": self.required_matches,
                "faces_enabled": self.faces,
                "blur_faces": self.blur_faces,
                "cctv_mode": self.cctv_mode,
                "clahe_enhance": self.clahe_enhance,
                "sharpness_boost": self.sharpness_boost,
                "tile_inference": self.tile_inference,
                "imgsz": self.imgsz,
                "detect_mode": self.detect_mode,
                "weapon_filter": self.weapon_filter,
                "active_detections": active_dets,
                "is_confirmed_alert": self.is_confirmed_alert
            }

    def get_alerts(self):
        return list(self.alert_history)
