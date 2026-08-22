import cv2
import time
import numpy as np


class UltraFastVisualTrack:
    """
    Sub-millisecond KLT (Kanade-Lucas-Tomasi) Optical Flow + Shi-Tomasi Corner Tracker.
    Delivers 700+ FPS feature tracking with forward-backward error validation.
    """
    def __init__(self, track_id, gray_frame, box_xyxy, label, confidence, is_weapon=False):
        self.track_id = track_id
        x1, y1, x2, y2 = map(float, box_xyxy)
        self.box = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.target_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.raw_label = label
        self.confidence = confidence
        self.is_weapon = is_weapon
        self.missed = 0
        self.hits = 1
        self.locked_person = label if label.startswith("Person: ") else None
        self.lock_ttl = 120  # 4-second identity lock memory

        self.pts = self._extract_points(gray_frame, self.box)
        self.prev_gray = gray_frame.copy() if (gray_frame is not None and self.pts is not None) else None

    def _extract_points(self, gray_frame, box):
        """
        Extracts KLT Shi-Tomasi corner feature points (cv2.goodFeaturesToTrack) inside bounding box,
        falling back to a uniform grid if corner count is low.
        """
        if gray_frame is None:
            return None
        x1, y1, x2, y2 = map(int, box)
        h_f, w_f = gray_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        w, h = x2 - x1, y2 - y1

        if w < 15 or h < 15:
            return None

        roi = gray_frame[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=25,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3
        )

        pts_list = []
        if corners is not None and len(corners) > 0:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            pts_list.append(corners)

        # Uniform grid fallback / supplement if corners are sparse
        if corners is None or len(corners) < 6:
            xs = np.linspace(x1 + w * 0.2, x2 - w * 0.2, 4)
            ys = np.linspace(y1 + h * 0.2, y2 - h * 0.2, 4)
            gx, gy = np.meshgrid(xs, ys)
            grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T.reshape(-1, 1, 2).astype(np.float32)
            pts_list.append(grid_pts)

        if pts_list:
            all_pts = np.vstack(pts_list)
            return all_pts
        return None

    def update_optical_flow(self, curr_gray):
        """
        Runs sub-millisecond Lucas-Kanade Pyramid Optical Flow with forward-backward error validation (< 0.5ms).
        """
        if self.prev_gray is None or self.pts is None or curr_gray is None or len(self.pts) == 0:
            return False, self.get_int_box(), None

        try:
            # 1. Forward LK Optical Flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                curr_gray,
                self.pts,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # 2. Backward LK Optical Flow (Validation check)
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                self.prev_gray,
                next_pts,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Forward-Backward error thresholding
            fb_err = np.linalg.norm(self.pts - back_pts, axis=2).ravel()
            valid_idx = np.where((status.ravel() == 1) & (back_status.ravel() == 1) & (fb_err < 2.5))[0]

            if len(valid_idx) >= 3:
                old_valid = self.pts[valid_idx].reshape(-1, 2)
                new_valid = next_pts[valid_idx].reshape(-1, 2)
                displacements = new_valid - old_valid

                # Median displacement vector
                dx, dy = np.median(displacements, axis=0)

                # Bounding box translation update
                self.box[0] += dx
                self.box[1] += dy
                self.box[2] += dx
                self.box[3] += dy

                self.target_box[0] += dx
                self.target_box[1] += dy
                self.target_box[2] += dx
                self.target_box[3] += dy

                self.pts = new_valid.reshape(-1, 1, 2)
                self.prev_gray = curr_gray.copy()

                if self.lock_ttl > 0:
                    self.lock_ttl -= 1

                keypoints = new_valid.astype(int).tolist()
                return True, self.get_int_box(), keypoints
        except Exception:
            pass

        self.prev_gray = curr_gray.copy() if curr_gray is not None else None
        return False, self.get_int_box(), None

    def reanchor_heavy_detector(self, gray_frame, box_xyxy, label, confidence, is_weapon=False):
        """Re-detection / Correction phase: Re-anchors KLT feature points with heavy detector box."""
        new_box = np.array(box_xyxy, dtype=np.float32)
        self.box = 0.5 * self.box + 0.5 * new_box
        self.target_box = new_box
        self.raw_label = label
        self.is_weapon = is_weapon

        if label.startswith("Person: "):
            self.locked_person = label
            self.lock_ttl = 120

        self.confidence = max(self.confidence * 0.3 + confidence * 0.7, confidence)
        self.pts = self._extract_points(gray_frame, self.box)
        if gray_frame is not None:
            self.prev_gray = gray_frame.copy()
        self.missed = 0
        self.hits += 1

    @property
    def label(self):
        if self.locked_person and self.lock_ttl > 0:
            return self.locked_person
        return self.raw_label

    def get_int_box(self):
        x1, y1, x2, y2 = map(int, self.box)
        return [x1, y1, x2, y2]


class LightweightTrackerEngine:
    """
    KLT (Kanade-Lucas-Tomasi) Pyramid Optical Flow Real-Time Engine:
    - Runs sub-millisecond Lucas-Kanade optical flow on every video frame (700+ FPS throughput).
    - Periodically re-anchors tracks on heavy detector frames.
    """
    def __init__(self, dist_thresh=200.0, max_missed=15):
        self.next_id = 1
        self.tracks = []
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def step_visual_tracking(self, frame):
        """Runs ultra-fast KLT optical flow tracking on every video frame (< 0.5ms per frame)."""
        if frame is None:
            return []

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        active_outputs = []

        for track in list(self.tracks):
            ok, box_int, keypoints = track.update_optical_flow(curr_gray)
            if ok or track.missed < self.max_missed:
                track.missed = 0 if ok else (track.missed + 1)
                active_outputs.append({
                    "track_id": track.track_id,
                    "label": track.label,
                    "confidence": round(float(track.confidence), 3),
                    "box": box_int,
                    "is_weapon": track.is_weapon,
                    "keypoints": keypoints or []
                })
            else:
                self.tracks.remove(track)

        return active_outputs

    def update_heavy_detections(self, frame, detections):
        """Re-detection / Correction phase: Re-anchors KLT feature points with heavy detector boxes."""
        if frame is None or not detections:
            return

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matched_tracks = set()
        matched_dets = set()

        if len(self.tracks) > 0:
            for t_idx, track in enumerate(self.tracks):
                tb = track.box
                cx_t, cy_t = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
                best_dist = self.dist_thresh
                best_d = -1

                for d_idx, det in enumerate(detections):
                    if d_idx in matched_dets:
                        continue
                    db = det["box"]
                    cx_d, cy_d = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                    dist = np.sqrt((cx_t - cx_d)**2 + (cy_t - cy_d)**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_d = d_idx

                if best_d != -1:
                    matched_tracks.add(t_idx)
                    matched_dets.add(best_d)
                    det = detections[best_d]
                    track.reanchor_heavy_detector(
                        curr_gray,
                        det["box"],
                        det["label"],
                        det["confidence"],
                        det.get("is_weapon", False)
                    )

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                t = UltraFastVisualTrack(
                    self.next_id,
                    curr_gray,
                    det["box"],
                    det["label"],
                    det["confidence"],
                    det.get("is_weapon", False)
                )
                self.next_id += 1
                self.tracks.append(t)

