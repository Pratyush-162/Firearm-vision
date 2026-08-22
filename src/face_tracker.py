import time
import collections
import numpy as np
from scipy.optimize import linear_sum_assignment


class OneEuroFilter:
    """
    Adaptive Low-Pass Filter for smooth, jitter-free bounding box rendering.
    Adapts cutoff frequency based on movement speed:
    - Low cutoff when static -> 0 jitter, 0 box breathing.
    - High cutoff when moving -> 0 lag, instant gliding responsiveness.
    """
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)

        dx = (float(x) - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class TrackState:
    UNCONFIRMED = "unconfirmed"       # Person detected, no confirmed face match yet (displays "Verifying...")
    CONFIRMED = "confirmed"           # Direct face recognition match happened on THIS specific track
    LOCKED_NO_FACE = "locked_no_face" # Was confirmed, face not directly visible right now (e.g. back turned, moving)


class FaceTrack:
    def __init__(self, track_id, box, label, confidence, is_weapon=False):
        self.track_id = track_id
        x1, y1, x2, y2 = box
        self.box = np.array(box, dtype=np.float32)  # [x1, y1, x2, y2]
        self.target_box = np.array(box, dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.raw_label = label
        self.confidence = float(confidence)
        self.is_weapon = is_weapon
        self.last_updated = time.time()
        self.missed_frames = 0
        self.hits = 1
        self.frames_since_face = 0

        # State Machine Initialization
        if label.startswith("Person: ") and self.confidence >= 0.18:
            self.is_locked = True
            self.locked_label = label
            self.state = TrackState.CONFIRMED
        else:
            self.is_locked = False
            self.locked_label = None
            self.state = TrackState.UNCONFIRMED

        # Temporal Label Smoothing
        self.label_history = collections.deque(maxlen=10)
        self.label_history.append(self.label)
        self.history = collections.deque(maxlen=10)

        # 4 One-Euro Filters per track: (cx, cy) for position, (w, h) for box size
        self.f_cx = OneEuroFilter(freq=60, min_cutoff=0.5, beta=0.015)
        self.f_cy = OneEuroFilter(freq=60, min_cutoff=0.5, beta=0.015)
        self.f_w = OneEuroFilter(freq=60, min_cutoff=0.25, beta=0.003)
        self.f_h = OneEuroFilter(freq=60, min_cutoff=0.25, beta=0.003)

    def update(self, new_box, label, confidence, is_weapon=False, **kwargs):
        new_box_arr = np.array(new_box, dtype=np.float32)
        
        # Calculate pure translational velocity (center offset) to prevent geometric explosion
        n_cx = (new_box_arr[0] + new_box_arr[2]) / 2.0
        n_cy = (new_box_arr[1] + new_box_arr[3]) / 2.0
        p_cx = (self.target_box[0] + self.target_box[2]) / 2.0
        p_cy = (self.target_box[1] + self.target_box[3]) / 2.0
        dx = n_cx - p_cx
        dy = n_cy - p_cy
        dist = float(np.hypot(dx, dy))

        if dist > 400:
            self.box = new_box_arr.copy()
            self.velocity = np.zeros(4, dtype=np.float32)
        else:
            trans_disp = np.array([dx, dy, dx, dy], dtype=np.float32)
            self.velocity = 0.6 * self.velocity + 0.4 * trans_disp

        self.target_box = new_box_arr
        self.raw_label = label
        self.missed_frames = 0
        self.hits += 1
        self.last_updated = time.time()
        self.history.append(new_box_arr)

        # Strict Single-Track Identity State Updates
        if label.startswith("Person: ") and float(confidence) >= 0.18:
            self.is_locked = True
            self.locked_label = label
            self.state = TrackState.CONFIRMED
            self.frames_since_face = 0
            self.label_history.append(label)
            self.stationary_frames = 0
        elif self.is_locked:
            # Person was previously confirmed, but face is not visible this frame (back turned / facing away)
            self.state = TrackState.LOCKED_NO_FACE
            self.frames_since_face += 1
            self.label_history.append(self.locked_label)

            # Prevent background hallucinations from stealing and holding an identity forever
            if len(self.history) >= 2:
                # Check how far the center has moved since the last frame
                prev_box = self.history[-2]
                curr_box = self.history[-1]
                p_cx, p_cy = (prev_box[0]+prev_box[2])/2, (prev_box[1]+prev_box[3])/2
                c_cx, c_cy = (curr_box[0]+curr_box[2])/2, (curr_box[1]+curr_box[3])/2
                if np.hypot(c_cx - p_cx, c_cy - p_cy) < 6.0:
                    self.stationary_frames = getattr(self, 'stationary_frames', 0) + 1
                else:
                    self.stationary_frames = 0
                
                # If a locked track without a face sits perfectly still for 90 frames (~3 seconds), drop the lock
                if self.stationary_frames > 90:
                    self.is_locked = False
                    self.locked_label = None
                    self.raw_label = "Unknown"
                    self.state = TrackState.UNCONFIRMED
        else:
            self.state = TrackState.UNCONFIRMED
            self.frames_since_face += 1
            self.label_history.append(label)

        self.confidence = max(self.confidence * 0.3 + float(confidence) * 0.7, float(confidence))
        self.is_weapon = is_weapon

    def predict_step(self, alpha=0.35, **kwargs):
        """Applies smooth pursuit interpolation to prevent step-function snapping in the OneEuroFilter."""
        if self.missed_frames > 0:
            # Kinematic velocity prediction: coast along momentum vector during temporary occlusion or fast blur
            self.target_box += self.velocity
            self.velocity *= 0.85  # Friction decay to prevent flying off screen

        self.box += alpha * (self.target_box - self.box)

    @property
    def label(self):
        if self.is_weapon:
            return self.raw_label
        if self.state in (TrackState.CONFIRMED, TrackState.LOCKED_NO_FACE) and self.locked_label:
            return self.locked_label
        return self.raw_label

    def get_int_box(self):
        x1, y1, x2, y2 = self.box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))

        # Filter position (cx, cy) and size (w, h) separately using One-Euro Filter
        fcx = self.f_cx.filter(cx)
        fcy = self.f_cy.filter(cy)
        fw = self.f_w.filter(w)
        fh = self.f_h.filter(h)

        fx1 = int(fcx - fw / 2.0)
        fy1 = int(fcy - fh / 2.0)
        fx2 = int(fcx + fw / 2.0)
        fy2 = int(fcy + fh / 2.0)
        
        # Strict geometry validation to prevent ghost boxes
        if fx1 >= fx2 or fy1 >= fy2 or fw > 1920 or fh > 1080:
            return None
            
        return [fx1, fy1, fx2, fy2]


class SmoothObjectTracker:
    def __init__(self, max_missed_frames=15, iou_threshold=0.10):
        self.next_id = 1
        self.tracks = []
        self.max_missed = max_missed_frames
        self.iou_threshold = iou_threshold

    def _compute_match_score(self, track, det):
        boxA = track.target_box + track.velocity
        boxB = det["box"]
        det_lbl = det.get("label", "")

        # Strict Identity Isolation: Never match two different recognized persons to each other
        if track.is_locked and track.locked_label and det_lbl.startswith("Person: "):
            if det_lbl != track.locked_label:
                return 0.0

        # Identity Match Override: If the exact same face is recognized, it's the same person
        # regardless of how far they teleported during a stream lag.
        if track.is_locked and track.locked_label and det_lbl == track.locked_label:
            return 1.0

        # 1. IoU score
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        # 2. Centroid distance score
        cxA, cyA = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
        cxB, cyB = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
        dist = np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

        # Relax distance threshold significantly (from 350 to 900) to allow for camera lag teleportation
        if dist > 900 and iou < 0.05:
            return 0.0

        # Anti-Identity-Stealing: If the track belongs to a known person, but the new detection 
        # is "Unknown" (face not matching) or "Identifying..." (no face seen), require a tighter spatial match.
        # Relaxed from 200 to 500 to allow coasting during fast runs.
        if track.is_locked and det_lbl in ("Unknown", "Identifying..."):
            if dist > 500 and iou < 0.10:
                return 0.0

        # Scale distance score up to the new max
        dist_score = max(0.0, 1.0 - (dist / 900.0))
        base_score = 0.5 * iou + 0.5 * dist_score

        return base_score

    def _enforce_unique_identities(self):
        """
        Global Uniqueness Safety Net: Guarantees that an enrolled person identity (e.g. 'Person: Shravan')
        can NEVER appear on more than one track simultaneously under any circumstances.
        """
        seen_identities = {}
        for track in self.tracks:
            if track.is_locked and track.locked_label and track.locked_label.startswith("Person: "):
                lbl = track.locked_label
                if lbl in seen_identities:
                    other = seen_identities[lbl]
                    # Keep whichever track has fresher direct face confirmation (or higher hits)
                    if track.frames_since_face > other.frames_since_face:
                        loser, winner = track, other
                    elif track.frames_since_face < other.frames_since_face:
                        loser, winner = other, track
                    else:
                        loser, winner = (track, other) if track.hits < other.hits else (other, track)

                    loser.is_locked = False
                    loser.locked_label = None
                    loser.raw_label = "Unknown"
                    loser.state = TrackState.UNCONFIRMED
                    seen_identities[lbl] = winner
                else:
                    seen_identities[lbl] = track

    def _deduplicate_tracks(self):
        """Merges overlapping active tracks for the SAME person and isolates distinct identities."""
        if len(self.tracks) <= 1:
            return

        self.tracks.sort(key=lambda t: (t.is_locked, t.confidence, t.hits), reverse=True)
        surviving_tracks = []

        for track in self.tracks:
            boxA = track.get_int_box()
            if boxA is None:
                continue
            
            keep = True
            for existing in surviving_tracks:
                # NEVER merge two different locked individuals even when standing close together!
                if track.is_locked and existing.is_locked and track.locked_label != existing.locked_label:
                    continue

                boxB = existing.get_int_box()
                if boxB is None:
                    continue
                
                xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
                areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
                union = areaA + areaB - inter
                iou = inter / float(max(1, union))
                iomin = inter / float(min(areaA, areaB))

                cxA, cyA = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
                cxB, cyB = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
                dist = np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

                # Merge duplicate tracks on the same physical person:
                # Require stricter overlap to merge two locked tracks, but allow easier merging if one is unconfirmed/unknown
                if track.is_locked and existing.is_locked:
                    required_iou, required_iomin = 0.65, 0.70
                else:
                    required_iou, required_iomin = 0.35, 0.45
                
                if iou > required_iou or iomin > required_iomin or (iomin > 0.25 and dist < 60.0):
                    keep = False
                    if track.is_locked and not existing.is_locked:
                        existing.is_locked = True
                        existing.locked_label = track.locked_label
                        existing.state = track.state
                        existing.frames_since_face = track.frames_since_face
                        existing.confidence = max(existing.confidence, track.confidence)
                    break
            if keep:
                surviving_tracks.append(track)

        self.tracks = surviving_tracks
        self._enforce_unique_identities()

    def update_detections(self, detections, **kwargs):
        """
        Updates tracks with fresh asynchronous YOLO/Face detections using true Scipy Hungarian matching.
        """
        matched_track_indices = set()
        matched_det_indices = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.ones((len(self.tracks), len(detections)), dtype=np.float32) * 1e5
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    score = self._compute_match_score(track, det)
                    if score >= self.iou_threshold:
                        cost_matrix[t_idx, d_idx] = 1.0 - score

            # 1. Optimal Global Hungarian Assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    det = detections[c]
                    self.tracks[r].update(
                        det["box"], det["label"], det["confidence"], det.get("is_weapon", False)
                    )

        # 2. Increment missed_frames on existing unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.missed_frames += 1

        # 3. Add brand new detections with fresh track IDs
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_t = FaceTrack(
                    self.next_id,
                    det["box"],
                    det["label"],
                    det["confidence"],
                    det.get("is_weapon", False)
                )
                self.next_id += 1
                self.tracks.append(new_t)

        surviving = []
        for t in self.tracks:
            # Purge tracks promptly when person is physically gone from the scene
            if t.missed_frames <= self.max_missed:
                surviving.append(t)

        self.tracks = surviving
        self._deduplicate_tracks()
        self._enforce_unique_identities()

    def step_frame(self, frame=None):
        self._deduplicate_tracks()

        active_outputs = []
        for track in self.tracks:
            # Only render tracks that currently have active physical detections in frame (allow more bridge frames to prevent flickering)
            if track.missed_frames <= 15:
                track.predict_step()
                box = track.get_int_box()
                
                if box is not None:
                    # CONFIRMATION STAGE: Require 3 valid associations across frames before displaying
                    # a non-weapon track to prevent single-frame false detections from persisting.
                    if not track.is_weapon and track.hits < 3:
                        continue
                        
                    active_outputs.append({
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(float(track.confidence), 3),
                        "box": box,
                        "is_weapon": track.is_weapon
                    })
                else:
                    track.missed_frames = 9999 # Force kill corrupted track
                    
        return active_outputs
