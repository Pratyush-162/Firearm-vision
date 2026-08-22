from collections import deque
from dataclasses import dataclass


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union else 0.0


@dataclass
class Candidate:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    is_brandished: bool = False


class TemporalConfirm:
    """Suppress one-frame flashes. It does not determine threat, possession, or intent."""
    def __init__(self, frames=5, required=3, match_iou=.20):
        self.history = deque(maxlen=frames); self.required = required; self.match_iou = match_iou

    def confirmed(self, candidate):
        self.history.append(candidate)
        matches = [old for old in self.history if old and old.label == candidate.label and iou(old.box, candidate.box) >= self.match_iou]
        return len(matches) >= self.required
