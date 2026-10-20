"""Hand-landmark features and gesture classifiers (no OpenCV or MediaPipe imports)."""
from collections import Counter, deque
from pathlib import Path
import numpy as np

LABELS = ["STOP", "FORWARD", "LEFT", "RIGHT", "BACKWARD"]
FEATURE_VERSION = 1
WRIST, MIDDLE_MCP = 0, 9
# A curled fingertip ends up level with its knuckle or folded into the palm (up to ~1.1x the
# knuckle's distance from the wrist); a straight finger reaches ~1.6-2x.
CURL_RATIO = 1.3


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """MediaPipe hand landmarks -> (21, 3) array of normalized image x, y, z."""
    return np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=float)


def normalize_landmarks(points) -> np.ndarray:
    """63-value feature vector that ignores where the hand is and how big it looks.

    Coordinates are taken relative to the wrist and divided by the wrist to
    middle-finger-MCP distance. Rotation is kept on purpose: the pointing
    direction is what separates FORWARD, LEFT, RIGHT and BACKWARD.
    """
    p = np.asarray(points, dtype=float).reshape(21, 3)
    relative = p - p[WRIST]
    scale = np.linalg.norm(relative[MIDDLE_MCP, :2])
    return (relative / max(scale, 1e-6)).reshape(-1)


def classify_rules(points) -> str:
    """Classify a single index-finger pointing direction.

    The camera image is mirrored before MediaPipe sees it, so horizontal
    directions match the user's view: point left for LEFT and right for RIGHT.
    Anything that is not a clear point (open palm, fist) is STOP.

    A finger counts as curled when its tip is not much farther from the wrist than its
    knuckle. That holds however the hand is turned, and whether the fingertips rest on the
    knuckles (back of the hand to the camera) or deep in the palm (palm to the camera).
    """
    p = np.asarray(points, dtype=float).reshape(21, 3)[:, :2]

    def distance(a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    # The index tip must reach past the middle joint along the finger. Being far from the
    # knuckle is not enough: in a tight fist the folded tip is far from it too, back in the palm.
    knuckle_to_joint = p[6] - p[5]
    index_extended = float(np.dot(p[8] - p[5], knuckle_to_joint)) > 1.15 * float(np.dot(knuckle_to_joint,
                                                                                        knuckle_to_joint))
    other_fingers_curled = all(
        distance(tip, WRIST) < distance(mcp, WRIST) * CURL_RATIO
        for tip, mcp in ((12, 9), (16, 13), (20, 17))
    )
    if not index_extended or not other_fingers_curled:
        return "STOP"

    dx, dy = p[8] - p[5]
    if abs(dx) > abs(dy) * 1.2:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy < 0:
        return "FORWARD"
    return "BACKWARD"


class GestureFilter:
    """Majority vote over the last few per-frame gestures.

    A gesture becomes active once it fills `needed` of the last `window` frames, and stays
    active until another one does, so a few misread or hand-lost frames neither start nor
    stop a command. `needed` must be a strict majority of `window`.
    """

    def __init__(self, window=7, needed=4, initial="NONE"):
        if not window // 2 < needed <= window:
            raise ValueError("needed must be more than half of window and at most window")
        self.recent = deque(maxlen=window)
        self.needed = needed
        self.active = initial

    def update(self, observed: str) -> str:
        self.recent.append(observed)
        label, count = Counter(self.recent).most_common(1)[0]
        if count >= self.needed:
            self.active = label
        return self.active


def load_model(path):
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"{path} uses an older feature format; retrain it with gesture_app/train_classifier.py")
    return bundle["model"]


class GestureRecognizer:
    """Rule-based or Random Forest gesture classification of one hand.

    mode is "rules", "rf", or "auto" (Random Forest when the model file exists).
    Random Forest predictions below min_confidence become STOP.
    """

    def __init__(self, mode="auto", model_path=None, min_confidence=0.6):
        if mode not in ("auto", "rules", "rf"):
            raise ValueError("mode must be auto, rules or rf")
        self.model = None
        self.min_confidence = min_confidence
        if mode != "rules" and model_path and Path(model_path).exists():
            self.model = load_model(model_path)
        elif mode == "rf":
            raise FileNotFoundError(f"{model_path} not found; run gesture_app/train_classifier.py first")
        self.name = "random forest" if self.model is not None else "rules"

    def predict(self, points) -> tuple[str, float]:
        if self.model is None:
            return classify_rules(points), 1.0
        proba = self.model.predict_proba(normalize_landmarks(points)[None, :])[0]
        best = int(np.argmax(proba))
        label = str(self.model.classes_[best])
        return (label if proba[best] >= self.min_confidence else "STOP"), float(proba[best])
