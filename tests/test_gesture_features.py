import numpy as np
import pytest
from gesture_app.features import GestureFilter, GestureRecognizer, classify_rules, normalize_landmarks

UP, DOWN, LEFT, RIGHT = (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)  # image coordinates, y down


def synthetic_hand(point=None, palm=UP, extended=(), wrist=(0.5, 0.8), size=1.0, noise=0.0, rng=None,
                   curl_tip=0.02):
    """21 MediaPipe-style landmarks. point: direction of an extended index finger (others curled);
    extended: fingers 1-4 (index..pinky) stretched along the palm direction, e.g. an open palm.
    curl_tip: where a curled fingertip lands along the palm direction, relative to its knuckle;
    negative values fold it back into the palm, as in a tight fist seen palm-first."""
    u, p = np.array(palm), np.array([-palm[1], palm[0]])
    w = np.array(wrist)
    pts = np.zeros((21, 3))
    pts[0, :2] = w
    for k in range(1, 5):  # thumb, roughly beside the palm
        pts[k, :2] = w + size * (0.05 * k * u + (0.06 + 0.02 * k) * p)
    for finger, (base, offset) in enumerate(((5, -0.06), (9, -0.02), (13, 0.02), (17, 0.06)), start=1):
        mcp = w + size * (0.2 * u + offset * p)
        if finger == 1 and point is not None:
            direction, lengths = np.array(point), (0.06, 0.10, 0.13)
        elif finger in extended:
            direction, lengths = u, (0.06, 0.10, 0.13)
        else:
            direction, lengths = u, (0.04, (0.04 + curl_tip) / 2, curl_tip)
        pts[base, :2] = mcp
        for j, length in enumerate(lengths, start=1):
            pts[base + j, :2] = mcp + size * length * direction
    if noise:
        pts[:, :2] += rng.normal(0, noise, (21, 2))
    return pts


@pytest.mark.parametrize("direction,label", [(UP, "FORWARD"), (DOWN, "BACKWARD"),
                                             (LEFT, "LEFT"), (RIGHT, "RIGHT")])
def test_rules_read_pointing_direction(direction, label):
    assert classify_rules(synthetic_hand(point=direction)) == label


@pytest.mark.parametrize("curl_tip", [0.02, -0.04, -0.08])
@pytest.mark.parametrize("palm,direction,label", [(UP, UP, "FORWARD"), (UP, LEFT, "LEFT"), (UP, RIGHT, "RIGHT"),
                                                  (DOWN, DOWN, "BACKWARD"), (LEFT, LEFT, "LEFT"),
                                                  (RIGHT, UP, "FORWARD")])
def test_rules_read_pointing_with_any_fist(palm, direction, label, curl_tip):
    """Curled fingertips may land anywhere between the knuckles and deep in the palm, and the
    hand may be rotated; the pointing direction must still be read."""
    hand = synthetic_hand(point=direction, palm=palm, curl_tip=curl_tip)
    assert classify_rules(hand) == label


def test_rules_stop_for_open_palm_and_fist():
    assert classify_rules(synthetic_hand(extended=(1, 2, 3, 4))) == "STOP"
    assert classify_rules(synthetic_hand(extended=(1, 2, 3, 4), palm=LEFT)) == "STOP"
    assert classify_rules(synthetic_hand()) == "STOP"
    assert classify_rules(synthetic_hand(curl_tip=-0.08)) == "STOP"
    # Pointing with the middle finger also out is not a clear single point.
    assert classify_rules(synthetic_hand(point=UP, extended=(2,))) == "STOP"


def test_filter_ignores_brief_misreads():
    gestures = GestureFilter(window=7, needed=4)
    assert gestures.active == "NONE"
    # A flickering classifier still engages once FORWARD dominates the window ...
    readings = ["FORWARD", "STOP", "FORWARD", "FORWARD", "NONE", "FORWARD"]
    assert [gestures.update(r) for r in readings][-1] == "FORWARD"
    # ... and isolated STOP / NONE frames do not drop it.
    for reading in ["STOP", "FORWARD", "NONE", "STOP", "FORWARD", "FORWARD", "STOP", "FORWARD"]:
        assert gestures.update(reading) == "FORWARD"
    # From a clean window, a sustained STOP takes over after `needed` frames.
    for _ in range(7):
        gestures.update("FORWARD")
    assert [gestures.update("STOP") for _ in range(4)] == ["FORWARD", "FORWARD", "FORWARD", "STOP"]


def test_filter_rejects_ambiguous_settings():
    with pytest.raises(ValueError):
        GestureFilter(window=6, needed=3)  # two labels could tie for the majority
    with pytest.raises(ValueError):
        GestureFilter(window=4, needed=5)


def test_features_ignore_position_and_scale():
    a = normalize_landmarks(synthetic_hand(point=LEFT))
    b = normalize_landmarks(synthetic_hand(point=LEFT, wrist=(0.2, 0.6), size=1.7))
    assert np.allclose(a, b)
    assert not np.allclose(a, normalize_landmarks(synthetic_hand(point=RIGHT)))


def test_recognizer_modes(tmp_path):
    hand = synthetic_hand(point=RIGHT)
    assert GestureRecognizer("rules").predict(hand) == ("RIGHT", 1.0)
    assert GestureRecognizer("auto", tmp_path / "missing.joblib").name == "rules"
    with pytest.raises(FileNotFoundError):
        GestureRecognizer("rf", tmp_path / "missing.joblib")


def test_random_forest_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    import joblib
    from gesture_app import train_classifier

    rng = np.random.default_rng(0)
    shapes = {"FORWARD": {"point": UP}, "BACKWARD": {"point": DOWN}, "LEFT": {"point": LEFT},
              "RIGHT": {"point": RIGHT}, "STOP": {"extended": (1, 2, 3, 4)}}
    raw, labels = [], []
    for label, kwargs in shapes.items():
        for _ in range(40):
            hand = synthetic_hand(**kwargs, wrist=rng.uniform(0.3, 0.7, 2), size=rng.uniform(0.7, 1.3),
                                  noise=0.004, rng=rng)
            raw.append(hand.reshape(-1))
            labels.append(label)
    model, features, test = train_classifier.train(np.array(raw), np.array(labels))
    assert (model.predict(features[test]) == np.array(labels)[test]).mean() > 0.9

    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "feature_version": 1, "labels": sorted(shapes)}, path)
    recognizer = GestureRecognizer("auto", path, min_confidence=0.0)
    assert recognizer.name == "random forest"
    assert recognizer.predict(synthetic_hand(point=LEFT))[0] == "LEFT"
