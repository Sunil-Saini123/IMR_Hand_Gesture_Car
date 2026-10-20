"""Train a Random Forest on data/gesture_samples.csv and compare it with the rule-based baseline."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gesture_app.features import FEATURE_VERSION, classify_rules, normalize_landmarks
from robot_core.config import REPO_ROOT

DATA = REPO_ROOT / "data" / "gesture_samples.csv"
OUT = REPO_ROOT / "models" / "gesture_rf.joblib"
PLOT = REPO_ROOT / "evaluation" / "plots" / "gesture_confusion.png"


def load_samples(path):
    df = pd.read_csv(path).dropna()
    return df.drop(columns=["label"]).to_numpy(float), df["label"].astype(str).to_numpy()


def train(raw, labels, seed=42):
    """Fit on an 80/20 stratified split; returns (model, test indices)."""
    features = np.stack([normalize_landmarks(row) for row in raw])
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=seed,
                                           stratify=labels)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=120, max_depth=14, class_weight="balanced", random_state=seed, n_jobs=-1))
    model.fit(features[train_idx], labels[train_idx])
    return model, features, test_idx


def plot_confusion(matrix, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--plot", type=Path, default=PLOT)
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run gesture_app/collect_gestures.py first.")
    raw, labels = load_samples(args.data)
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise SystemExit(f"Collect at least 5 samples for at least two labels (have {counts.to_dict()}).")

    model, features, test = train(raw, labels)
    predicted = model.predict(features[test])
    rules = np.array([classify_rules(row) for row in raw[test]])
    names = sorted(set(labels))
    print(f"Samples: {counts.to_dict()} (test split: {len(test)})")
    print(f"Rule-based accuracy:     {accuracy_score(labels[test], rules):.3f}")
    print(f"Random Forest accuracy:  {accuracy_score(labels[test], predicted):.3f}")
    print(classification_report(labels[test], predicted))
    matrix = confusion_matrix(labels[test], predicted, labels=names)
    print("Confusion matrix (rows = true, columns = predicted):", names, "\n", matrix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_version": FEATURE_VERSION, "labels": names}, args.out)
    plot_confusion(matrix, names, args.plot, "Random Forest gesture confusion")
    print("Saved", args.out, "and", args.plot)


if __name__ == "__main__":
    main()
