"""Train a small Random Forest from data/gesture_samples.csv."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA, OUT = "data/gesture_samples.csv", "models/gesture_rf.joblib"


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError("Run gesture_app/collect_gestures.py first.")
    df = pd.read_csv(DATA).dropna()
    if df["label"].nunique() < 2:
        raise ValueError("Collect samples for at least two labels.")
    X = df.drop(columns=["label"]).to_numpy(float)
    y = df["label"].astype(str).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, max_depth=14,
                          class_weight="balanced", random_state=42, n_jobs=-1))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Accuracy:", accuracy_score(yte, pred))
    print(classification_report(yte, pred))
    print("Confusion matrix:\n", confusion_matrix(yte, pred, labels=sorted(set(y))))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, OUT)
    print("Saved", OUT)


if __name__ == "__main__": main()
