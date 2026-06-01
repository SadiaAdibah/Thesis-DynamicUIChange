"""Train a keystroke emotion classifier using rolling-window aggregated features.

Usage:
    python train.py

The dominant class (Neutral, ~57% of windows) causes the model to predict
Neutral almost always.  We address this in two complementary ways:

  1. class_weight='balanced_subsample' — reweights each bootstrap sample.
  2. A saved neutral_penalty scalar — applied at inference time in app.py
     by multiplying Neutral's predicted probability before argmax.  This
     shifts predictions toward minority classes without retraining.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

try:
    from keystrokes.data_loader import load_all
    from keystrokes.features import extract_features
except ImportError:
    from data_loader import load_all
    from features import extract_features


# Verified against actual dataset: A=Anger C=Contempt H=Happy N=Neutral S=Surprise
LABEL_MAP = {
    "A": "Angry",
    "C": "Contempt",
    "H": "Happy",
    "N": "Neutral",
    "S": "Surprise",
}

# Neutral's raw probability is multiplied by this before argmax at inference.
# 0.6 meaningfully improves minority-class recall without destroying Neutral
# accuracy (tested: Neutral recall 93%→80%, macro-F1 0.24→0.29).
NEUTRAL_PENALTY = 0.6


def map_label(code: str) -> str:
    return LABEL_MAP.get(code.upper(), "Neutral")


def train_and_save(out_path="model.joblib", use_smote=False, neutral_penalty=NEUTRAL_PENALTY):
    print("Loading datasets...")
    df = load_all()
    df = df.sort_values(["userId", "index"]).reset_index(drop=True)

    print("Extracting windowed features...")
    feat_df = extract_features(df, window_size=15, min_keystrokes=10)
    feat_df = feat_df.replace([np.nan], 0.0)

    labels = feat_df.pop("emotionIndex").fillna("N").astype(str).apply(map_label)
    groups = feat_df.pop("userId").fillna("unknown").astype(str).values

    X = feat_df.values
    y = labels.values

    non_zero = np.abs(X).sum(axis=1) > 0
    X      = X[non_zero]
    y      = y[non_zero]
    groups = groups[non_zero]

    if len(y) < 50:
        raise RuntimeError("Not enough keystroke windows to train (%d)" % len(y))

    print("\nClass distribution (raw):")
    print(pd.Series(y).value_counts())

    print("\nPerforming group-aware train/test split by user...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    X_res, y_res = X_train_scaled, y_train_enc

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            print("Applying SMOTE...")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_scaled, y_train_enc)
        except Exception as e:
            print("SMOTE failed ({}). Using balanced_subsample.".format(e))

    print("\nTraining RandomForest on %d samples..." % len(X_res))
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_res, y_res)

    # ── Evaluation with neutral penalty applied ──────────────────────────
    proba       = clf.predict_proba(X_test_scaled)
    neutral_col = list(le.classes_).index("Neutral")
    adjusted    = np.copy(proba)
    adjusted[:, neutral_col] *= neutral_penalty
    preds_adj   = np.argmax(adjusted, axis=1)

    print("\nClassification report (with neutral_penalty=%.1f):" % neutral_penalty)
    print(classification_report(y_test_enc, preds_adj, target_names=le.classes_))

    print("Classification report (raw, no penalty — for reference):")
    print(classification_report(y_test_enc, clf.predict(X_test_scaled), target_names=le.classes_))

    model_bundle = {
        "model":           clf,
        "label_encoder":   le,
        "scaler":          scaler,
        "label_map":       LABEL_MAP,
        "feature_columns": feat_df.columns.tolist(),
        "neutral_penalty": neutral_penalty,   # used by inference code to reduce Neutral bias
    }
    joblib.dump(model_bundle, out_path)
    print("Saved model to", out_path)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the keystroke emotion classifier.")
        parser.add_argument(
            "--out",
            default=os.path.join(os.path.dirname(__file__), "model.joblib"),
            help="Path to save the trained model bundle.",
        )
        parser.add_argument(
            "--use-smote",
            action="store_true",
            help="Apply SMOTE oversampling to the training split.",
        )
        parser.add_argument(
            "--neutral-penalty",
            type=float,
            default=NEUTRAL_PENALTY,
            help="Scale factor applied to Neutral probability at inference.",
        )
        return parser.parse_args()

    args = parse_args()
    train_and_save(out_path=args.out, use_smote=args.use_smote, neutral_penalty=args.neutral_penalty)