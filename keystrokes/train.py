"""Train a simple keystroke-to-emotion classifier.

Usage:
    python train.py

This script will load datasets from ../datasets, extract features, train a RandomForest,
print a classification report and save the model to `model.joblib`.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from keystrokes.data_loader import load_all
from keystrokes.features import extract_features


LABEL_MAP = {
    "A": "Angry",
    "H": "Happy",
    "S": "Surprise",
    "N": "Neutral",
    "C": "Neutral",
    "D": "Disgust",
    "F": "Fear",
}


def map_label(code: str) -> str:
    return LABEL_MAP.get(code.upper(), "Neutral")


def train_and_save(out_path="model.joblib", use_smote=True):
    print("Loading datasets...")
    df = load_all()

    print("Filtering rows and extracting features...")
    # drop rows without any numeric timing
    feat_df = extract_features(df)
    feat_df = feat_df.replace([np.nan], 0.0)

    labels = df["emotionIndex"].fillna("N").astype(str).apply(map_label)

    X = feat_df.values
    y = labels.values

    # quick filter: drop rows where all features are zero
    non_zero = (np.abs(X).sum(axis=1) > 0)
    X = X[non_zero]
    y = y[non_zero]

    if len(y) < 50:
        raise RuntimeError("Not enough keystroke rows to train (%d)" % len(y))

    # Try SMOTE oversampling after scaling (preferred) otherwise fallback to neutral downsampling
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_res, y_res = X_scaled, y_enc

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            print("Applying SMOTE to balance classes...")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_scaled, y_enc)
        except Exception as e:
            print("SMOTE not available or failed ({}). Falling back to downsampling.".format(e))
            use_smote = False

    if not use_smote:
        # Build dataframe to allow per-class sampling (downsample Neutral)
        feat_cols = feat_df.columns.tolist()
        df_all = pd.DataFrame(X, columns=feat_cols)
        df_all["label"] = y

        print("Class distribution before downsampling:")
        print(df_all["label"].value_counts())

        counts = df_all["label"].value_counts()
        if "Neutral" in counts.index:
            minority_max = int(counts.drop(labels=["Neutral"]).max())
            target = minority_max
        else:
            target = int(counts.max())

        parts = []
        for lbl, grp in df_all.groupby("label"):
            if lbl == "Neutral" and len(grp) > target:
                grp = grp.sample(n=target, random_state=42)
            parts.append(grp)

        df_bal = pd.concat(parts, ignore_index=True)

        print("Class distribution after downsampling:")
        print(df_bal["label"].value_counts())

        X_res = df_bal[feat_cols].values
        y_res = LabelEncoder().fit_transform(df_bal["label"].values)

    # Train on X_res, y_res
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    print("Training RandomForest on %d samples..." % len(X_train))
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=le.inverse_transform(sorted(set(y_res)))))

    model_bundle = {"model": clf, "label_encoder": le, "scaler": scaler}
    joblib.dump(model_bundle, out_path)
    print("Saved model to", out_path)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    train_and_save(out)
