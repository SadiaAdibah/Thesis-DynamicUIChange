import numpy as np
import pandas as pd

FEATURE_COLS = [
    "hold_time",
    "prev_hold_time",
    "down_to_down",
    "up_to_down",
    "up_to_up",
    "down_to_up_prev",
    "down_to_down2",
    "down_to_up2",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create keystroke timing features for model training and live prediction.

    The features are computed from absolute keyDown/keyUp timestamps and use
    a short history window of the previous two key presses.
    """
    out = pd.DataFrame()

    df = df.copy()
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")

    # sort rows to ensure timeline order per user if possible
    if "userId" in df.columns:
        df["userId"] = df["userId"].astype(str)
        df = df.sort_values(["userId", "index"]).reset_index(drop=True)
    elif "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "keyDown" in df.columns and "keyUp" in df.columns:
        df["keyDown"] = df["keyDown"].astype(float)
        df["keyUp"] = df["keyUp"].astype(float)
        raw_hold = (df["keyUp"] - df["keyDown"]).abs().sum()
    else:
        raw_hold = 0.0

    delta_cols = ["D1U1", "D1D2", "U1D2", "U1U2", "D1U2", "D1U3", "D1D3"]
    has_delta_cols = all(col in df.columns for col in delta_cols)

    if has_delta_cols:
        for col in delta_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].abs() > 1e9, col] = np.nan

        # Dataset provides timing deltas directly; use shifted deltas to match live feature semantics.
        out["hold_time"] = df["D1U1"].astype(float)
        out["prev_hold_time"] = df.groupby("userId")["D1U1"].shift(1).fillna(0.0) if "userId" in df.columns else df["D1U1"].shift(1).fillna(0.0)
        out["down_to_down"] = df.groupby("userId")["D1D2"].shift(1).fillna(0.0) if "userId" in df.columns else df["D1D2"].shift(1).fillna(0.0)
        out["up_to_down"] = df.groupby("userId")["U1D2"].shift(1).fillna(0.0) if "userId" in df.columns else df["U1D2"].shift(1).fillna(0.0)
        out["up_to_up"] = df.groupby("userId")["U1U2"].shift(1).fillna(0.0) if "userId" in df.columns else df["U1U2"].shift(1).fillna(0.0)
        out["down_to_up_prev"] = df.groupby("userId")["D1U2"].shift(1).fillna(0.0) if "userId" in df.columns else df["D1U2"].shift(1).fillna(0.0)
        out["down_to_down2"] = df.groupby("userId")["D1D3"].shift(2).fillna(0.0) if "userId" in df.columns else df["D1D3"].shift(2).fillna(0.0)
        out["down_to_up2"] = df.groupby("userId")["D1U3"].shift(2).fillna(0.0) if "userId" in df.columns else df["D1U3"].shift(2).fillna(0.0)
    elif raw_hold > 0:
        # Dataset contains absolute key timestamps; use them when valid.
        df["prev_keyDown"] = df.groupby("userId")["keyDown"].shift(1) if "userId" in df.columns else df["keyDown"].shift(1)
        df["prev_keyUp"] = df.groupby("userId")["keyUp"].shift(1) if "userId" in df.columns else df["keyUp"].shift(1)
        df["prev2_keyDown"] = df.groupby("userId")["keyDown"].shift(2) if "userId" in df.columns else df["keyDown"].shift(2)

        out["hold_time"] = df["keyUp"] - df["keyDown"]
        out["prev_hold_time"] = (df["prev_keyUp"] - df["prev_keyDown"]).fillna(0.0)
        out["down_to_down"] = (df["keyDown"] - df["prev_keyDown"]).fillna(0.0)
        out["up_to_down"] = (df["keyDown"] - df["prev_keyUp"]).fillna(0.0)
        out["up_to_up"] = (df["keyUp"] - df["prev_keyUp"]).fillna(0.0)
        out["down_to_up_prev"] = (df["keyUp"] - df["prev_keyDown"]).fillna(0.0)
        out["down_to_down2"] = (df["keyDown"] - df["prev2_keyDown"]).fillna(0.0)
        out["down_to_up2"] = (df["keyUp"] - df["prev2_keyDown"]).fillna(0.0)
    else:
        raise ValueError("DataFrame does not contain usable timing columns for feature extraction")

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def extract_live_features(history):
    """Compute the same feature set from a live keystroke history list."""
    if len(history) < 3:
        raise ValueError("At least 3 events are required to compute live features")

    current = history[-1]
    prev1 = history[-2]
    prev2 = history[-3]

    return {
        "hold_time": current.up - current.down,
        "prev_hold_time": prev1.up - prev1.down,
        "down_to_down": current.down - prev1.down,
        "up_to_down": current.down - prev1.up,
        "up_to_up": current.up - prev1.up,
        "down_to_up_prev": current.up - prev1.down,
        "down_to_down2": current.down - prev2.down,
        "down_to_up2": current.up - prev2.down,
    }


if __name__ == "__main__":
    print("Feature module: FEATURE_COLS=", FEATURE_COLS)
