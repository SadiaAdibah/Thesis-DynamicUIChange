"""Keystroke feature extraction — rolling-window statistics.

Two public entry points:

    extract_features(df, window_size, min_keystrokes)
        Batch mode: operates on a full DataFrame (output of data_loader.load_all).
        Returns a DataFrame of window-level features **plus** ``emotionIndex``
        and ``userId`` columns so train.py can extract labels/groups without
        fragile index re-alignment.

    extract_live_features(history, window_size, min_keystrokes)
        Real-time mode: operates on a list of keystroke namedtuple/objects from
        a live session.  Returns a plain dict matching FEATURE_COLS.
"""
import numpy as np
import pandas as pd

# Raw per-keystroke timing columns used as input signals.
EVENT_FEATURE_COLS = [
    "hold_time",        # D1U1  — how long the key was held
    "down_to_down",     # D1D2  — inter-keypress interval
    "up_to_down",       # U1D2  — flight time between keys
    "up_to_up",         # U1U2  — release-to-release interval
    "down_to_up_prev",  # D1U2  — overlap / gap across key boundary
]

# Output feature column names (used by both batch and live paths).
FEATURE_COLS = [
    "hold_time_mean",        "hold_time_std",        "hold_time_median",
    "down_to_down_mean",     "down_to_down_std",     "down_to_down_median",
    "up_to_down_mean",       "up_to_down_std",       "up_to_down_median",
    "up_to_up_mean",         "up_to_up_std",         "up_to_up_median",
    "down_to_up_prev_mean",  "down_to_up_prev_std",  "down_to_up_prev_median",
    "window_size",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _ensure_numeric(df: pd.DataFrame, cols) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = _to_float(df[col])


def _build_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw keystroke columns into per-event timing signals.

    Supports two data layouts:
      1. Pre-computed digraph columns (D1U1, D1D2, …) — preferred.
      2. Raw keyDown / keyUp timestamps — computes digraphs on the fly.
    """
    delta_cols = ["D1U1", "D1D2", "U1D2", "U1U2", "D1U2"]
    has_delta = all(c in df.columns for c in delta_cols)

    if has_delta:
        _ensure_numeric(df, delta_cols)
        # Clamp implausible values (e.g. negative timestamps stored as large ints).
        for col in delta_cols:
            df.loc[df[col].abs() > 1e9, col] = np.nan

        event = pd.DataFrame(
            {
                "hold_time":       df["D1U1"],
                "down_to_down":    df["D1D2"],
                "up_to_down":      df["U1D2"],
                "up_to_up":        df["U1U2"],
                "down_to_up_prev": df["D1U2"],
            },
            index=df.index,
        )

    elif "keyDown" in df.columns and "keyUp" in df.columns:
        df["keyDown"] = _to_float(df["keyDown"])
        df["keyUp"]   = _to_float(df["keyUp"])

        group_col = df["userId"] if "userId" in df.columns else None
        if group_col is not None:
            prev_down = df.groupby("userId")["keyDown"].shift(1)
            prev_up   = df.groupby("userId")["keyUp"].shift(1)
        else:
            prev_down = df["keyDown"].shift(1)
            prev_up   = df["keyUp"].shift(1)

        event = pd.DataFrame(
            {
                "hold_time":       df["keyUp"] - df["keyDown"],
                "down_to_down":    (df["keyDown"] - prev_down).fillna(0.0),
                "up_to_down":      (df["keyDown"] - prev_up).fillna(0.0),
                "up_to_up":        (df["keyUp"]   - prev_up).fillna(0.0),
                "down_to_up_prev": (df["keyUp"]   - prev_down).fillna(0.0),
            },
            index=df.index,
        )

    else:
        raise ValueError(
            "DataFrame must contain either pre-computed digraph columns "
            "(D1U1, D1D2, U1D2, U1U2, D1U2) or raw keyDown/keyUp timestamps."
        )

    if "userId" in df.columns:
        event["userId"] = df["userId"]
    if "emotionIndex" in df.columns:
        event["emotionIndex"] = df["emotionIndex"]

    return event


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    window_size: int = 15,
    min_keystrokes: int = 10,
) -> pd.DataFrame:
    """Extract rolling-window statistics from a full keystroke DataFrame.

    For each user, a rolling window of ``window_size`` keystrokes is slid
    over the sequence.  Mean, std, and median of each timing signal are
    computed per window, giving 16 features total (5 signals × 3 stats + 1
    window-size column).

    The returned DataFrame also carries ``emotionIndex`` and ``userId``
    columns (taken from the last keystroke in each window) so that
    train.py can extract labels and group assignments without having to
    re-align indices.

    Args:
        df:              Full keystroke DataFrame from data_loader.load_all().
        window_size:     Rolling window length in keystrokes.  Default 15.
                         Values below ~10 make std/median unreliable.
        min_keystrokes:  Minimum keystrokes per user before any windows are
                         produced for that user.  Also the ``min_periods``
                         argument for the rolling aggregation.

    Returns:
        DataFrame with columns = FEATURE_COLS + [emotionIndex, userId].

    Raises:
        ValueError: if no valid windows could be produced.
    """
    df = df.copy()
    df["userId"] = df["userId"].astype(str) if "userId" in df.columns else "unknown"

    if "userId" in df.columns:
        df = df.sort_values(["userId", "index"]).reset_index(drop=True)
    else:
        df = df.sort_values("index").reset_index(drop=True)

    events = _build_event_frame(df)

    groups = (
        events.groupby("userId")
        if "userId" in events.columns
        else [(None, events)]
    )

    windows = []
    for _, group in groups:
        if len(group) < min_keystrokes:
            continue

        timing = group[EVENT_FEATURE_COLS]
        rolling = timing.rolling(window=window_size, min_periods=min_keystrokes)
        agg = rolling.agg(["mean", "std", "median"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg["window_size"] = rolling.count()["hold_time"].astype(float)
        agg = agg.dropna()

        # Carry the label and group identifier from the last keystroke in
        # each window (the index already points there after rolling).
        if "emotionIndex" in group.columns:
            agg["emotionIndex"] = group.loc[agg.index, "emotionIndex"]
        if "userId" in group.columns:
            agg["userId"] = group.loc[agg.index, "userId"]

        windows.append(agg)

    if not windows:
        raise ValueError(
            "No valid windows found. Need at least %d keystrokes per user." % min_keystrokes
        )

    result = (
        pd.concat(windows)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURE_COLS)  # only drop rows where features are NaN
    )
    return result


def extract_live_features(
    history,
    window_size: int = 15,
    min_keystrokes: int = 10,
) -> dict:
    """Compute aggregated window features from a live keystroke buffer.

    Intended for real-time inference inside the UI pipeline.  Each element
    of ``history`` must expose ``.down`` and ``.up`` attributes (timestamps
    in any consistent unit — milliseconds recommended).

    Args:
        history:         List of keystroke objects with .down and .up fields.
        window_size:     Maximum number of recent keystrokes to use.
        min_keystrokes:  Minimum buffer length before features can be computed.

    Returns:
        Dict mapping FEATURE_COLS names to float values.

    Raises:
        ValueError: if the buffer is too short.
    """
    if len(history) < min_keystrokes:
        raise ValueError(
            "Need at least %d keystrokes for live features, got %d."
            % (min_keystrokes, len(history))
        )

    recent = history[-window_size:]
    rows = []
    for i in range(1, len(recent)):
        prev    = recent[i - 1]
        current = recent[i]
        rows.append(
            {
                "hold_time":       current.up   - current.down,
                "down_to_down":    current.down - prev.down,
                "up_to_down":      current.down - prev.up,
                "up_to_up":        current.up   - prev.up,
                "down_to_up_prev": current.up   - prev.down,
            }
        )

    if not rows:
        raise ValueError("Not enough interval-level keystrokes to compute live features.")

    df_hist = pd.DataFrame(rows)
    stats = {}
    for col in EVENT_FEATURE_COLS:
        values = df_hist[col].astype(float)
        stats[f"{col}_mean"]   = float(values.mean())
        stats[f"{col}_std"]    = float(values.std(ddof=0))
        stats[f"{col}_median"] = float(values.median())

    stats["window_size"] = float(len(recent))
    return stats


if __name__ == "__main__":
    print("FEATURE_COLS:", FEATURE_COLS)