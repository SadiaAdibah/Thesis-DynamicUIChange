import pandas as pd
import os

# Columns in the datasets
KS_COLUMNS = [
    "userId",
    "emotionIndex",
    "index",
    "keyCode",
    "keyDown",
    "keyUp",
    "D1U1",
    "D1U2",
    "D1D2",
    "U1D2",
    "U1U2",
    "D1U3",
    "D1D3",
    "answer",
]


def _read_csv(path):
    df = pd.read_csv(path, sep=";", names=KS_COLUMNS, header=0, dtype=str, low_memory=False)

    # Normalize numeric representation: replace comma decimal separators with dot
    for col in ["keyDown", "keyUp", "D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # emotion index as string
    df["emotionIndex"] = df["emotionIndex"].astype(str).str.strip()

    return df


def load_all(datasets_dir=None):
    """Load Fixed and Free text datasets and concatenate them.

    Args:
        datasets_dir: path to the `datasets/` folder. If None, looks up one level.
    Returns:
        pandas.DataFrame
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    fixed = os.path.join(datasets_dir, "Fixed Text Typing Dataset.csv")
    free = os.path.join(datasets_dir, "Free Text Typing Dataset.csv")

    parts = []
    if os.path.exists(fixed):
        parts.append(_read_csv(fixed))
    if os.path.exists(free):
        parts.append(_read_csv(free))

    if not parts:
        raise FileNotFoundError("No dataset files found in %s" % datasets_dir)

    df = pd.concat(parts, ignore_index=True)

    return df


if __name__ == "__main__":
    df = load_all()
    print("Loaded", len(df), "rows")
