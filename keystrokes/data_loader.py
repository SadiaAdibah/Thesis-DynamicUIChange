"""Load and concatenate the Fixed and Free text keystroke datasets."""
import os
import pandas as pd

# All columns present in both datasets (Free text has an extra _id column
# which is handled separately below).
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

# Columns that contain timing values and need numeric conversion.
TIMING_COLS = ["keyDown", "keyUp", "D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]


def _read_csv(path):
    """Read one dataset CSV, normalising decimal separators and types."""
    # Peek at the header to detect whether the file has the extra _id column
    # (Free text dataset exported from MongoDB).
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().strip()

    has_id = header_line.startswith("_id") or "_id" in header_line.split(";")[:2]
    names = (["_id"] + KS_COLUMNS) if has_id else KS_COLUMNS

    df = pd.read_csv(
        path,
        sep=";",
        names=names,
        header=0,
        dtype=str,
        low_memory=False,
    )

    # Drop the MongoDB ObjectId column — not useful for modelling.
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # Normalise European decimal comma → dot, then cast to float.
    for col in TIMING_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise column name differences between the two files.
    if "userid" in df.columns:
        df = df.rename(columns={"userid": "userId"})

    df["emotionIndex"] = df["emotionIndex"].astype(str).str.strip()
    df["userId"] = df["userId"].astype(str).str.strip()

    return df


def load_all(datasets_dir=None):
    """Load Fixed and Free text datasets and concatenate them.

    Args:
        datasets_dir: path to the folder containing the CSV files.
                      Defaults to a ``datasets/`` folder one level above
                      this file.

    Returns:
        pandas.DataFrame with columns matching KS_COLUMNS.

    Raises:
        FileNotFoundError: if neither dataset file is found.
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    fixed_path = os.path.join(datasets_dir, "Fixed Text Typing Dataset.csv")
    free_path  = os.path.join(datasets_dir, "Free Text Typing Dataset.csv")

    parts = []
    for path in [fixed_path, free_path]:
        if os.path.exists(path):
            parts.append(_read_csv(path))
        else:
            print("Warning: dataset not found, skipping:", path)

    if not parts:
        raise FileNotFoundError("No dataset files found in %s" % datasets_dir)

    df = pd.concat(parts, ignore_index=True)
    print("Loaded %d keystroke rows from %d file(s)." % (len(df), len(parts)))
    return df


if __name__ == "__main__":
    df = load_all()
    print(df["emotionIndex"].value_counts())