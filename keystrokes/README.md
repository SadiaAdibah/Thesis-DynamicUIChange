Keystroke-based Emotion Recognition
=================================

This folder contains tools to train and serve a keystroke-based emotion classifier using the provided datasets.

Files:
- `data_loader.py` - loads the provided CSV datasets and normalizes numeric columns
- `features.py` - extracts per-keystroke features
- `train.py` - trains a RandomForest and writes `model.joblib`
- `server.py` - Flask server that loads `model.joblib` and exposes `/predict`
- `static/index.html` - minimal client to capture key timings and call `/predict`
- `requirements.txt` - Python dependencies

Quick start:

1. From the repository root, create a virtualenv and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r keystrokes/requirements.txt
```

2. Train the model (this uses `datasets/Fixed Text Typing Dataset.csv` and `Free Text Typing Dataset.csv`):

Option A: from the repository root

```bash
python -m keystrokes.train
```

Option B: from the `keystrokes/` folder

```bash
cd keystrokes
python train.py
```

3. Run the server and open the demo in a browser:

```bash
python -m keystrokes.server
# then open http://localhost:9000/
```

Notes:
- Label mapping is now broader to reduce overfitting: `A/D/F->Negative`, `H/S->Positive`, `N/C->Neutral`.
- Training uses rolling-window aggregates over 10-20 keystrokes rather than only the last 3 keypresses.
- The model is split using user-aware group validation to avoid leakage between users.
