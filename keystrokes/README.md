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

```bash
python -m keystrokes.train
```

3. Run the server and open the demo in a browser:

```bash
python -m keystrokes.server
# then open http://localhost:9000/
```

Notes:
- Label mapping is heuristic: `A->Angry, H->Happy, S->Surprise, N/C->Neutral` and unknown codes map to Neutral.
- The feature set is intentionally small (per-row timing columns) so it runs quickly. You can extend `features.py`.
