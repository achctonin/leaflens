# LeafLens: Mango Leaf Disease Detection (CS333)

Image classification of 8 mango leaf classes using MobileNetV3 (TensorFlow/Keras).

## Setup
1. Install Python 3.10 or 3.11 and Git.
2. Clone the repo and enter it:
```
   git clone <repo-url>
   cd leaflens
```
3. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac/Linux
```
4. Install dependencies:
```
   python -m pip install -r requirements.txt          # data prep (light)
   python -m pip install -r requirements-train.txt    # adds TensorFlow, for training
```
5. Verify:
```
   python -c "import pandas, PIL, sklearn, cv2; print('ok')"
```

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

Extract into `data/raw/` so you get `data/raw/<class_name>/*.jpg`.
Or with the Kaggle CLI (token in `~/.kaggle/kaggle.json`):
```
kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset -p data/raw --unzip
```
`data/` is not committed. Everyone downloads it themselves.

## Project layout
- `data/raw/` original images (one folder per class = the label)
- `data/manifest.csv` filepath, label, split (committed, so splits are identical)
- `notebooks/` experiments and EDA
- `src/` data.py, model.py, train.py, evaluate.py
- `outputs/` models, plots, metrics

## Team rules
One branch per member, merge via pull request reviewed by the other, fixed seed 42, test set used once at the end.