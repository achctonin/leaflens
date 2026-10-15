# LeafLens: Mango Leaf Disease Detection (CS333)

Image classification of 8 mango leaf classes (Healthy, Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould) using MobileNetV3 (TensorFlow/Keras).

Team: **Member A**, **Member B** (replace with names)

---

## Project status (read this first)

| Step | Status | Output |
|---|---|---|
| Environment + repo | Done | `requirements.txt`, `requirements-train.txt` |
| Dataset download | Done (each member downloads their own copy) | `data/raw/` (not committed) |
| Manifest from folder names | Done | `data/manifest.csv` |
| Corrupted-file check | Done | `src/check_integrity.py` |
| Near-duplicate check and removal | Done | `src/check_duplicates.py`, `src/dedupe.py`, `data/duplicates_removed.csv` |
| Stratified 70/15/15 split, seed 42 | Done | `src/split.py`, `split` column in `data/manifest.csv` |
| EDA | Done | `notebooks/01_eda.ipynb`, plots in `outputs/plots/` |
| Preprocessing + augmentation | Done | `src/data.py` |
| Baseline CNN | Done | `src/model.py`, `src/train.py` |
| MobileNetV3-Small (frozen, then fine-tuned) | Done | `src/model.py`, `src/train.py` |
| MobileNetV3-Large (frozen, then fine-tuned) | Done | `src/model.py`, `src/train.py` |
| Local leaf photos (real-world test set, 30 to 50) | **Not started** | `data/local/` (to be created) |
| Evaluation: test set, confusion matrix, per-class metrics, Grad-CAM | **Not started** | `src/evaluate.py` (not written yet) |
| Inference script for a single photo | **Not started** | `src/predict.py` (not written yet) |
| Final model choice, then one run on the test set | **Not started** | |

The manifest has **3,706 images** (the original dataset has 4,000; 294 near-duplicates were removed). The `split` column is already filled in: 2,594 train / 556 val / 556 test. Nobody should re-split.

## Results so far (validation set only)

The test set has **not** been used yet. These numbers were used for early stopping and for choosing between stage 1 and stage 2, so they are slightly optimistic. All runs: seed 42, batch size 32, class weights on, augmentation on the training split only. Raw numbers are in `outputs/runs.csv`.

| Model | Val accuracy | Val macro-F1 | Val loss | Wrong images (of 556) | Train time (CPU laptop) |
|---|---|---|---|---|---|
| Baseline CNN | 95.86% | 0.960 | 0.176 | about 23 | 25.5 min |
| MobileNetV3-Small | 99.82% | 0.998 | 0.015 | about 1 | 8.2 min |
| MobileNetV3-Large | 99.64% | 0.997 | 0.007 | about 2 | 20.9 min |

Small and Large are within one image of each other on validation, so accuracy alone will not decide between them. We will also compare model size, speed, Grad-CAM and the local leaf photos. MobileNetV3-Small is about a third the size of Large.

---

## Setup (do this once on your own machine)

1. **Install** Python 3.10 or 3.11 (we use 3.11) and Git. Install VS Code too, with the **Python** and **Jupyter** extensions.
2. **Clone the repo** and enter it:
   ```
   git clone https://github.com/achctonin/leaflens.git
   cd leaflens
   ```
3. **Create and activate a virtual environment:**
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows (PowerShell)
   source .venv/bin/activate     # Mac/Linux
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` and try again. You should see `(.venv)` at the start of the prompt.
4. **Install dependencies:**
   ```
   python -m pip install -r requirements.txt          # data prep and EDA (light)
   python -m pip install -r requirements-train.txt    # adds TensorFlow, needed for training
   python -m pip install ipykernel                    # needed to run notebooks in VS Code
   ```
5. **Verify:**
   ```
   python -c "import pandas, PIL, sklearn, cv2; print('ok')"
   python src/model.py
   ```
   The second command builds all three models and prints their sizes (baseline 242,920 parameters, MobileNetV3-Small 943,736, MobileNetV3-Large 3,004,040).

Native Windows TensorFlow does not use the GPU. All our runs were on a CPU laptop and finished in under 30 minutes each. Use Google Colab (free GPU) or WSL2 if your machine is slower.

## Dataset (each member downloads their own copy)

`data/` is not committed to Git. The images are not in the repo, so you must download them.

1. Open https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset (a free Kaggle account is needed) and click **Download**.
2. Extract the zip into `data/raw/` so the layout is `data/raw/<class_name>/*.jpg` (for example `data/raw/Anthracnose/20211008_124249 (Custom).jpg`). There must be no extra wrapper folder between `raw` and the class folders.
3. Check: you should see 8 class folders inside `data/raw/`.

Alternative with the Kaggle CLI (token saved in `~/.kaggle/kaggle.json`):
```
kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset -p data/raw --unzip
```

**Important:** always load images through `data/manifest.csv`, never by walking the folders. Your `data/raw/` folder still contains the near-duplicate images that were removed from the manifest, and the manifest is what keeps everyone on identical data and splits.

## Project layout
- `data/raw/` original images (one folder per class = the label), not committed
- `data/manifest.csv` filepath, label, split (committed, so splits are identical for everyone)
- `data/duplicates_removed.csv` the near-duplicate images that were removed
- `notebooks/` EDA (`01_eda.ipynb`)
- `src/` data-prep scripts (`make_manifest.py`, `check_integrity.py`, `check_duplicates.py`, `dedupe.py`, `split.py`) and the ML code: `data.py` (datasets and augmentation), `model.py` (baseline and MobileNetV3), `train.py` (training), `diagnose_bn.py` (a debugging check for the baseline). `evaluate.py` and `predict.py` are still to be written.
- `outputs/plots/` EDA plots and the augmentation preview
- `outputs/models/` trained models (not committed, can be large; share through Drive)
- `outputs/runs.csv` one row per training run (committed)

## Data and pipeline decisions
- **Resize:** every image is stretched to 224x224 (no padding). Source images come in three shapes (320x240 landscape, 240x320 portrait, 240x240 square), so stretching distorts the leaf shape a little.
- **No manual normalization.** Keras MobileNetV3 has preprocessing built in and expects raw 0-255 pixels. The baseline rescales to 0-1 in its first layer.
- **Augmentation (training split only):** horizontal and vertical flip, rotation up to about 18 degrees, zoom 10%, brightness and contrast 20%. Validation and test are never augmented.
- **Class weights** are computed from the training split. Cutting Weevil has only 172 training images (about half of the other classes), so its weight is about 1.9 and the others are about 0.93 to 0.96.
- **Split first, augment after.** The test set is used once, at the very end.

## EDA findings (summary)
- 8 classes, 500 images each in the original dataset. After removing 294 near-duplicates: 3,706 images. Cutting Weevil lost 254 of its 500 images (51%), Anthracnose 19, Die Back 10, Bacterial Canker 4, Gall Midge 4, Powdery Mildew 2, Sooty Mould 1, Healthy none. Cutting Weevil is now the smallest class (246 images).
- 96.6% of images are one of three sizes: 320x240, 240x320 or 240x240. A few classes have small irregular crops.
- Median brightness is similar across classes (about 165 to 182). Cutting Weevil images are very uniform (all 240 px tall, narrow brightness range), which could let a model recognize that class from capture conditions instead of the disease. We check this with Grad-CAM and the local leaf photos.
- Photos show a single leaf on a plain surface, so the model does not have to deal with cluttered backgrounds.

## Training

Run from the repo root with the `.venv` active. Each run adds a row to `outputs/runs.csv` and saves models to `outputs/models/`. The script never touches the test split.

```
python src/train.py --model baseline --epochs 12 --run-name baseline_gn
python src/train.py --model mobilenet_small --epochs 15 --finetune-epochs 15 --run-name mnv3_small_v1
python src/train.py --model mobilenet_large --epochs 15 --finetune-epochs 15 --run-name mnv3_large_v1
```

- **MobileNet** trains in two stages: stage 1 trains only the new head on the frozen backbone (learning rate 1e-3); stage 2 unfreezes the top 30 backbone layers and fine-tunes at 1e-5. The script keeps whichever stage has the lower validation loss. BatchNorm layers stay frozen.
- **Early stopping** on validation loss (patience 5) with best-weights restore, and the learning rate halves when validation loss stalls.
- **The first MobileNet run downloads the ImageNet weights**, so it needs internet.
- Use a unique `--run-name` for every run. If you and your teammate both append to `outputs/runs.csv`, keep both sets of rows when git reports a conflict.
- Results can differ by a few images between machines even with seed 42.

### Why the baseline uses group normalization
With BatchNorm, the small baseline CNN reached 80 to 90% training accuracy but stayed at chance level (5 to 18%) on validation, even on its own training images in evaluation mode. `src/diagnose_bn.py` showed that accuracy returned to about 78% when BatchNorm used batch statistics, so the saved running statistics were the problem, not the data. MobileNet was not affected because its BatchNorm layers are frozen. We replaced BatchNorm with group normalization in the baseline, which has no running statistics. Mention this in the report.

---

## What is left

| Task | Member A | Member B |
|---|---|---|
| Local leaf photos (30 to 50 total, one leaf per photo on a plain background, labeled by disease) | Half (suggest: split by class) | Half |
| Evaluation on the test set, once: confusion matrix, per-class precision/recall/F1 | `src/evaluate.py` | Review |
| Grad-CAM and misclassified-image review | Review | Grad-CAM script and analysis |
| Inference script for one photo | `src/predict.py` | Review |
| Final model choice (Small vs Large) | Decide together after Grad-CAM and the local photo test | |
| Proposal and report items from the plan (abstract, data description, evaluation plan, related work, timeline, framework decision) | Check the project plan for what is still missing | |

**Open decision:** one code layout for the repo. This README uses the flat `src/` layout (`python src/train.py ...`). If the team prefers one folder per stage (for example `python -m src.training.train`), move the files in one go and update this README so both members use the same commands.

## Workflow

- One branch per member (`git checkout -b yourname/feature`), merge to `main` via pull request reviewed by the other member.
- Run `git pull` before starting work and commit small, often.
- Fixed seed 42 everywhere. Never re-split the data.
- Split first, augment after. Augment the training set only.
- Never tune anything on the test set. Use it once, at the very end, after the final model is chosen.
- Report macro-F1, per-class recall and the confusion matrix, not just accuracy.
- Do not commit `data/raw/` or `outputs/models/`.