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
| EDA | **In progress** | `notebooks/01_eda.ipynb` (see "Finishing the EDA") |
| Preprocessing + augmentation | Not started | `src/data.py` |
| Baseline CNN, MobileNetV3 Small/Large | Not started | `src/model.py`, `src/train.py` |
| Evaluation, Grad-CAM, real-world test | Not started | `src/evaluate.py` |

After duplicate removal the manifest has **3,706 images** (the original dataset has about 4,000). The `split` column is already filled in, so nobody should re-split.

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
   python -m pip install -r requirements-train.txt    # adds TensorFlow, only needed for training
   python -m pip install ipykernel                    # needed to run notebooks in VS Code
   ```
5. **Verify:**
   ```
   python -c "import pandas, PIL, sklearn, cv2; print('ok')"
   ```

Note: native Windows TensorFlow does not use the GPU. Train on Google Colab (free GPU) or WSL2 if your laptop has no suitable GPU.

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
- `data/duplicates_removed.csv` list of the near-duplicate images that were removed
- `notebooks/` experiments and EDA
- `src/` data.py, model.py, train.py, evaluate.py plus the data-prep scripts
- `outputs/` models, plots, metrics

## Finishing the EDA

Notebook: `notebooks/01_eda.ipynb`. Run it from VS Code.

**One-time notebook setup:** open the notebook, click **Select Kernel** (top right), choose **Python Environments...**, then pick the `.venv` entry. If it is not listed, run `python -m pip install ipykernel` in the terminal, then run "Developer: Reload Window" from the command palette (`Ctrl+Shift+P`).

The notebook is a series of cells, run in order with `Shift+Enter`. Cell 1 must run first because the others use its `df`.

| Cell | What it does | What to look for |
|---|---|---|
| 1. Setup | Moves to the repo root (so relative paths work), loads `data/manifest.csv`, prints shape | Shape should be `(3706, 3)` |
| 2. Class distribution | Table and stacked bar chart of images per class and split, saved to `outputs/plots/class_distribution.png` | Each class is about 70/15/15; note which class is smallest (expected: Cutting Weevil) and how many images each class lost to deduplication |
| 3. Sample grid | 5 random images per class, saved to `outputs/plots/sample_grid.png` | Backgrounds, lighting, leaves that look alike across classes. Change the seed (42) and rerun to see more |
| 4. Image sizes | Width and height of every image, grouped by size and by class | Are all images the same size? Portrait vs landscape matters because a plain resize to 224x224 stretches non-square images |
| 5. Brightness and color | Mean brightness and RGB per image, boxplot per class, saved to `outputs/plots/brightness_per_class.png` | Does any class differ strongly in brightness, which the model could learn instead of the disease? |

**If the notebook is not finished when you pick it up:**
1. Run `git pull` to get the latest notebook. Open it and see which cells already have outputs.
2. Run Cell 1, then run whichever of Cells 2 to 5 are missing or empty.
3. Save the plots to `outputs/plots/` (the cells already do this) and commit the notebook and the plots.
4. Write 4 to 6 bullet points under a final Markdown cell titled "EDA findings": class balance, image size and orientation, lighting/background differences, anything unusual. These go straight into the data description of the proposal and the final paper.
5. Check the box in the status table above and tell your teammate.

## Workflow

- One branch per member (`git checkout -b yourname/feature`), merge to `main` via pull request reviewed by the other member.
- Run `git pull` before starting work and commit small, often.
- Fixed seed 42 everywhere. Never re-split the data.
- Split first, augment after. Augment the training set only.
- Never tune anything on the test set. Use it once, at the very end.
- Report macro-F1, per-class recall and the confusion matrix, not just accuracy.

## What comes next (after EDA)

Task division from the project plan (Member A / Member B):

| Phase | Member A | Member B |
|---|---|---|
| Preprocessing | Resize and preprocessing pipeline | Augmentation pipeline |
| Baseline | Small CNN from scratch | Training/logging utility (shared) |
| Transfer learning | MobileNetV3-Small, frozen then fine-tuned | MobileNetV3-Large, frozen then fine-tuned |
| Evaluation | Confusion matrix, per-class precision/recall/F1 | Grad-CAM, misclassified-image review |
| Real-world test | Collect and label half of the local leaf photos (30 to 50 total) | Collect and label the other half |

Keras MobileNetV3 has preprocessing built in and expects raw 0-255 pixels, so do not normalize twice.

Still missing for the proposal: abstract, data description, evaluation plan, at least 4 related-work papers, timeline, and the confirmed framework decision (TensorFlow/Keras).