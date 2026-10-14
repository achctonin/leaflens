# LeafLens: Project Kickoff Plan (CS333, 2 members)

Topic: LeafLens: Detecting Mango Leaf Diseases Using Deep Learning
Task: image classification, 8 classes (Healthy, Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould)
Model: CNN via transfer learning, MobileNetV3
Dataset: https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset
Split: 70% train / 15% validation / 15% test (stratified)

Members: **Member A** and **Member B** (assign names at the top once decided).

---

## 1. The dataset has no annotation file or CSV: what to do

For image classification you do not need annotation files. In this kind of dataset the **folder name is the label** (one folder per class). So the "annotation" is something you generate yourselves:

1. Walk `data/raw/<class_name>/*.jpg` and record `filepath, label`.
2. Save it as `data/manifest.csv` (columns: `filepath, label, split`).
3. Everyone loads images through this CSV, never by guessing folders, so both of you always use identical data and splits.

Things to verify after download (do not assume):
- Number of classes is 8 and images per class (the dataset is believed to be about 500 per class, about 4,000 total).
- Image sizes (believed to be around 240x320).
- No corrupted or unreadable files.
- **Near-duplicates.** Some images may be augmented copies of the same leaf. If copies land in both train and test, test accuracy will be inflated (data leakage). Check with perceptual hashing before splitting.

---

## 2. Environment setup (BOTH members do this on their own machine)

Recommended: Python 3.10 or 3.11 with TensorFlow/Keras (MobileNetV3 is built into `keras.applications`, and TFLite export is easy later). Both members must use the same framework. Do not mix TensorFlow and PyTorch.

1. Install Python 3.10 or 3.11 and Git.
2. Create the repo (private GitHub), add `data/` and `.venv/` to `.gitignore`.
3. Create and activate a virtual environment:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate` | Mac/Linux: `source .venv/bin/activate`
4. Install libraries:
   `pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas numpy pillow imagehash jupyter kaggle`
5. Verify:
   `python -c "import tensorflow as tf, cv2, sklearn; print(tf.__version__, cv2.__version__, sklearn.__version__)"`
6. Freeze versions: `pip freeze > requirements.txt` and commit it. The other member installs with `pip install -r requirements.txt`.
7. GPU note: native Windows TensorFlow does not use the GPU. If your laptops are CPU-only, train on **Google Colab** (free GPU) or use WSL2.
8. Download the dataset into `data/raw/`:
   - Manual: download the zip from Kaggle and extract it, or
   - Kaggle CLI: put your `kaggle.json` token in `~/.kaggle/`, then `kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset -p data/raw --unzip`

Suggested folder layout:
```
leaflens/
  data/raw/<class folders>/
  data/manifest.csv
  notebooks/
  src/            (data.py, model.py, train.py, evaluate.py)
  outputs/        (models, plots, metrics)
  requirements.txt
  README.md
```

---

## 3. Data preparation pipeline (in order)

1. **Manifest:** build `manifest.csv` from folders (see section 1).
2. **Integrity check:** try to open every image; list and remove corrupted files.
3. **Duplicate check:** perceptual hash (`imagehash.phash`) per image; group or remove near-duplicates.
4. **Stratified split 70/15/15** with a fixed seed (for example 42). Use `train_test_split` twice: first 70 / 30, then split the 30% in half. Stratify by label both times. Write the result into the `split` column of the CSV. About 350 / 75 / 75 images per class if classes are balanced.
5. **EDA:** class counts, sample grid per class, image size and lighting/background review.
6. **Preprocessing:** resize to 224x224. Keras MobileNetV3 has preprocessing built in and expects raw 0-255 pixels, so do not normalize twice.
7. **Augmentation (training set only):** flip, small rotation, brightness/contrast, zoom. Never augment validation or test.
8. **Rule:** split first, augment after. Never fit or tune anything on the test set.

---

## 4. Modeling plan

1. Baseline: small CNN from scratch (so MobileNetV3 has something to beat).
2. Stage 1: MobileNetV3 with frozen backbone, new classification head, train the head.
3. Stage 2: unfreeze the top layers, fine-tune with a low learning rate.
4. Use early stopping and best-checkpoint saving on validation loss.
5. Log hyperparameters, seed and results for every run (a simple CSV is enough).
6. Evaluate on the test set once, at the very end.

---

## 5. Task division (equal workload, each gets a full pipeline)

| Phase | Member A | Member B |
|---|---|---|
| Setup | Repo, `.gitignore`, `requirements.txt`, Colab notebook template | Kaggle download, folder structure, README with run instructions |
| Data | Manifest CSV + corrupted-file check | Duplicate check + stratified split script |
| EDA | Class distribution plots and counts | Sample grids, image size / lighting / background review |
| Preprocessing | Resize and preprocessing pipeline | Augmentation pipeline |
| Baseline | Small CNN from scratch | Training/logging utility (shared by both) |
| Transfer learning | MobileNetV3-Small, frozen then fine-tuned | MobileNetV3-Large, frozen then fine-tuned |
| Evaluation | Confusion matrix, per-class precision/recall/F1 | Grad-CAM visualizations, misclassified-image review |
| Real-world test | Collect and label half of the local leaf photos | Collect and label the other half |
| Cross-check | Reviews B's code and results | Reviews A's code and results |

Both models use the same split and seed, so Small vs Large is a fair comparison.

Working rules: one branch per member, merge through pull requests reviewed by the other person, short sync twice a week, and do not start a phase until the previous one's output is committed.

---

## 6. Things to take note of and where to focus

- **Metrics:** report macro-F1, per-class recall, confusion matrix, not just accuracy. Missing a disease is worse than a false alarm.
- **Leakage:** duplicates and augmentation-before-split are the biggest ways to get fake high accuracy.
- **Test set discipline:** use it once.
- **Class balance:** if classes turn out imbalanced, use class weights or stratified sampling.
- **Real-world gap:** the objective targets Filipino mango farmers, but the dataset was probably collected elsewhere. Take 30 to 50 photos of local mango leaves for an extra test set and state the domain gap as a limitation.
- **Explainability:** the objective mentions color, spot patterns and texture. A CNN learns these itself, so use Grad-CAM to show what it looks at.
- **Reproducibility:** fixed seeds, pinned `requirements.txt`, saved split CSV.
- **Course target:** check the prof's required minimum accuracy and make sure your evaluation plan reports against it.

---

## 7. What the proposal / project description still lacks

The course handout expects these, and the current topic text does not yet include them:

- Abstract (rationale, high-level description, expected outcomes)
- Data description: source, volume, justification that it is sufficient, and the effort for preparation (here: generating the manifest, cleaning, deduplicating, splitting)
- Evaluation plan: metrics, baseline comparison, success target
- Related work: at least 4 papers in the proposal (at least 6 references in the final paper), for example plant disease detection with CNNs, MobileNet/MobileNetV3, transfer learning on leaf datasets
- Timeline: major and minor work items with effort estimates (check whether the prof wants a Gantt chart)
- Team: confirm group size and that no other group has the same topic
- Clarify wording: say explicitly that labels come from folder names, so no annotation work is needed, and decide TensorFlow/Keras vs PyTorch (the topic currently says "or")

---

## 8. First-week checklist

- [ ] Both members: environment installed and verified
- [ ] Repo created, both can push
- [ ] Dataset downloaded and class counts confirmed
- [ ] `manifest.csv` created
- [ ] Corrupted and duplicate images handled
- [ ] 70/15/15 split saved in the CSV
- [ ] EDA plots committed
- [ ] Framework decision written down
- [ ] Proposal gaps in section 7 assigned
