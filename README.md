
# LeafLens: Mango Leaf Disease Detection (CS333)

Image classification of eight mango leaf classes using TensorFlow/Keras and MobileNetV3:

- Healthy
- Anthracnose
- Bacterial Canker
- Cutting Weevil
- Die Back
- Gall Midge
- Powdery Mildew
- Sooty Mould

Team: **Member A**, **Member B** (replace with names)

---

## Project status

| Step | Status | Output |
|---|---|---|
| Environment and repository | Done | `requirements.txt`, `requirements-train.txt` |
| Dataset download | Done | `data/raw/` (not committed) |
| Manifest generation | Done | `data/manifest.csv` |
| Corrupted-image check | Done | `src/check_integrity.py` |
| Near-duplicate detection and removal | Done | `src/check_duplicates.py`, `src/dedupe.py` |
| Grouped 10-second split | Done | `src/make_grouped_split_v2.py`, `data/manifest_grouped_10s.csv` |
| Exploratory data analysis | Done | `notebooks/01_eda.ipynb`, `outputs/plots/` |
| Preprocessing and augmentation | Done | `src/data.py` |
| Baseline CNN training | Done | `src/model.py`, `src/train.py` |
| MobileNetV3-Small training | Done | `src/model.py`, `src/train.py` |
| MobileNetV3-Large training | Done | `src/model.py`, `src/train.py` |
| Grouped-split test evaluation | Done | `src/evaluate.py`, `outputs/evaluation/` |
| Single-image inference script | Implemented | `src/inference.py` |
| Grad-CAM analysis | Pending update | `src/gradcam.py` |
| Local Philippine mango-leaf evaluation | Pending | Local photo dataset |
| Final model selection and report | Pending | Model comparison and analysis |

The dataset originally contained **4,000 images**. After removing 294 near-duplicates, **3,706 images** remained.

The final grouped manifest contains:

| Split | Images |
|---|---:|
| Training | 2,593 |
| Validation | 558 |
| Test | 555 |
| **Total** | **3,706** |

The final pipeline uses `data/manifest_grouped_10s.csv`. Do not regenerate or randomly reassign these splits.

---

## Final grouped-split test results

All three models were evaluated on the same 555-image grouped test set.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Baseline CNN (GroupNorm) | 91.35% | 90.90% |
| MobileNetV3-Small | 97.12% | 97.33% |
| MobileNetV3-Large | **98.02%** | **98.17%** |

These figures come from the saved `summary.csv` files in `outputs/evaluation/`.

MobileNetV3-Large achieved the highest test accuracy and macro-F1. However, the final model decision should also consider model size, inference speed, Grad-CAM findings, and performance on local Philippine mango-leaf photographs.

Each model's evaluation folder contains:

- `summary.csv` — overall test metrics and efficiency measurements
- `classification_report.csv` — per-class precision, recall, and F1
- `confusion_matrix.csv` — confusion-matrix values
- `confusion_matrix.png` — confusion-matrix visualization
- `predictions.csv` — predicted classes and confidence scores

**Important:** These test results describe performance on the grouped Kaggle dataset. They do not establish equivalent performance on real-world Philippine mango-leaf photographs.

### Historical random-split results

The original experiments used an image-level random split and produced unusually high validation results.

| Model | Historical validation accuracy | Historical validation macro-F1 |
|---|---:|---:|
| Baseline CNN | 95.86% | 95.99% |
| MobileNetV3-Small | 99.82% | 99.83% |
| MobileNetV3-Large | 99.64% | 99.67% |

These results should **not** be compared directly with the final grouped-split test scores.

The original random split allowed images captured seconds apart to appear in different splits. This introduced potential same-leaf leakage and made the original validation results optimistic.

Historical run records are preserved in `outputs/runs_old.csv`. The current `outputs/runs.csv` contains the locally recorded grouped-split smoke test; it is not a complete record of all final Colab training runs.

---

## Dataset leakage investigation

The original random split was investigated using image similarity and capture timestamps embedded in image filenames.

Photographs taken seconds apart can show the same leaf from slightly different angles. If these photographs appear in both training and evaluation splits, the model may benefit from visual similarities that do not reflect generalization to independent leaves.

To reduce this problem, we generated a deterministic grouped split using:

- `src/make_grouped_split_v2.py`
- A 10-second grouping threshold
- Fixed random seed 42
- Class-aware split allocation

The grouped manifest was independently regenerated and its assignments matched the saved version.

### Final timestamp leakage check

The final grouped manifest was checked using:

```powershell
python src/check_time_leakage.py --manifest data/manifest_grouped_10s.csv --out outputs/time_leakage_10s.csv
```

Results:

| Time gap to closest same-class training photo | Test | Validation |
|---|---:|---:|
| Within 10 seconds | 0/555 (0.0%) | 0/558 (0.0%) |
| Within 30 seconds | 186/555 (33.5%) | 218/558 (39.1%) |
| Within 60 seconds | 337/555 (60.7%) | 385/558 (69.0%) |

The median gap was 45 seconds for test images and 39 seconds for validation images.

**Limitation:** The grouped split eliminates cross-split same-class timestamp pairs within 10 seconds, but this does not prove that every image belongs to an independent leaf. Some photographs taken more than 10 seconds apart may still depict the same leaf.

The dataset does not provide verified leaf identities, so residual same-leaf leakage cannot be ruled out.

---

## Setup

Run these commands from the repository root.

### 1. Install the tools

Install Python 3.10 or 3.11, Git, and VS Code with the Python and Jupyter extensions.

### 2. Clone the repository

```powershell
git clone https://github.com/achctonin/leaflens.git
cd leaflens
```

### 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the environment again.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
python -m pip install ipykernel
```

### 5. Verify the environment

```powershell
python -c "import pandas, PIL, sklearn, cv2; print('ok')"
python src/model.py
```

Native Windows TensorFlow generally uses the CPU. Google Colab or a compatible Linux GPU environment can be used for GPU training.

---

## Dataset setup

Dataset source:

[Mango Leaf Disease Dataset — Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)

Each team member downloads the dataset separately.

Extract the images into:

```text
data/
└── raw/
    ├── Anthracnose/
    ├── Bacterial Canker/
    ├── Cutting Weevil/
    ├── Die Back/
    ├── Gall Midge/
    ├── Healthy/
    ├── Powdery Mildew/
    └── Sooty Mould/
```

The images are not committed to GitHub.

The original `data/raw/` folder may contain images excluded during deduplication. Always use the manifest to determine which images belong to the dataset and which split they belong to.

**Current manifest:** `data/manifest_grouped_10s.csv`

**Historical manifest:** `data/manifest.csv`

The current dataset loader defaults to the grouped manifest.

---

## Project layout

```text
leaflens/
├── data/
│   ├── raw/                       # Downloaded images (not committed)
│   ├── manifest.csv               # Original manifest
│   ├── manifest_dedup.csv         # Deduplicated dataset
│   └── manifest_grouped_10s.csv   # Final grouped split
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data.py                    # Dataset loading and augmentation
│   ├── model.py                   # CNN and MobileNetV3 architectures
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Test evaluation
│   ├── inference.py               # Single-image inference
│   ├── gradcam.py                 # Grad-CAM (pending update)
│   ├── make_grouped_split_v2.py   # Grouped split generation
│   ├── check_time_leakage.py      # Timestamp leakage analysis
│   ├── show_time_pairs.py         # Leakage-pair visualization
│   └── diagnose_bn.py             # Historical BatchNorm investigation
├── outputs/
│   ├── evaluation/                # Final grouped test results
│   ├── plots/                     # EDA and diagnostic figures
│   ├── runs.csv                   # Current local run log
│   └── runs_old.csv               # Historical random-split runs
├── requirements.txt
├── requirements-train.txt
└── README.md
```

Additional data-preparation and analysis scripts remain in `src/`.

Trained `.keras` checkpoints are not committed to GitHub. Final grouped model checkpoints are stored separately in Google Drive.

---

## Data and pipeline decisions

- **Input size:** 224 × 224 RGB images, resized without preserving the original aspect ratio.
- **Pixel range:** Raw float32 pixel values from 0 to 255. MobileNetV3 performs its expected preprocessing internally; the baseline model handles rescaling.
- **Augmentation:** Applied only to training images, including flips, rotation, zoom, brightness, and contrast adjustments.
- **Class weights:** Computed from the training split to address class imbalance.
- **Split before augmentation:** Validation and test images are never augmented.
- **Reproducibility:** Fixed seed 42 and a committed grouped manifest.
- **Evaluation:** Accuracy, macro-F1, per-class recall, and confusion matrices.

### Dataset limitations

Cutting Weevil is the smallest class after deduplication, with 246 images. Its images also show relatively uniform capture characteristics, which may encourage the model to learn background or acquisition cues rather than disease symptoms.

The Kaggle photographs commonly show individual leaves against relatively simple backgrounds. Real-world leaves may have different lighting, backgrounds, camera quality, disease severity, and overlapping symptoms.

These limitations motivate Grad-CAM analysis and a separate local-photo evaluation.

---

## Exploratory data analysis

The original dataset contained 500 images per class.

After removing 294 near-duplicates, the dataset contained 3,706 images. Cutting Weevil lost 254 images and became the smallest class.

Other observations:

- Most images have dimensions of 320 × 240, 240 × 320, or 240 × 240.
- Some classes have relatively uniform image characteristics.
- Brightness distributions are broadly similar across classes.
- Many images use plain backgrounds, which differ from natural field conditions.

EDA plots are saved in `outputs/plots/`.

---

## Training

Training uses `src/train.py` and the grouped manifest configured in `src/data.py`.

The baseline CNN uses group normalization. MobileNetV3-Small and MobileNetV3-Large use ImageNet-pretrained backbones and a two-stage transfer-learning approach.

For MobileNetV3:

1. Train the classification head with the backbone frozen.
2. Fine-tune selected upper backbone layers using a lower learning rate.

Training uses validation loss for early stopping and checkpoint selection. This means validation results are part of the model-selection process, not an independent final performance estimate.

The training script records local run information in `outputs/runs.csv` and saves model checkpoints under `outputs/models/`.

Final grouped model checkpoints from Colab are stored separately in Google Drive.

### Why the baseline uses group normalization

An earlier baseline CNN using BatchNorm showed a large difference between training-mode and evaluation-mode performance.

The investigation in `src/diagnose_bn.py` indicated problems involving BatchNorm running statistics. The baseline was changed to group normalization, which does not rely on running batch statistics.

This debugging history is retained for reproducibility and the final report.

---

## Evaluation

The final grouped test evaluation has already been completed for all three models.

To evaluate a saved model when necessary:

```powershell
python src/evaluate.py --model-path "path/to/model.keras"
```

The script saves results under:

```text
outputs/evaluation/<model_name>/
```

Do not repeatedly use test results to tune model settings or select new hyperparameters. The test set has already been used for final evaluation.

The existing results should be preserved rather than rerunning training or evaluation unnecessarily.

---

## Single-image inference

Use `src/inference.py` to predict the class of one mango-leaf photograph:

```powershell
python src/inference.py --model "path/to/model.keras" --image "path/to/leaf.jpg"
```

The script:

1. Loads the saved model.
2. Converts the input image to RGB.
3. Resizes it to 224 × 224.
4. Predicts the disease class.
5. Displays the predicted class, confidence, and all class probabilities.

The confidence value is a model probability, not a guarantee of diagnostic correctness.

The script has been implemented and reviewed, but independent real-world inference testing remains part of the remaining work.

---

## Remaining work

| Task | Status |
|---|---|
| Grouped-split training | Completed |
| Final Kaggle test evaluation | Completed |
| Single-image inference implementation | Completed |
| Grad-CAM update and analysis | Pending; assigned to teammate |
| Misclassification analysis | Pending |
| Collect and evaluate local Philippine mango-leaf photos | Pending |
| Compare model size, speed, and real-world performance | Pending final analysis |
| Final model selection | Pending |
| Final project report and presentation | Pending |

For the local-photo evaluation, collect independently sourced images and document how their labels were determined. Self-assigned labels should not be treated as expert-verified diagnoses.

Avoid using local-photo results to repeatedly tune the model and then reporting those same images as an independent evaluation.

---

## Git and collaboration workflow

- Keep the flat `src/` project structure unless both members agree to change it.
- Use separate branches and pull requests for team changes.
- Pull the latest changes before starting new work.
- Preserve the final grouped manifest and seed 42.
- Do not randomly re-split the dataset.
- Do not augment validation or test images.
- Do not tune models using final test results.
- Report macro-F1, per-class recall, and confusion matrices alongside accuracy.
- Do not commit `data/raw/` or large `.keras` model checkpoints.
- Keep historical random-split results clearly separated from final grouped-split results.
- Update Grad-CAM separately after coordinating with the teammate responsible for it.

LeafLens is a university image-classification project. Its predictions should not be treated as independently validated agricultural diagnoses.
