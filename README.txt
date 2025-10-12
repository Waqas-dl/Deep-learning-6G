6G-Hybrid DL vs Classical Baselines — Reproducible Experiments (README.txt)
============================================================================

This README accompanies the notebook:
  - Best-results-6G_Deep_learning.ipynb

It provides step-by-step instructions to set up the environment, prepare data,
run the experiments (single split), and optionally run 5-fold CV, bootstrap
confidence intervals, and calibration (ECE) plots. It is written to match the
manuscript exactly (single stratified train/test split by default).

----------------------------------------------------------------------------
1) Environment
----------------------------------------------------------------------------
Option A — Conda (recommended)
  conda create -n sixg-dl python=3.10 -y
  conda activate sixg-dl
  pip install -r requirements.txt

Option B — Python venv + pip
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # Linux/Mac: source .venv/bin/activate
  pip install -r requirements.txt

Create a minimal requirements.txt with:
  numpy>=1.24
  pandas>=2.0
  scikit-learn>=1.3
  xgboost>=1.7
  tensorflow>=2.12
  matplotlib>=3.7
  scipy>=1.10
  tqdm>=4.66

GPU (optional): Install the TensorFlow build compatible with your CUDA/CUDNN
stack per TensorFlow documentation. All experiments can run on CPU.

----------------------------------------------------------------------------
2) Repository Layout 
----------------------------------------------------------------------------
.
├── Best-results-6G_Deep_learning.ipynb   # Main notebook (end-to-end)
├── README.txt                            # This file
├── requirements.txt
├── data/                                 # Place your input CSV here (not tracked)
│   └── medical_resource_allocation_dataset.csv
└── outputs/                              # Notebook saves metrics/plots here
    ├── metrics.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── reliability_diagram.png
    └── cv_metrics.csv   # (created if you run optional 5-fold CV)

----------------------------------------------------------------------------
3) Data
----------------------------------------------------------------------------
- Input format: tabular CSV with feature columns and a binary target column.
- Default target column name used by the notebook: label (values 0/1).


Edit the top of the notebook to set:
  DATA_PATH  -> path to your CSV (e.g., medical_resource_allocation_dataset.csv)
  TARGET_COL -> name of the binary target column (default: "label")
  OUTPUT_DIR -> directory for plots/metrics (default: outputs/)

Example schema :
  age: float
  sex: int (0/1)
  comorbidity_score: float
  lqp: float (link-quality proxy)
  nsi: float (network-stress index)
  ... other numeric/encoded categorical features ...
  label: int (0/1 target)

Note: The notebook fits all preprocessing on TRAIN ONLY and applies it to TEST
to avoid leakage. Categorical encoding/scaling steps are commented in-line.

----------------------------------------------------------------------------
4) Reproducibility Settings
----------------------------------------------------------------------------
- RANDOM_STATE = 42 for splitting and model seeds.
- Split: stratified 80/20 train_test_split (as reported in the manuscript).
- DL early stopping: patience=7 on validation loss;
  ReduceLROnPlateau: patience=3, factor=0.5, min_lr=1e-6.
- Pin versions with requirements.txt; optionally save "pip freeze" to
  outputs/requirements_freeze.txt for archival.

----------------------------------------------------------------------------
5) How to Run
----------------------------------------------------------------------------
Interactive (Jupyter or JupyterLab)
  jupyter lab
  # open Best-results-6G_Deep_learning.ipynb and "Run All"

Headless (Papermill)
  pip install papermill
  papermill Best-results-6G_Deep_learning.ipynb outputs/run.ipynb     -p DATA_PATH "data/<your_dataset>.csv"     -p TARGET_COL "label"     -p OUTPUT_DIR "outputs"

The notebook will:
  1. Load data and print shape/class balance.
  2. Preprocess features (fit on train; transform test).
  3. Train/evaluate RF, SVM (RBF), XGB (fixed configs).
  4. Train/evaluate the hybrid DL model with early stopping.
  5. Report Accuracy, F1, AUC; save confusion matrix and ROC curve.
  6. Optionally compute bootstrap 95% CIs and ECE; save reliability diagram.

Artifacts are written to the OUTPUT_DIR (default: outputs/).

----------------------------------------------------------------------------
6) Optional: 5-Fold Cross-Validation
----------------------------------------------------------------------------
To run k-fold CV in addition to the single split, set RUN_KFOLD = True near the
top of the notebook. The notebook will then:
  - Use StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
  - For classical models (RF/SVM/XGB): fit/evaluate per fold, aggregate mean±std.
  - For the DL model: within each fold, hold out a validation split for early
    stopping; aggregate metrics across folds.
Results are saved to outputs/cv_metrics.csv.

This is optional and does not change the primary single-split results.

----------------------------------------------------------------------------
7) Calibration & Uncertainty
----------------------------------------------------------------------------
- Bootstrap 95% CIs: 1,000 stratified bootstrap resamples on held-out test
  predictions for Accuracy, F1, AUC. Saved to outputs/metrics.json.
- Expected Calibration Error (ECE): default 10 bins. Reliability diagram saved
  to outputs/reliability_diagram.png.
Toggle these via RUN_BOOTSTRAP and RUN_CALIBRATION parameters at the top.

----------------------------------------------------------------------------
8) Hyperparameters 
----------------------------------------------------------------------------
Hybrid DL
  Conv1D filters=32, kernel_size=3
  LSTM units=32
  Multi-Head Attention: heads=4, key_dim=32
  Dense=128, Dropout=0.25, L2=5e-4
  Optimizer: Adam, base lr=5e-4 (warm-up from 1e-5)
  ReduceLROnPlateau: patience=3, factor=0.5, min_lr=1e-6
  Early stopping on val_loss (patience=7), batch_size=32, class weights enabled

Random Forest
  n_estimators=100, max_depth=None (defaults), random_state=42

SVM (RBF)
  C=1.0, gamma="scale", probability=True, random_state=42

XGBoost
  learning_rate=0.3, max_depth=6, subsample=1.0, colsample_bytree=1.0,
  eval_metric="logloss", random_state=42

You can broaden searches or adopt nested CV in future work; defaults here are
kept to mirror the manuscript.

----------------------------------------------------------------------------
9) Reproducing Figures
----------------------------------------------------------------------------
- outputs/confusion_matrix.png
- outputs/roc_curve.png
- outputs/reliability_diagram.png
- outputs/cv_metrics.csv (if RUN_KFOLD=True)

----------------------------------------------------------------------------
10) Troubleshooting
----------------------------------------------------------------------------
- Out-of-memory with DL: reduce batch_size (e.g., 16 or 8).
- Severe class imbalance: class weights are enabled; you can also use
  StratifiedKFold or try threshold moving in analysis.
- Different target name: set TARGET_COL accordingly.
- Non-numeric columns: ensure preprocessing encodes/scales them; see comments.

----------------------------------------------------------------------------
