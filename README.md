# Active Thermography for Non-destructive Determination of Bud Mortality for Grapes: Model Analysis

**Guangxun Zhai, Yu Jiang, John Owens, Justine E. Vanden Heuvel, Jinhong Yu, Alan Zayd Zoubi**

> Code repository for the paper: *Active Thermography for Non-destructive Determination of Bud Mortality for Grapes: Model Analysis* (Part II). This work benchmarks learning-based classification of grape-bud viability from pulsed thermography sequences across eight model families and three data representations under universal and cultivar-specific pipelines.

---

## Overview

We benchmark three complementary data representations of bud-level thermal responses:

| Representation | Format | Models |
|---|---|---|
| Full ROI video tensor | 21 × 21 × 600 | ViViT |
| Mean-ROI time-series curve | 1 × 600 | LSTM, Hybrid LSTM, LSTM-FCN, Hybrid LSTM-FCN |
| Engineered / PCA features | 6–7 variables | Logistic Regression, Random Forest, SVM |

Two modeling strategies are evaluated:
- **Universal** — trained on pooled data from all four cultivars (Cabernet Franc, Concord, Pinot Noir, Riesling)
- **Cultivar-specific** — trained and evaluated independently per cultivar

Performance is reported across accuracy, F1, precision, recall, and ROC–AUC over **30 independent runs** with bootstrap 95% confidence intervals, and compared with Kruskal–Wallis + Dunn–Holm post-hoc tests.

---

## Repository Structure

```
.
├── data-preprocess/          # Data preparation pipeline
│   ├── process/              # Notebooks: ROI extraction, mean-curve, feature engineering, PCA
│   ├── analysis/             # Exploratory analysis and outlier inspection notebooks
│   └── data-split/           # Train/test split notebook and outputs
│
├── roi_data/                 # Raw ROI arrays (.npy, excluded from git — see Data section)
│
├── traditional_ml/           # Logistic Regression, Random Forest, SVM
│   ├── pipelines/            # Training scripts (run.sh, generate_report.py)
│   ├── cultivar-specific/    # Per-cultivar notebooks (CF, Concord, PN, Riesling)
│   └── *.ipynb               # Universal model notebooks
│
├── lstm/                     # LSTM, Hybrid LSTM, LSTM-FCN, Hybrid LSTM-FCN
│   ├── model/                # Architecture definitions (lstm.py, lstmfcn.py, fcn.py)
│   ├── pipelines/            # Training scripts (run.sh, generate_report.py)
│   ├── cultivar-specific/    # Per-cultivar notebooks
│   └── *.ipynb               # Universal model notebooks
│
├── transformer/              # ViViT video transformer
│   ├── utils/                # Data handling, preprocessing, model configuration
│   ├── data_augumentation/   # Flip, rotate, augment scripts
│   ├── vivit-run.py          # Universal training entry point
│   ├── abnormal-vivit-run.py # Outlier-filtered training
│   └── requirements.txt      # Full dependency list for the ViViT environment
│
└── model_compare/            # Aggregation and visualization
    ├── figures/              # Final comparison bar charts (SVG) for paper figures
    ├── tables/               # Dunn post-hoc test result CSVs (S1–S30)
    ├── csv/                  # Aggregated summary metrics per cultivar
    └── compare_*.ipynb       # Notebooks to reproduce Figures 3 & 4
```

---

## Data

Thermal imaging data were collected from mid-January to early April 2024 using a synchronized pulsed active thermography system (described in the companion Part I paper). Each bud sample was recorded during a heating–cooling cycle (heating: 1–10 s; acquisition end: 20 s) at 30 fps, yielding 600 frames.

After ROI extraction, each sample is a **21 × 21 × 600** grayscale tensor. Three modalities are derived:

- **ROI video** — raw spatial-temporal tensor (used by ViViT)
- **Mean-ROI time-series** — spatial mean over the 21 × 21 region, yielding a 600-step curve (used by LSTM models)
- **Engineered features** — PC1–PC3 from PCA, rising/falling waveform parameters, and peak temperature (6–7 variables; used by traditional ML models)

### Dataset Split

| Split | Viable | Non-viable | Total |
|---|---|---|---|
| Universal train | 1529 | 694 | 2223 |
| Universal test | 280 | 280 | 560 |
| Per-cultivar train | ~350–405 | ~145–192 | ~526–588 |
| Per-cultivar test | 70 | 70 | 140 |

The test set is **fixed** and **class-balanced** (70 viable + 70 non-viable per cultivar). Raw `.npy` ROI files are excluded from this repository due to file size. Processed feature CSV files are included under `roi_data/csv/` and `data-preprocess/data/`.

---

## Reproducing the Results

### 1. Data Preprocessing

Run the notebooks in `data-preprocess/process/` in order:

1. `roi-video.ipynb` — ROI video extraction per round
2. `roi-video-combine.ipynb` — combine rounds
3. `mean-time-series.ipynb` — compute mean-ROI curves
4. `features.ipynb` — PCA and waveform feature extraction
5. `features-combine.ipynb` — merge feature rounds
6. `data-split/split.ipynb` — generate fixed train/test split

### 2. Traditional ML (LR, RF, SVM)

```bash
# Universal
cd traditional_ml/pipelines
bash run.sh

# Cultivar-specific — open and run per-cultivar notebooks in traditional_ml/cultivar-specific/
```

### 3. LSTM Family (LSTM, Hybrid LSTM, LSTM-FCN, Hybrid LSTM-FCN)

```bash
# Universal
cd lstm/pipelines
bash run.sh

# Cultivar-specific
cd lstm/cultivar-specific
# Run notebooks for each cultivar (cf/, conc/, pn/, ries/)
```

### 4. ViViT

Requires a Linux system with multiple GPUs. Install dependencies first:

```bash
pip install -r transformer/requirements.txt
```

Key packages: `transformers==4.46.3`, `torch==2.6.0`, `tensorflow==2.18.0`, `accelerate==1.6.0`

```bash
cd transformer

# Universal
python vivit-run.py

# Cultivar-specific (set cultivar flag in script)
python vivit-run.py --cultivar CF   # or CON, PN, RIES
```

Preprocessed ViViT-compatible tensors are expected under `transformer/refined_data_21/` (excluded from git due to size).

### 5. Comparison and Figures

Open notebooks in `model_compare/`:

```
compare_all.ipynb    → universal model comparison (Figure 3)
compare_cf.ipynb     → Cabernet Franc (Figure 4a)
compare_con.ipynb    → Concord (Figure 4b)
compare_pn.ipynb     → Pinot Noir (Figure 4c)
compare_ries.ipynb   → Riesling (Figure 4d)
tables.ipynb         → Dunn–Holm post-hoc tables (Supplementary S1–S30)
```

---

## Key Results

Performances clustered tightly across all eight model families, with overlapping 95% CIs for most metric-model pairs. Different families showed small, metric-specific advantages.

### Deployment Recommendations

**Sampling-based estimation (precision-leaning):**

| Setting | Recommended Model |
|---|---|
| Universal | LSTM |
| Cabernet Franc | Hybrid LSTM |
| Concord | Random Forest |
| Pinot Noir | RF / SVM / LSTM-family |
| Riesling | LSTM-family / ViViT |

**Full-area classification (recall-leaning):**

| Setting | Recommended Model |
|---|---|
| Universal | RF / SVM |
| Cabernet Franc | LR / RF |
| Concord | SVM / ViViT |
| Pinot Noir | SVM / ViViT |
| Riesling | SVM |

---

## Hardware

| Component | Hardware |
|---|---|
| Traditional ML + LSTM models | Windows PC, NVIDIA RTX 4070 Super (12 GB) |
| ViViT fine-tuning | Linux, 8× NVIDIA RTX A6000 (48 GB each) |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhai2026thermography,
  title   = {Active Thermography for Non-destructive Determination of Bud Mortality for Grapes: Model Analysis},
  author  = {Zhai, Guangxun and Jiang, Yu and Owens, John and Vanden Heuvel, Justine E. and Yu, Jinhong and Zoubi, Alan Zayd},
  year    = {2026}
}
```

---

## License

See [LICENSE](LICENSE) for code license. See [DATA_LICENSE.md](DATA_LICENSE.md) for data terms.