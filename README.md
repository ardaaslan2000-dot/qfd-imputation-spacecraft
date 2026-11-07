# Data Imputation in Quality Function Deployment under Missing Data Conditions: A Spacecraft Design Application

Code and results accompanying the article

> Aslan, A., & Cebeci, U. *Data imputation in quality function deployment under missing data conditions: A spacecraft design application.* Journal of Engineering Technology and Applied Sciences (under review).

The notebook builds a data-driven House of Quality (HoQ) for spacecraft payload design, introduces MCAR missing values, reconstructs the data with five imputation methods (Mean, KNN, MICE, MissForest, GAIN) and measures how well each method preserves the QFD priorities of the complete data.

## Contents

| File | Description |
|---|---|
| `QFD_imputation.ipynb` | Full analysis with all outputs (HoQ construction, imputation, comparison, sensitivity analyses) |
| `gain_numpy.py` | NumPy implementation of GAIN (Yoon et al., 2018); used when PyTorch is not installed |
| `results/revision_tables.xlsx` | All tables of the article, one sheet per table (including the numerical HoQ matrices) |
| `results/revision_all_runs.csv` | Metrics of all 250 imputation runs (5 missing rates × 10 seeds × 5 methods) |
| `results/figures/` | Figures of the article |

## Data

The analysis uses the payload table of the public *Space Mission Dataset: 1957–2024* (Altaf, 2024):
https://www.kaggle.com/datasets/mzeeshanaltaf/space-mission-dataset-1957-2024

Download `payloads.csv` from Kaggle and place it in the same folder as the notebook. The data file is not redistributed in this repository.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook QFD_imputation.ipynb
```

Run all cells from top to bottom. The original analysis cells come first; the cells marked `[REVISION - R…]` at the end contain the extended experiments:

- missing rates 5 %, 10 %, 15 %, 20 %, 30 % with random seeds 1–10 (MCAR on all 15 technical characteristics),
- composite score / Overall Rank and Friedman test,
- roof attenuation analysis,
- sensitivity to λ, to the treatment of negative correlations, to Pearson vs. Spearman roofs, to the target level and to ±1 perturbations of the expert scores (CP, D, CT).

The replicated experiment takes about 1–2 hours on a laptop (MissForest dominates the run time). Results are cached in `revision_results.pkl`; delete this file to recompute.

## Reproducibility notes

- All stochastic steps use fixed seeds (MCAR masks: 1–10; illustrative single scenario: 42; Monte Carlo analysis: 2024).
- The reported results were produced with Python 3, NumPy 2.5, pandas 3.0, scikit-learn 1.9, SciPy 1.18. MissForest results may differ slightly with other scikit-learn versions.
- GAIN uses PyTorch when it is installed and otherwise falls back to `gain_numpy.py`. The results in `results/` were obtained with the NumPy implementation; running with PyTorch gives slightly different GAIN numbers. All other methods are deterministic given the seed.

## Citation

If you use this code, please cite the article above.

## License

Code: MIT License (see `LICENSE`). The dataset is subject to the terms of its original source.
