# Heart Disease Prediction — Machine Learning Model Comparison

An academic statistical learning project in **R** exploring the prediction of heart disease from clinical and demographic measurements. The analysis compares several classification algorithms and investigates whether smaller feature sets can achieve performance comparable to the full model.

**Authors:** Elena Lardies and Daniela Bialy  
**Project type:** University coursework. Bachelor in Data Science and Engineering. Universidad Carlos III de Madrid.

## Dataset

The project uses `heart.csv`, containing **918 patient records** and **12 original columns** (11 predictors and the binary `HeartDisease` outcome). Predictors include age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and ST-slope measurements.

## Methods

- Data inspection, missing-value handling, and exploratory visualizations
- K-nearest-neighbors imputation for zero-coded cholesterol values
- Filtering of implausible clinical values and construction of additional risk indicators
- Logistic regression, regularized logistic regression, linear and quadratic discriminant analysis, naive Bayes, decision trees, random forests, and k-nearest neighbors
- Comparison of full, random-forest-selected, and risk-focused feature sets using accuracy, ROC AUC, and Cohen's kappa

## Selected results

The following results reproduce the **feature-selection comparison reported in the coursework**:

| Logistic regression feature set | Accuracy | ROC AUC | Cohen's kappa |
|---|---:|---:|---:|
| Full model | 0.879 | 0.933 | 0.755 |
| Top 6 RF features | 0.868 | 0.902 | 0.735 |
| Risk-focused features | 0.846 | 0.907 | 0.691 |

The full feature set performed best in this comparison, although the smaller models retained reasonably strong discrimination.

## Files

- `PROJECT2HEART_cleaned.R` — R analysis script
- `heart.csv` — dataset
- `PROJECT2-REPORT.pdf` — coursework report and interpretation

## Running the analysis

1. Install **R** and, optionally, **RStudio**.
2. Open the folder containing `PROJECT2HEART_cleaned.R` and `heart.csv` as your working directory.
3. Open the script in RStudio. It checks for required R packages and lists any that need to be installed.
4. Run the script using **Source**. Some model-tuning steps may take time.

## Scope and limitations

This repository preserves and reproduces an academic analysis. In the original workflow, some preprocessing is performed before the train/test split and some model-selection choices may involve test-set feedback. Consequently, the reported test metrics should be treated as **coursework results**, not as an independently validated estimate of deployment performance. The project is intended for learning and portfolio demonstration, not medical decision-making.
