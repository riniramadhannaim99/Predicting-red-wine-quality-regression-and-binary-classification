# Predicting Red Wine Quality from Physicochemical Properties

Machine learning on red wine chemistry: predict a wine's sensory quality score (regression) and classify wines as "good" or "not good" (binary classification).

## Overview

This project uses the UCI Red Wine Quality dataset (Vinho Verde, 1,599 samples). Each wine is described by 11 lab-measured features and a quality score from 0 to 10 given by expert tasters.

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

The same data is framed two ways:

| Task | Target | Models | Metrics |
|------|--------|--------|---------|
| Regression | Quality score (0-10) | Linear Regression, Random Forest, Gradient Boosting | RMSE, MAE, R² |
| Binary classification | Good (quality ≥ 7) vs. not good | Logistic Regression, SVM, Random Forest | Accuracy, Precision, Recall, F1, ROC-AUC |

## Workflow

1. Exploratory data analysis (distributions, correlations)
2. Feature scaling
3. Handling class imbalance (the "good" class is a minority)
4. Train/test split with cross-validation
5. Hyperparameter tuning
6. Model comparison and feature-importance analysis

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Typical dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.

## Usage

```bash
jupyter notebook
```

Open the notebook and run the cells in order. Dataset: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (`winequality-red.csv`).

## Results

_Fill in with your own results._

**Regression**

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | – | – | – |
| Random Forest | – | – | – |
| Gradient Boosting | – | – | – |

**Classification**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | – | – | – | – | – |
| SVM | – | – | – | – | – |
| Random Forest | – | – | – | – | – |

## Key Findings

- Alcohol, volatile acidity, and sulphates are typically the strongest predictors of quality.
- Predicting an exact score is harder than a simple good/bad decision.

## Project Structure

```
├── data/          # winequality-red.csv
├── notebooks/     # analysis and modeling
├── requirements.txt
└── README.md
```

## License

MIT
