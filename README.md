# Finding Donors for CharityML — Supervised Learning Project

Udacity Machine Learning Engineer / Data Scientist Nanodegree — Supervised Learning (cd0025)

## Project Overview

This project applies supervised learning techniques to census income data to help **CharityML** identify individuals most likely to donate (those earning more than $50,000/year). The solution is implemented in `project_solution/finding_donors.ipynb` using Python 3.12 and scikit-learn.

## Dataset

- **Source:** 1994 U.S. Census data via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income)
- **File:** `project_solution/census.csv`
- **Records:** 45,222 individuals, 13 features + 1 binary target label (`income`: `<=50K` or `>50K`)
- **Class distribution:** ~24.8% earn more than $50,000

## Project Structure

```
project_solution/
├── finding_donors.ipynb   # Main solution notebook
├── census.csv             # Training data
├── test_census.csv        # Test data
├── visuals.py             # Visualization helper functions
└── example_submission.csv

starter/                   # Original unmodified starter files
```

## What Was Done

### Data Exploration
- Computed total records, class counts, and the percentage of high earners (~24.8%)

### Data Preprocessing
- Applied log transformation to skewed features (`capital-gain`, `capital-loss`)
- Normalized numerical features using `MinMaxScaler`
- One-hot encoded categorical features (103 total features after encoding)
- Encoded the target label (`>50K` → 1, `<=50K` → 0)
- Split data 80/20 into training (36,177) and test (9,045) sets

### Naive Predictor Benchmark
- Established baseline by predicting all individuals earn >$50K
- Naive accuracy: **0.2478**, Naive F₀.₅ score: **0.2917**

### Model Evaluation
Three supervised learning models were evaluated at 1%, 10%, and 100% of training data:

| Model | Test Accuracy (100%) | Test F₀.₅ Score (100%) |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.69 |
| Random Forest | ~0.85 | ~0.71 |
| **Gradient Boosting** | **~0.87** | **~0.74** |

**Gradient Boosting** was selected as the best model based on highest F₀.₅ score and practical training/prediction speed.

### Model Tuning
- Grid search over `n_estimators`, `max_depth`, and `min_samples_split`
- **Optimized model:** Accuracy **0.8683**, F₀.₅ score **0.7464**
- **Unoptimized model:** Accuracy **0.8630**, F₀.₅ score **0.7395**

### Feature Importance
Top 5 most important features (from Gradient Boosting):
1. `marital-status_Married-civ-spouse`
2. `capital-gain`
3. `education-num`
4. `capital-loss`
5. `age`

### Feature Selection (Reduced Model)
Trained optimized model using only the top 5 features:
- Reduced accuracy: **0.8585** (−0.0098)
- Reduced F₀.₅ score: **0.7241** (−0.0223)

The full-feature model is preferred when training time is not a constraint.

## Requirements

- Python 3.12
- numpy, pandas, scikit-learn, matplotlib, IPython

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
