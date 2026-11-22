# Regression & Classification Analysis (CIE 2)

**A Supervised Machine Learning and Data Analysis Project using Python**

## Overview

This project demonstrates end-to-end regression and classification workflows for CIE 2 (Units 1 & 2) using Python. It covers exploratory data inspection, data cleaning, multicollinearity and class-imbalance checks, model training and evaluation, threshold tuning, and inference on new records.

The project uses Python-based data manipulation, statistical diagnostics, visualization, and machine learning libraries to build transparent and reproducible analytical pipelines.

## Project Objectives

- Inspect and understand dataset structures, distributions, and class balance.
- Identify missing values, domain-invalid zeros, duplicate records, and multicollinearity.
- Apply appropriate preprocessing and leak-free modeling pipelines.
- Compare baseline, linear regression, logistic regression, and ensemble models.
- Evaluate regression errors, residual diagnostics, and classification metrics (including ROC AUC and threshold trade-offs).
- Test predictions on new records with range and validity checks.

## Technology Stack

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Manipulation | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Statistical Diagnostics | Statsmodels |
| Development Environment | Jupyter Notebook / Google Colab |

## Datasets

### Activity 1 — Multiple Linear Regression (`RCA1.ipynb`)

- **Name:** College Students Habits Dataset
- **Source:** `college_students_habits_1M.csv` (1,000,000 rows; 25,000-row working sample)
- **Domain:** Academic performance prediction
- **Target column:** `gpa`

### Activity 2 — Binary Classification (`RCA 2.ipynb`)

- **Name:** Pima Indians Diabetes Dataset
- **Source:** `pima-indians-diabetes.csv` (768 rows, 8 features + target)
- **Domain:** Medical screening / binary classification
- **Target column:** `Outcome` (`0` = No Diabetes, `1` = Diabetes)

## Workflow

### 1. Data Loading & Inspection

- Loaded both datasets using Pandas and recorded library versions and fixed random seeds.
- Inspected shapes, data types, missing values, duplicates, and target distributions.
- Generated summary statistics (including ranges and units in `RCA1.ipynb`, and class counts/proportions in `RCA 2.ipynb`).

### 2. Data Cleaning & Preprocessing

- **`RCA1.ipynb`:** Extracted a reproducible working sample, selected predictors (`study_hours`, `attendance`, `sleep_hours`, `previous_gpa`, `screen_time`), and removed duplicates and missing entries.
- **`RCA 2.ipynb`:** Converted domain-impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` to `NaN`, and built leak-free `Pipeline` objects with `SimpleImputer(strategy="median")` and `StandardScaler()`.

### 3. Exploratory Data Analysis

Created visualizations and diagnostic tables to explore:

- **`RCA1.ipynb`:** Feature-vs-GPA scatter plots, correlation matrix heatmap, and Variance Inflation Factor (VIF) scores.
- **`RCA 2.ipynb`:** Target class balance bar chart, feature distributions, and ROC curves.

### 4. Model Training & Evaluation

- **`RCA1.ipynb`:** Solved a 6-row normal-equation ($\mathbf{X}^T\mathbf{X}\mathbf{b} = \mathbf{X}^T\mathbf{y}$) hand calculation verified against `LinearRegression`, and compared a mean baseline, a single-feature model (`attendance`), and a 5-feature multiple linear regression model across train and test splits ($\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$, Adjusted $R^2$) plus residual plots.
- **`RCA 2.ipynb`:** Trained and evaluated `LogisticRegression` and `RandomForestClassifier` on a stratified train/test split and via 5-fold stratified cross-validation using Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC AUC.

### 5. Threshold Tuning & Inference

- **`RCA 2.ipynb`:** Converted predicted probabilities into class labels and compared classification thresholds (`0.50` vs. `0.35`) to examine false-negative vs. false-positive trade-offs.
- **`RCA1.ipynb` & `RCA 2.ipynb`:** Generated sample predictions on new records and checked feature ranges/notes before interpreting outputs.

## Key Skills Demonstrated

- Multiple Linear Regression & Normal Equations
- Multicollinearity & Residual Diagnostics
- Binary Classification & ROC-AUC Evaluation
- Leak-Free Preprocessing Pipelines
- Decision Threshold Analysis
- Data Visualization & Reproducible Python Workflows

## Repository Structure

```text
CIE2-RCA-Regression-Classification/
│
├── README.md
├── RCA1.ipynb
├── RCA 2.ipynb
├── college_students_habits_1M.csv
└── pima-indians-diabetes.csv
```

## Getting Started

Place the dataset files (`college_students_habits_1M.csv` and `pima-indians-diabetes.csv`) in your project folder and install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `RCA1.ipynb` or `RCA 2.ipynb` and run the cells sequentially from top to bottom.

## Project Outcome

The project demonstrates complete regression and classification workflows—from data cleaning and exploratory diagnostics to model evaluation, threshold selection, and prediction checks—helping build reliable and interpretable machine learning solutions.

## Author

**Tejas Borate**

B.Tech – Artificial Intelligence & Data Science

- LinkedIn: [Tejas Borate](https://www.linkedin.com/in/tejas-borate-b90367302/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B32HyXrUMQO%2BnS3Ta5QGScg%3D%3D)