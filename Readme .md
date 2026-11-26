# Multiple Linear Regression Analysis — CIE 2 Activity 1 (Roll No: AD2127)

**A Supervised Regression and Root Cause Analysis Project using Python**

## Overview

This repository contains the **CIE 2 Activity 1** submission focused on multiple linear regression and root cause analysis (RCA) using Python. It covers exploratory data inspection, data cleaning, multicollinearity checks, normal-equation verification, baseline and multi-feature regression modeling, residual diagnostics, and inference on new records.

## Project Objectives

- Inspect dataset structure, feature distributions, summary statistics, ranges, and units.
- Identify missing values, duplicate records, and multicollinearity among predictors.
- Build a reproducible preprocessing and train/test evaluation workflow with fixed random seeds.
- Compare a mean baseline, a single-feature model (`attendance`), and a 5-feature multiple linear regression model.
- Evaluate regression accuracy and inspect residual diagnostics and feature relationships.
- Run sample predictions on new records with range and validity checks.

## Technology Stack

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Manipulation | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Statistical Diagnostics | Statsmodels |
| Development Environment | Jupyter Notebook / Google Colab |

## Dataset

- **Name:** College Students Habits Dataset
- **Source:** `college_students_habits_1M.csv` (1,000,000 rows; 25,000-row working sample)
- **Domain:** Academic performance prediction
- **Target Column:** `gpa`
- **Predictors Explored:** `study_hours`, `attendance`, `sleep_hours`, `previous_gpa`, and `screen_time`

## Workflow

### 1. Data Loading & Inspection
- Loaded the dataset using Pandas and recorded library versions and fixed random seeds for reproducibility.
- Inspected dataset shape, data types, missing values, duplicate rows, and target (`gpa`) distribution.
- Reviewed summary statistics including value ranges and units.

### 2. Data Cleaning & Preprocessing
- Extracted a reproducible working sample and selected candidate predictors (`study_hours`, `attendance`, `sleep_hours`, `previous_gpa`, `screen_time`).
- Handled missing entries and duplicate records cleanly prior to model training and evaluation.

### 3. Exploratory Data Analysis
- Plotted feature-vs-GPA scatter plots to inspect linearity and trends.
- Computed a correlation matrix heatmap and Variance Inflation Factor (VIF) scores to check for multicollinearity.

### 4. Model Training & Evaluation
- Solved a 6-row normal-equation ($\mathbf{X}^T\mathbf{X}\mathbf{b} = \mathbf{X}^T\mathbf{y}$) hand calculation and verified the result against Scikit-learn's `LinearRegression`.
- Compared a mean baseline, a single-feature regression model (`attendance`), and a 5-feature multiple linear regression model on train and test splits using $\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$, and Adjusted $R^2$.
- Inspected residual plots to evaluate model fit and error patterns.

### 5. Inference & Validity Checks
- Tested predictions on new sample records.
- Checked feature ranges and domain notes before interpreting predicted GPA values.

## Key Skills Demonstrated

- Multiple Linear Regression & Normal Equations
- Multicollinearity (VIF) & Residual Diagnostics
- Train/Test Evaluation ($\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$, Adjusted $R^2$)
- Data Visualization & Reproducible Python Workflows

## Repository Structure

```text
CIE2-Activity1-MLR-AD2127/
│
├── README.md
├── RCA1.ipynb
└── college_students_habits_1M.csv
