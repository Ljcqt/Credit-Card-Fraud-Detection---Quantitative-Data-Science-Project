# Credit Card Fraud Detection - Quantitative Data Science Project


## Project Overview
This project focuses on identifying fraudulent credit card transactions within a highly imbalanced dataset (0.13% fraud rate) using machine learning. Based on a representative sample of 4.4 million transactions from the IBM Credit Card Dataset (24 million lines), this repository demonstrates a quantitative Data Science pipeline tailored for financial risk management.

The primary objective is to solve the core business trade-off in fraud detection: maximizing fraud interception while strictly controlling the cost of false positives.

## Key Technical Highlights
* **Large-Scale Data Optimization:** Reduced memory footprint via downcasting (`float32`, `int8`, `category`), allowing local execution on 4.4 million rows.
* **Behavioral Feature Engineering:** Engineered predictive features such as `Amt_Mean_7d` (7-day rolling average) and `Amount_vs_Habit_Ratio` to capture deviations from normal customer behavior.
* **Prevention of Data Leakage:** Implemented a strictly chronological Train/Test Split (80% past / 20% future) to simulate a production environment.
* **Cost-Sensitive Learning:** Utilized `class_weight='balanced_subsample'` to force the models to penalize missed frauds.
* **Threshold Optimization:** Analyzed the Precision-Recall curve to find the optimal operational probability threshold, reducing false alerts.

## Modeling & Results
Two models were evaluated to demonstrate the impact of linear versus non-linear approaches on highly imbalanced data:

1. **Baseline (Logistic Regression):**
   * Intercepted 417 frauds but generated 168,482 false positives. Operationally unviable.
2. **Champion Model (Random Forest Classifier):**
   * Leveraged behavioral features to reduce noise.
   * **Threshold Optimization:** By adjusting the decision threshold to 0.85, the model reduced false positives to 5,762 (out of ~896k test transactions), representing a viable business compromise.

## Repository Structure
```text
├── data/                   # (Ignored in git) Raw and processed datasets
├── notebooks/              # Jupyter notebooks for the project
│   ├── 01_EDA_et_Visualisation ipynb        # Data wrangling, optimization and exploration
│   └── 02_Modelisation.ipynb  # Feature engineering, training and threshold optimization
├── src/                    # Reusable Python scripts
│   └── sample_data.py            # Helper functions
├── requirements.txt        # Project dependencies
└── README.md