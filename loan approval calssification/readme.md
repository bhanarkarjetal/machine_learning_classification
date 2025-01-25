# Loan Approval Classification

This project focuses on building machine learning models to classify loan applications as either **Approved** (`1`) or **Not Approved** (`0`) based on various demographic, financial, and loan-related features.

---

## Dataset Overview

The dataset contains **45000 rows** and **14 columns**:
- **13 Features**:
  - `person_age`: Person's age (int)
  - `person_gender`: Gender (`Male` or `Female`) (object)
  - `person_education`: Highest education level (`High School`, `Bachelor`, `Master`, `Associate`, `Doctorate`) (object)
  - `person_income`: Person's income (float)
  - `person_emp_exp`: Employment experience in years (int)
  - `person_home_ownership`: Homeownership status (`Own`, `Rent`, `Mortgage`, `Other`) (object)
  - `loan_amnt`: Loan amount (float)
  - `loan_intent`: Loan intention (`Education`, `Medical`, `Venture`, `Personal`, `Debt Consolidation`, `Home Improvement`) (object)
  - `loan_int_rate`: Interest rate on loan (float)
  - `loan_percent_income`: Loan-to-income ratio (float)
  - `cb_person_cred_hist_length`: Credit history length in years (float)
  - `credit_score`: Credit score (float)
  - `previous_loan_defaults_on_file`: Previous loan defaults (`Yes` or `No`) (object)
- **1 Target Variable**:
  - `loan_status`: Loan status (`1` = Approved, `0` = Not Approved) (int)

The dataset is imbalanced, with significantly more approved than not-approved applications.

---

## Objective

The goal of this project is to fit the dataset into three classification models and evaluate their performance:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

The dataset is split into training and testing sets using **Stratified Shuffle Split** to ensure balanced representation in both sets.

---

## Key Steps

1. **Data Preprocessing**:
   - Handling categorical variables using one-hot encoding.
   - Normalization/standardization of numerical features.
   - Addressing class imbalance using appropriate strategies.

2. **Model Training**:
   - Logistic Regression
   - Decision Tree Classifier
     - Optimized using GridSearchCV for parameters: `max_depth` and `max_features`.
   - Random Forest Classifier
     - Optimized using GridSearchCV for parameters: `n_estimators`, `max_depth`, and `max_features`.

3. **Performance Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC Score

4. **Visualizations**:
   - Confusion matrices for all models.
   - ROC-AUC curves to compare classifier performance.

---

## Results Summary

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 89.6%    | 78.3%     | 73.8%  | 76.0%    | 0.95    |
| Decision Tree          | 91.9%    | 87.7%     | 73.9%  | 80.2%    | 0.96    |
| Random Forest          | 91.7%    | 90.2%     | 70.5%  | 79.2%    | 0.97    |

### Observations:
- **Decision Tree**: 
  - Best accuracy at **91.9%** with `max_depth = 10` and `max_features = 10`.
  - Higher precision than recall, indicating more `not approved` cases were misclassified.
- **Random Forest**:
  - Similar accuracy as Decision Tree (**91.7%**) with slightly better precision.
  - Computationally intensive compared to Decision Tree.
- **Logistic Regression**:
  - Lowest accuracy (**89.6%**) but balanced precision and recall.
  - Least computational time among all models.

---

## Feature Importance Analysis

- Top 5 features for **Decision Tree** and **Random Forest**:
  - `previous_loan_default_on_file`
  - `loan_percent_income`
  - `loan_int_rate`
  - `person_income`
  - `person_home_ownership`

- Logistic Regression shows different feature importance due to linear separability assumptions.
  - `loan_percent_income`
  - `loan_int_rate`
  - `person_homw_ownership`
  - `person_age`
  - `person_income`

---

## Model Selection Guide

- **Random Forest**: Best for high performance, if computational time is not a concern.
- **Decision Tree**: Optimal balance between performance and simplicity.
- **Logistic Regression**: Suitable for faster results with acceptable performance.

---

## Conclusion

Based on the metric analysis:
- **Decision Tree Classifier** and **Random Forest Classifier** are the most suitable for this project, with Random Forest showing slightly better precision and ROC-AUC scores.
- Computational cost is the trade-off: Decision Tree is faster, while Random Forest is more resource-intensive.

---

## Future Enhancements

1. Apply advanced techniques to handle class imbalance, such as SMOTE or ensemble methods.
2. Experiment with other algorithms like Gradient Boosting (XGBoost, LightGBM).
3. Hyperparameter optimization using Bayesian Search for better efficiency.

---
