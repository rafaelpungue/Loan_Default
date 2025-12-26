# Loan Default Prediction

## Project Overview
This project aims to predict whether a loan applicant will default or not using machine learning techniques.  
The main challenge of this problem is **class imbalance**, where default cases are much fewer than non-default cases.

The project focuses on building reliable classification models while avoiding **data leakage** and ensuring realistic evaluation.

---

## Dataset
- The dataset contains historical loan application data.
- Target variable: **Loan Status (Default / Non-Default)**
- The target variable is **imbalanced**, which makes accuracy an unreliable metric.

---

##  Data Preprocessing
- Separated features (`X`) and target (`y`)
- Used **stratified train-test split** to preserve class distribution
- Applied feature scaling where required 

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"# Loan_Default" 
