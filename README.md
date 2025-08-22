# 🩺 Disease Prediction from Medical Data

## 📌 Project Overview
This project predicts the likelihood of **Diabetes**, **Heart Disease**, and **Breast Cancer** using structured medical datasets.  
Multiple machine learning models were trained, evaluated, and the best model was selected for each dataset based on performance metrics.

---

## 🎯 Objectives
- Predict disease risk based on patient medical data.
- Compare ML algorithms: **Logistic Regression, SVM, Random Forest, XGBoost**.
- Handle missing values and class imbalance using **imputation** and **SMOTE**.
- Evaluate with **5-fold Stratified Cross-Validation**.
- Provide options for **real-time prediction** using saved models or an API.

---

## 📂 Datasets
1. **Diabetes**: Pima Indians Diabetes dataset (CSV)  
2. **Heart Disease**: UCI Cleveland Heart Disease dataset  
3. **Breast Cancer**: Breast Cancer Wisconsin dataset (from scikit-learn)

Preprocessing steps:
- Missing value imputation  
- Scaling (for LR, SVM)  
- SMOTE for class imbalance  

---

## 🧠 Models Used
- Logistic Regression  
- SVM (RBF)  
- Random Forest  
- XGBoost  

Best model per dataset was selected based on **ROC-AUC** and **F1-score**.

---

## 📊 Results Summary

| Dataset        | Best Model            | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------|-----------------------|----------|-----------|--------|----------|---------|
| Diabetes        | Logistic Regression   | 0.760    | 0.643     | 0.739  | 0.685    | 0.837   |
| Heart Disease    | Random Forest         | 0.818    | 0.823     | 0.773  | 0.796    | 0.894   |
| Breast Cancer    | SVM                   | 0.975    | 0.978     | 0.983  | 0.980    | 0.995   |

---

## 🛠 Tech Stack
- **Language**: Python 3  
- **Libraries**: scikit-learn, pandas, numpy, xgboost, imbalanced-learn  
- **Visualization**: Matplotlib, Seaborn  
- **Optional API**: FastAPI  

---

## 📦 Installation
Clone the repository:
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
