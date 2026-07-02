# Employee Attrition Predictor

An end-to-end machine learning project that predicts whether an employee is likely to leave a company, deployed as an interactive web app.

**Live Demo:** [ibm-attrition-predictor.streamlit.app](https://ibm-attrition-predictor.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Employee attrition costs companies significant time and money in rehiring and training. This project builds a classification model that flags employees at risk of leaving, based on factors like overtime, income, job satisfaction, and tenure — enabling HR teams to act proactively rather than reactively.

The project covers the full ML lifecycle: exploratory data analysis, feature engineering, handling class imbalance, model comparison, evaluation, and deployment as a live web application.

## Dataset

- **Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset (publicly available on Kaggle)
- **Size:** 1,470 employees, 35 features
- **Target:** `Attrition` (Yes/No) — imbalanced (~16% attrition rate)

> This is an independent project built using IBM's publicly available dataset. It is not affiliated with or endorsed by IBM.

## Approach

1. **EDA** — explored distributions and attrition patterns across department, overtime, income, and satisfaction metrics
2. **Preprocessing** — encoded categorical variables (label + one-hot encoding), removed constant/unhelpful columns
3. **Feature Engineering** — created `TenureRatio`, `PromotionGap`, `SatisfactionScore`, and `IncomePerYear` to capture derived signal
4. **Class Imbalance** — applied **SMOTE** on the training set only (to avoid data leakage into evaluation)
5. **Modeling** — trained and compared 4 classifiers using 5-fold stratified cross-validation
6. **Evaluation** — compared models on Accuracy, F1, Precision, Recall, and ROC-AUC
7. **Deployment** — serialized the best model with `joblib` and built an interactive prediction dashboard with **Streamlit**

## Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **86.39%** | **47.37%** | 62.07% | 38.30% | **82.91%** |
| Gradient Boosting | 84.69% | 44.44% | 52.94% | 38.30% | 78.41% |
| Random Forest | 83.67% | 35.14% | 48.15% | 27.66% | 77.30% |
| Decision Tree | 76.87% | 34.62% | 31.58% | 38.30% | 71.35% |

**Logistic Regression** was selected as the final model — it had the best F1-score and ROC-AUC, and its coefficients also give directly interpretable insight into which factors drive attrition (important for HR stakeholders).

## Key Insights

- Employees working **overtime** are significantly more likely to leave
- **Low job satisfaction** and **low work-life balance** scores are strong attrition signals
- Employees with **low income relative to their role/experience** show higher attrition risk
- Attrition is highest among employees with **shorter tenure** and **fewer years in their current role**

## Tech Stack

- **Language:** Python
- **Data & ML:** pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit
- **Model Persistence:** joblib

## Project Structure

```
attrition-predictor/
├── app.py                                   # Streamlit web app
├── IBM_Attrition.ipynb                      # Full analysis: EDA, modeling, evaluation
├── model.pkl                                # Trained Logistic Regression model
├── scaler.pkl                                # Fitted StandardScaler
├── columns.pkl                               # Feature column order used at inference
├── WA_Fn-UseC_-HR-Employee-Attrition.csv     # Dataset
├── requirements.txt
└── README.md
```

## Running Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/IBM-hr-attrition-predictor.git
cd IBM-hr-attrition-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Future Improvements

- Add SHAP-based explainability for individual predictions
- Experiment with threshold tuning to improve recall on the minority (attrition) class
- Add a model retraining pipeline with newer data
- Containerize with Docker for easier deployment

## License

This project is licensed under the MIT License.
