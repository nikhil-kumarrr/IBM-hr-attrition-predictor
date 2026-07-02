# Employee Attrition Prediction

A ML-powered employee attrition prediction system built for HR analytics using Logistic Regression, Decision Tree, and Random Forest/Gradient Boosting — with an interactive **AttritionSense** dashboard built in Streamlit.

Enter employee profile data and instantly get an AI-powered attrition risk prediction with probability scores and an HR action plan.

---

## Features

- ML-based employee attrition risk prediction
- 4 models trained and compared (LR · DT · RF · GB)
- Instant risk classification with probability scores
- Risk gauge meter, attrition vs retention breakdown
- HR action plan based on prediction outcome
- Professional dark UI dashboard
- Uses IBM HR Analytics dataset (1,470 employees)
- Real-time prediction engine with saved model artifacts

---

## How It Works

### 1️⃣ Dataset

**WA_Fn-UseC_-HR-Employee-Attrition.csv** — 1,470 employees across departments (Sales, R&D, HR)

| Feature | Description |
|---|---|
| Age | Employee age |
| Department | Sales / R&D / HR |
| MonthlyIncome | Monthly salary ($) |
| OverTime | Works overtime? (Yes/No) |
| JobSatisfaction | Satisfaction score (1–4) |
| WorkLifeBalance | Work-life balance score (1–4) |
| YearsAtCompany | Tenure at company |
| BusinessTravel | Travel frequency |
| Attrition | **Target** — Left (Yes) / Stayed (No) |

---

### 2️⃣ Data Processing (Notebook)

- Dropped constant/irrelevant columns → `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`
- Checked for missing values and duplicates
- Label encoding for binary features → `OverTime`, `Gender`
- One-hot encoding for nominal features → `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
- Feature engineering → `TenureRatio`, `PromotionGap`, `SatisfactionScore`, `IncomePerYear`
- Feature scaling → `StandardScaler`
- Class imbalance handling → `SMOTE` (applied on training set only)
- Train-test split → 80/20, stratified by target

---

### 3️⃣ EDA Performed

- Attrition distribution (Stayed vs Left)
- Attrition rate by Department and OverTime
- Age vs Attrition boxplot
- Correlation heatmap across all numeric features

---

### 4️⃣ ML Models

- **4 Models Trained** → Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Evaluation** → 5-Fold Stratified Cross-Validation, Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best Model** → Logistic Regression (Accuracy: 86.4% | ROC-AUC: 82.9%)
- **Saved as** → `model.pkl` + `scaler.pkl` + `columns.pkl`

---

## Model Results

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **86.39%** | **47.37%** | 62.07% | 38.30% | **82.91%** |
| Gradient Boosting | 84.69% | 44.44% | 52.94% | 38.30% | 78.41% |
| Random Forest | 83.67% | 35.14% | 48.15% | 27.66% | 77.30% |
| Decision Tree | 76.87% | 34.62% | 31.58% | 38.30% | 71.35% |

> Logistic Regression selected as best model — highest F1-score and ROC-AUC, with directly interpretable coefficients for HR stakeholders.

---

## Key Findings

- Employees working **overtime** churn significantly more
- **Low job satisfaction** and **low work-life balance** are strong attrition signals
- Employees with **lower income relative to role/experience** show higher attrition risk
- Attrition is highest among employees with **shorter tenure** and **fewer years in current role**
- **Sales Representatives** show notably higher attrition than other roles

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | EDA visualizations |
| Scikit-learn | ML models, preprocessing, evaluation |
| imbalanced-learn | SMOTE class balancing |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

---

## Project Structure
```
attrition-predictor/
│
├── app.py                                   ← Streamlit dashboard
├── IBM_Attrition.ipynb                      ← Full ML pipeline notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv     ← Raw dataset
│
├── model.pkl                                ← Trained Logistic Regression model
├── scaler.pkl                               ← StandardScaler
├── columns.pkl                              ← Feature name/order list
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/IBM-hr-attrition-predictor.git
cd IBM-hr-attrition-predictor
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the notebook first (to generate model files)
```bash
jupyter notebook IBM_Attrition.ipynb
```

### 6️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## requirements.txt
```streamlit>=1.30
pandas
numpy
scikit-learn==1.8.0
joblib
plotly
```

---

## Dataset

Available on Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?fbclid=PAT01DUASs6C1leHRuA2FlbQIxMABzcnRjBmFwcF9pZA81NjcwNjczNDMzNTI0MjcAAaflZeiHHd-NoG97EBB4u1HtOsqva0g7eZWs4oGbnziRbAjmUtwRf4VKTrwRBg_aem_DUfNl6WnKSYL9zo87vicEA

> Independent project built using IBM's publicly available dataset. Not affiliated with or endorsed by IBM.

---

## Live Demo
https://ibm-attrition-predictor.streamlit.app/

---

## Screenshots

![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-07-02%20132132.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202026-07-02%20132207.png?raw=true)
