"""
IBM Workforce Analytics — Employee Attrition Risk Assessment
Carbon Design System styled Streamlit application.
Model artifacts: model.pkl / scaler.pkl / columns.pkl (scikit-learn 1.8.0)
Asset: bob.svg (robot illustration used in the hero banner)

Run with:
    streamlit run app.py
"""

import os
import textwrap
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sklearn

st.set_page_config(
    page_title="IBM",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE_60 = "#0f62fe"
BLUE_70 = "#0043ce"
BLUE_50 = "#4589ff"
BLACK_100 = "#161616"
GRAY_100 = "#161616"
GRAY_80 = "#393939"
GRAY_70 = "#525252"
GRAY_30 = "#c6c6c6"
GRAY_20 = "#e0e0e0"
GRAY_10 = "#f4f4f4"
SURFACE = "#ffffff"
RED_60 = "#da1e28"
RED_10 = "#fff1f1"
GREEN_60 = "#24a148"
GREEN_10 = "#defbe6"
AMBER = "#815c00"
AMBER_10 = "#fcf4d6"

IBM_LOGO = "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BOB_SVG_PATH = os.path.join(APP_DIR, "bob.svg")


def md(html: str):
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def load_svg(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


BOB_SVG = load_svg(BOB_SVG_PATH)

md(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    color: {GRAY_100};
}}
#MainMenu, header[data-testid="stHeader"], footer {{visibility: hidden;}}
.stApp {{ background-color: {GRAY_10}; }}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}}

.ibm-nav {{
    background-color: {BLACK_100};
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 28px;
}}
.ibm-nav img {{ height: 24px; }}
.ibm-nav-divider {{ width: 1px; height: 24px; background-color: {GRAY_80}; }}
.ibm-nav-title {{ color: {SURFACE}; font-size: 15px; font-weight: 500; }}
.ibm-nav-tag {{
    margin-left: auto; color: {GRAY_30}; font-size: 12px;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px;
    border: 1px solid {GRAY_80}; padding: 3px 10px; text-transform: uppercase;
}}

.ibm-hero {{
    background-color: {BLACK_100};
    padding: 48px 28px 40px 28px;
    margin-bottom: 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 32px;
}}
.ibm-hero-left {{ flex: 1; min-width: 0; }}
.ibm-hero-right {{ flex-shrink: 0; width: 170px; height: 170px; }}
.ibm-hero-right svg {{ width: 100%; height: 100%; }}
.ibm-eyebrow {{
    color: {BLUE_50}; font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px;
}}
.ibm-hero-title {{
    color: {SURFACE}; font-size: 38px; font-weight: 600; line-height: 1.2;
    letter-spacing: -0.5px; margin-bottom: 16px;
}}
.ibm-hero-sub {{
    color: #c6c6c6; font-size: 15px; line-height: 1.6; max-width: 700px; font-weight: 300;
}}
.ibm-hero-rule {{ width: 64px; height: 4px; background-color: {BLUE_60}; margin-top: 24px; }}

.ibm-section-heading {{
    font-size: 22px; font-weight: 600; color: {GRAY_100};
    letter-spacing: -0.2px; margin: 8px 0 4px 0;
}}
.ibm-section-desc {{ color: {GRAY_70}; font-size: 14px; margin: 4px 0 24px 0; }}

.ibm-tile {{
    background-color: {SURFACE};
    border: 1px solid {GRAY_20};
    border-top: 3px solid {BLUE_60};
    padding: 24px 28px 20px 28px;
    margin-bottom: 24px;
}}
.ibm-tile-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
    letter-spacing: 1.2px; text-transform: uppercase; color: {GRAY_70};
    margin-bottom: 18px; border-bottom: 1px solid {GRAY_20}; padding-bottom: 12px;
}}

.kpi-card {{
    background-color: {SURFACE};
    border: 1px solid {GRAY_20};
    border-left: 3px solid {BLUE_60};
    padding: 20px 22px;
    height: 100%;
}}
.kpi-value {{
    font-size: 32px; font-weight: 600; color: {GRAY_100};
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}}
.kpi-label {{
    font-size: 12px; font-weight: 500; color: {GRAY_70};
    text-transform: uppercase; letter-spacing: 0.6px; margin-top: 8px;
}}
.kpi-sub {{ font-size: 12px; color: {GRAY_70}; margin-top: 6px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {GRAY_20}; }}
.stTabs [data-baseweb="tab"] {{
    height: 44px; font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase;
    color: {GRAY_70}; background-color: transparent; border-radius: 0; padding: 0 20px;
}}
.stTabs [aria-selected="true"] {{
    color: {BLUE_60} !important; border-bottom: 2px solid {BLUE_60} !important;
    background-color: transparent !important;
}}

.stSlider label, .stSelectbox label, .stNumberInput label,
div[data-testid="stWidgetLabel"] label p {{
    font-size: 13px !important; font-weight: 500 !important; color: {GRAY_100} !important;
}}

div[data-testid="stSlider"] > div > div > div > div {{
    background-color: {BLUE_60} !important;
}}
div[data-baseweb="slider"] [role="slider"] {{
    background-color: {BLUE_60} !important;
    border-color: {BLUE_60} !important;
}}

div.stButton > button, div.stFormSubmitButton > button {{
    background-color: {BLUE_60}; color: {SURFACE}; border-radius: 0; border: none;
    padding: 0.75rem 2rem; font-weight: 500; font-size: 14px;
    letter-spacing: 0.2px; width: 100%; transition: background-color 0.15s ease;
}}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
    background-color: {BLUE_70}; color: {SURFACE};
}}

.result-tile {{ border: 1px solid {GRAY_20}; padding: 28px 32px; background-color: {SURFACE}; }}
.result-tile-high {{ border-left: 4px solid {RED_60}; }}
.result-tile-medium {{ border-left: 4px solid {AMBER}; }}
.result-tile-low {{ border-left: 4px solid {GREEN_60}; }}
.result-status-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
    letter-spacing: 1.2px; text-transform: uppercase; color: {GRAY_70};
}}
.result-status-value {{ font-size: 28px; font-weight: 600; color: {GRAY_100}; margin: 6px 0 10px 0; }}
.result-status-desc {{ font-size: 14px; color: {GRAY_70}; line-height: 1.5; }}

.metric-block {{ background-color: {GRAY_10}; padding: 18px 20px; border-left: 3px solid {GRAY_30}; }}
.metric-block .metric-value {{
    font-size: 30px; font-weight: 600; color: {GRAY_100}; font-family: 'IBM Plex Mono', monospace;
}}
.metric-block .metric-label {{
    font-size: 11px; font-weight: 500; color: {GRAY_70}; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 4px;
}}

.factor-row {{ display: flex; align-items: flex-start; gap: 14px; padding: 14px 0; border-bottom: 1px solid {GRAY_20}; }}
.factor-row:last-child {{ border-bottom: none; }}
.factor-tag {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600;
    letter-spacing: 0.6px; text-transform: uppercase; padding: 3px 8px;
    color: {SURFACE}; white-space: nowrap; margin-top: 1px;
}}
.factor-tag-high {{ background-color: {RED_60}; }}
.factor-tag-medium {{ background-color: {AMBER}; }}
.factor-text {{ font-size: 14px; color: {GRAY_100}; line-height: 1.5; }}
.factor-text b {{ font-weight: 600; }}

.ibm-footer {{
    background-color: {BLACK_100};
    padding: 28px 28px 24px 28px;
    margin-top: 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.ibm-footer img {{ height: 18px; opacity: 0.85; }}
.ibm-footer-text {{ color: {GRAY_30}; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }}

.stExpander {{ border: 1px solid {GRAY_20} !important; border-radius: 0 !important; background-color: {SURFACE} !important; }}
</style>
""")

md(f"""
<div class="ibm-nav">
  <img src="{IBM_LOGO}" />
  <div class="ibm-nav-divider"></div>
  <div class="ibm-nav-title">Workforce Analytics</div>
  <div class="ibm-nav-tag">Internal Tool · v1.0</div>
</div>
""")

hero_bob_html = f'<div class="ibm-hero-right">{BOB_SVG}</div>' if BOB_SVG else ""

md(f"""
<div class="ibm-hero">
  <div class="ibm-hero-left">
    <div class="ibm-eyebrow">Predictive HR Analytics</div>
    <div class="ibm-hero-title">Employee Attrition Risk Assessment</div>
    <div class="ibm-hero-sub">
      A machine learning model trained on IBM's HR Employee Attrition dataset
      estimates the likelihood that an employee will leave the organization,
      based on role, compensation, tenure, and satisfaction signals.
    </div>
    <div class="ibm-hero-rule"></div>
  </div>
  {hero_bob_html}
</div>
""")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model.pkl / scaler.pkl / columns.pkl.\n\n"
        f"Installed scikit-learn version: {sklearn.__version__}. "
        "Ensure your environment matches requirements.txt.\n\n"
        "Run: pip install -r requirements.txt"
    )
    st.exception(e)
    st.stop()

NOMINAL_COLS = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]


def build_feature_row(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df["OverTime"] = df["OverTime"].map({"No": 0, "Yes": 1})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # NOTE: pd.get_dummies on a single-row DataFrame breaks with drop_first=True —
    # the one category present always gets treated as the "first" and dropped,
    # so every dummy column ends up 0 regardless of what the user selected.
    # Build the dummy columns manually instead.
    for col in NOMINAL_COLS:
        df[f"{col}_{raw[col]}"] = 1
    df = df.drop(columns=NOMINAL_COLS)

    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionGap"] = df["YearsSinceLastPromotion"] - df["YearsInCurrentRole"]
    df["SatisfactionScore"] = (
        df["JobSatisfaction"] + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"] + df["WorkLifeBalance"]
    ) / 4
    df["IncomePerYear"] = df["MonthlyIncome"] * 1
    df = df.reindex(columns=model_columns, fill_value=0)
    return df


def gauge_chart(probability: float):
    if probability < 0.30:
        color = GREEN_60
    elif probability < 0.60:
        color = AMBER
    else:
        color = RED_60

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": GRAY_100, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRAY_70},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": SURFACE,
                "borderwidth": 1,
                "bordercolor": GRAY_20,
                "steps": [
                    {"range": [0, 30], "color": GREEN_10},
                    {"range": [30, 60], "color": AMBER_10},
                    {"range": [60, 100], "color": RED_10},
                ],
            },
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "IBM Plex Sans"},
    )
    return fig


def bar_chart(labels, values, title, color=BLUE_60):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=38 * len(labels) + 60,
        margin=dict(l=10, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans", "color": GRAY_100, "size": 12},
        xaxis=dict(showgrid=False, visible=False, range=[0, max(values) * 1.25]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        title=dict(text=title, font=dict(size=13, family="IBM Plex Mono", color=GRAY_70)),
        showlegend=False,
    )
    return fig


TOTAL_EMPLOYEES = 1470
ATTRITION_COUNT = 237
ATTRITION_RATE = 16.1
AVG_INCOME = 6503
AVG_AGE = 36.9
AVG_TENURE = 7.0

DEPT_LABELS = ["Sales", "Human Resources", "Research & Development"]
DEPT_RATES = [20.6, 19.0, 13.8]

ROLE_LABELS = ["Sales Representative", "Laboratory Technician", "Human Resources",
               "Sales Executive", "Research Scientist"]
ROLE_RATES = [39.8, 23.9, 23.1, 17.5, 16.1]

OVERTIME_LABELS = ["OverTime: Yes", "OverTime: No"]
OVERTIME_RATES = [30.5, 10.4]

with st.sidebar:
    st.image(IBM_LOGO, width=80)
    st.markdown("---")
    st.markdown("**ABOUT THIS TOOL**")
    st.markdown("Estimates attrition probability for a single employee profile using a model trained on IBM's HR Employee Attrition dataset.")
    st.markdown("---")
    st.markdown("**MODEL PIPELINE**")
    st.markdown("- SMOTE class balancing\n- StandardScaler feature scaling\n- Logistic Regression Classifier\n- 86.4% accuracy · 0.829 ROC-AUC")
    st.markdown("---")
    st.caption("IBM HR Analytics — internal demo application.")

md("""
<div class="ibm-section-heading">Employee Profile</div>
<div class="ibm-section-desc">Enter the employee's details to generate a risk assessment.</div>
""")

with st.form("attrition_form"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["PERSONAL", "JOB & ROLE", "COMPENSATION", "SATISFACTION & TENURE", "WORKFORCE OVERVIEW"]
    )

    with tab1:
        md('<div class="ibm-tile"><div class="ibm-tile-label">Demographic Details</div>')
        c1, c2 = st.columns(2)
        with c1:
            Age = st.slider("Age", 18, 60, 35)
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            DistanceFromHome = st.slider("Distance From Home (km)", 1, 29, 9)
        with c2:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Education = st.selectbox(
                "Education Level", [1, 2, 3, 4, 5],
                format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}[x],
                index=2,
            )
            EducationField = st.selectbox(
                "Education Field",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
            )
        md("</div>")

    with tab2:
        md('<div class="ibm-tile"><div class="ibm-tile-label">Role &amp; Employment</div>')
        c1, c2 = st.columns(2)
        with c1:
            Department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            JobRole = st.selectbox(
                "Job Role",
                ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                 "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
            )
            JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
        with c2:
            BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            OverTime = st.selectbox("OverTime", ["No", "Yes"])
            NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 9, 2)
        md("</div>")

    with tab3:
        md('<div class="ibm-tile"><div class="ibm-tile-label">Pay &amp; Compensation</div>')
        c1, c2 = st.columns(2)
        with c1:
            MonthlyIncome = st.number_input("Monthly Income ($)", 1009, 19999, 6500, step=100)
            DailyRate = st.slider("Daily Rate", 102, 1499, 800)
            HourlyRate = st.slider("Hourly Rate", 30, 100, 65)
        with c2:
            MonthlyRate = st.slider("Monthly Rate", 2094, 26999, 14300)
            PercentSalaryHike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
            StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)
        md("</div>")

    with tab4:
        md('<div class="ibm-tile"><div class="ibm-tile-label">Satisfaction &amp; Engagement</div>')
        c1, c2 = st.columns(2)
        with c1:
            EnvironmentSatisfaction = st.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
            JobSatisfaction = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
        with c2:
            RelationshipSatisfaction = st.select_slider("Relationship Satisfaction", [1, 2, 3, 4], value=3)
            WorkLifeBalance = st.select_slider("Work-Life Balance", [1, 2, 3, 4], value=3)
        c3, c4 = st.columns(2)
        with c3:
            JobInvolvement = st.select_slider("Job Involvement", [1, 2, 3, 4], value=3)
        with c4:
            PerformanceRating = st.select_slider("Performance Rating", [3, 4], value=3)
        md("</div>")

        md('<div class="ibm-tile"><div class="ibm-tile-label">Tenure &amp; Career Growth</div>')
        c1, c2 = st.columns(2)
        with c1:
            TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
            YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
            YearsInCurrentRole = st.slider("Years In Current Role", 0, 18, 4)
        with c2:
            YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
            YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 4)
            TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 3)
        md("</div>")

    with tab5:
        md("""
        <div class="ibm-tile-label" style="border-bottom:none; margin-bottom:8px;">Workforce Overview</div>
        <div class="ibm-section-desc" style="margin-top:0;">Aggregate attrition trends across 1,470 employee records in the underlying dataset.</div>
        """)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            md(f'<div class="kpi-card"><div class="kpi-value">{TOTAL_EMPLOYEES:,}</div><div class="kpi-label">Total Employees</div><div class="kpi-sub">Records analyzed</div></div>')
        with k2:
            md(f'<div class="kpi-card"><div class="kpi-value">{ATTRITION_RATE}%</div><div class="kpi-label">Attrition Rate</div><div class="kpi-sub">{ATTRITION_COUNT} employees left</div></div>')
        with k3:
            md(f'<div class="kpi-card"><div class="kpi-value">${AVG_INCOME:,}</div><div class="kpi-label">Avg. Monthly Income</div><div class="kpi-sub">Across all roles</div></div>')
        with k4:
            md(f'<div class="kpi-card"><div class="kpi-value">{AVG_TENURE}</div><div class="kpi-label">Avg. Years At Company</div><div class="kpi-sub">Avg. age {AVG_AGE}</div></div>')

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            md('<div class="ibm-tile"><div class="ibm-tile-label">Attrition Rate by Department</div>')
            st.plotly_chart(bar_chart(DEPT_LABELS, DEPT_RATES, "% who left"), use_container_width=True)
            md("</div>")
        with c2:
            md('<div class="ibm-tile"><div class="ibm-tile-label">Attrition Rate by OverTime Status</div>')
            st.plotly_chart(bar_chart(OVERTIME_LABELS, OVERTIME_RATES, "% who left", color=RED_60), use_container_width=True)
            md("</div>")

        md('<div class="ibm-tile"><div class="ibm-tile-label">Highest-Risk Job Roles</div>')
        st.plotly_chart(bar_chart(ROLE_LABELS, ROLE_RATES, "% who left", color=AMBER), use_container_width=True)
        md("</div>")

    submitted = st.form_submit_button("Predict Attrition Risk")

if submitted:
    raw_input = {
        "Age": Age, "DailyRate": DailyRate, "DistanceFromHome": DistanceFromHome,
        "Education": Education, "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender, "HourlyRate": HourlyRate, "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel, "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome, "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked, "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike, "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction, "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager, "BusinessTravel": BusinessTravel,
        "Department": Department, "EducationField": EducationField, "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
    }

    try:
        features = build_feature_row(raw_input)
        scaled = scaler.transform(features)
        proba = model.predict_proba(scaled)[0][1]
        prediction = model.predict(scaled)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    md('<div class="ibm-section-heading">Risk Assessment Result</div>')

    if proba >= 0.60:
        risk_label, tile_class = "HIGH RISK", "result-tile-high"
    elif proba >= 0.30:
        risk_label, tile_class = "MEDIUM RISK", "result-tile-medium"
    else:
        risk_label, tile_class = "LOW RISK", "result-tile-low"

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        verdict = "likely to leave the organization" if prediction == 1 else "likely to remain with the organization"
        md(f"""
        <div class="result-tile {tile_class}">
          <div class="result-status-label">Risk Classification</div>
          <div class="result-status-value">{risk_label}</div>
          <div class="result-status-desc">Based on the submitted profile, this employee is <b>{verdict}</b> within the modeled time horizon.</div>
        </div>
        """)
        st.write("")
        m1, m2 = st.columns(2)
        with m1:
            md(f'<div class="metric-block"><div class="metric-value">{proba*100:.1f}%</div><div class="metric-label">Attrition Probability</div></div>')
        with m2:
            md(f'<div class="metric-block"><div class="metric-value">{(1-proba)*100:.1f}%</div><div class="metric-label">Retention Probability</div></div>')

    with col_right:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)

    md('<div class="ibm-section-heading" style="margin-top:32px;">Key Contributing Factors</div>')

    factors = []
    if OverTime == "Yes":
        factors.append(("HIGH", "OverTime", "working overtime is the strongest known driver of attrition in this dataset."))
    if MonthlyIncome < 4000:
        factors.append(("HIGH", "Low Monthly Income", "below-average pay is associated with higher attrition risk."))
    if Age < 30:
        factors.append(("MEDIUM", "Early-Career Employee", "younger employees show higher attrition rates historically."))
    if YearsSinceLastPromotion - YearsInCurrentRole > 2:
        factors.append(("MEDIUM", "Promotion Gap", "extended time since last promotion relative to current role tenure."))
    if WorkLifeBalance <= 2:
        factors.append(("MEDIUM", "Work-Life Balance", "low score correlates with increased attrition likelihood."))
    if JobSatisfaction <= 2:
        factors.append(("MEDIUM", "Job Satisfaction", "low satisfaction is a leading indicator of attrition."))
    if BusinessTravel == "Travel_Frequently":
        factors.append(("MEDIUM", "Frequent Business Travel", "adds measurable attrition risk."))
    if NumCompaniesWorked >= 5:
        factors.append(("MEDIUM", "Employment History", "high number of prior employers."))

    if factors:
        rows = ""
        for severity, label, desc in factors:
            tag_class = "factor-tag-high" if severity == "HIGH" else "factor-tag-medium"
            rows += (
                f'<div class="factor-row">'
                f'<div class="factor-tag {tag_class}">{severity}</div>'
                f'<div class="factor-text"><b>{label}</b> — {desc}</div>'
                f'</div>'
            )
        md(f'<div class="ibm-tile" style="border-top-color:{GRAY_30};">{rows}</div>')
    else:
        md(f'<div class="ibm-tile" style="border-top-color:{GREEN_60};"><div class="factor-text">No major risk factors were identified for this employee profile.</div></div>')

    with st.expander("HR RECOMMENDATIONS"):
        st.markdown(
            "- Reduce overtime workload — the strongest predictor of attrition in this model\n"
            "- Revise compensation bands for junior and lower-income employees\n"
            "- Fast-track promotions for high performers with extended promotion gaps\n"
            "- Limit frequent business travel for employees flagged at-risk\n"
            "- Establish regular manager check-ins to monitor job satisfaction\n"
            "- Track work-life balance scores as an early warning signal"
        )

md(f"""
<div class="ibm-footer">
  <img src="{IBM_LOGO}" />
  <div class="ibm-footer-text">IBM WORKFORCE ANALYTICS — INTERNAL DEMO — NOT FOR PRODUCTION HR DECISIONS</div>
</div>
""")