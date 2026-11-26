
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartScore AI",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🎓 SmartScore AI")

st.subheader(
    "Student Performance Prediction & Personalized Study Assistant"
)

st.write(
    "An interactive machine learning system that predicts "
    "student performance and provides personalized study guidance."
)

# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv("student_exam_datasss.csv")
except FileNotFoundError:
    st.error(
        "CSV file not found. Put student_exam_datasss.csv "
        "in the same folder as this Python file."
    )
    st.stop()

df.columns = df.columns.str.strip()

# =========================================================
# COLUMN NAMES
# =========================================================

study_col = "Study Hours"
score_col = "Previous Exam Scores"
attendance_col = "Attendance Rate"
result_col = "Pass/Fail"

required_columns = [
    study_col,
    score_col,
    attendance_col,
    result_col,
]

for column in required_columns:
    if column not in df.columns:
        st.error("Column not found: " + column)
        st.write("Columns available in your CSV:")
        st.write(list(df.columns))
        st.stop()

# =========================================================
# SELECT REQUIRED DATA
# =========================================================

df = df[
    [
        study_col,
        score_col,
        attendance_col,
        result_col
    ]
].copy()

# =========================================================
# DATA CLEANING
# =========================================================

df[study_col] = pd.to_numeric(
    df[study_col],
    errors="coerce"
)

df[score_col] = pd.to_numeric(
    df[score_col],
    errors="coerce"
)

df[result_col] = (
    df[result_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "pass": 1,
        "fail": 0
    })
)

df[attendance_col] = pd.to_numeric(
    df[attendance_col],
    errors="coerce"
)

df = df.dropna()
df[result_col] = df[result_col].astype(int)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 SmartScore AI")

st.sidebar.write("Machine Learning Algorithm:")

st.sidebar.success("Logistic Regression")

st.sidebar.write("Learning Type:")

st.sidebar.info("Supervised Learning")

st.sidebar.write("Study Hours")
st.sidebar.write("Previous Exam Scores")
st.sidebar.write("Attendance Rate")

# =========================================================
# DASHBOARD
# =========================================================

st.header("📊 Student Performance Dashboard")

total_students = len(df)

passed_students = int(
    (df[result_col] == 1).sum()
)

failed_students = int(
    (df[result_col] == 0).sum()
)

pass_rate = (
    passed_students / total_students
) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👨‍🎓 Total Students", total_students)

with col2:
    st.metric("✅ Passed", passed_students)

with col3:
    st.metric("❌ Failed", failed_students)

with col4:
    st.metric("📈 Pass Rate", f"{pass_rate:.1f}%")

# =========================================================
# DATASET PREVIEW
# =========================================================

st.header("📋 Dataset Preview")

st.dataframe(
    df,
    use_container_width=True
)

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[
    [
        study_col,
        score_col,
        attendance_col
    ]
]

y = df[result_col]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =========================================================
# TEST MODEL
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")

with col2:
    st.metric("Training Records", len(X_train))

with col3:
    st.metric("Testing Records", len(X_test))

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Fail", "Pass"])
ax.set_yticklabels(["Fail", "Pass"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("📑 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Fail", "Pass"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# =========================================================
# GRAPH
# =========================================================

st.header("📈 Student Performance Visualization")

fig2, ax2 = plt.subplots()

failed = df[df[result_col] == 0]
passed = df[df[result_col] == 1]

ax2.scatter(
    failed[study_col],
    failed[score_col],
    label="Fail"
)

ax2.scatter(
    passed[study_col],
    passed[score_col],
    label="Pass"
)

ax2.set_xlabel("Study Hours")
ax2.set_ylabel("Previous Exam Score")
ax2.set_title("Study Hours vs Previous Exam Score")
ax2.legend()

st.pyplot(fig2)

# =========================================================
# STUDENT INPUT
# =========================================================

st.header("🎯 Student Performance Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Current Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

with col2:
    previous_score = st.number_input(
        "📝 Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

with col3:
    attendance_rate = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

# =========================================================
# PREDICT
# =========================================================

if st.button(
    "🔮 Predict Student Performance",
    type="primary"
):

    input_data = pd.DataFrame(
        [[
            study_hours,
            previous_score,
            attendance_rate
        ]],
        columns=[
            study_col,
            score_col,
            attendance_col
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    fail_probability = probability[0] * 100
    pass_probability = probability[1] * 100

    st.session_state["prediction"] = prediction
    st.session_state["pass_probability"] = pass_probability

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.header("📌 Prediction Result")

    if prediction == 1:
        st.success("🎉 PREDICTION: PASS")
    else:
        st.error("⚠️ PREDICTION: FAIL")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "✅ Pass Probability",
            f"{pass_probability:.2f}%"
        )

    with col2:
        st.metric(
            "❌ Fail Probability",
            f"{fail_probability:.2f}%"
        )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.subheader("🚦 Academic Risk Level")

    if previous_score >= 75:
        risk = "🟢 LOW RISK"
        st.success(risk)

    elif previous_score >= 60:
        risk = "🟡 MODERATE RISK"
        st.warning(risk)

    else:
        risk = "🔴 HIGH RISK"
        st.error(risk)

    # =====================================================
    # WHY THIS PREDICTION?
    # =====================================================

    st.header("🔍 Why Did I Get This Prediction?")

    if study_hours < 4:
        st.warning(
            "📚 Study hours are relatively low. "
            "Increasing consistent study time may improve preparation."
        )
    else:
        st.success(
            "📚 Study hours are at a reasonable level "
            "for the current model input."
        )

    if previous_score < 50:
        st.warning(
            "📝 Previous exam score is relatively low. "
            "Focus on understanding weak areas and regular practice."
        )

    elif previous_score < 75:
        st.info(
            "📝 Previous exam score is moderate. "
            "Additional revision may help improve performance."
        )

    else:
        st.success(
            "📝 Previous exam score is relatively strong."
        )

    if attendance_rate < 75:
        st.warning(
            "Attendance rate is relatively low."
            "Regular attendance can help maintain consistent learning."
        )
    else:
        st.success(
            "Attendance rate is at a reasonable level."
        )

# =========================================================
# TARGET SCORE ANALYZER
# =========================================================

st.header("🎯 Target Score Analyzer")

target_score = st.number_input(
    "Enter Your Target Score",
    min_value=1.0,
    max_value=100.0,
    value=None,
    step=1.0,
    placeholder="Enter target score"
)

if target_score is None:
    st.info("👆 Please enter your target score.")

else:
    score_difference = target_score - previous_score

    if score_difference <= 0:
        recommended_hours = study_hours

        st.success(
            "🎉 Your previous score already meets your target."
        )

    else:
        recommended_hours = max(
            1.0,
            min(
                12.0,
                study_hours + (score_difference / 10)
            )
        )

        st.info(
            f"You need approximately {score_difference:.1f} "
            f"more marks to reach your target."
        )

    st.metric(
        "Suggested Daily Study Time",
        f"{recommended_hours:.1f} hours"
    )

    # =====================================================
    # WHAT-IF SIMULATOR
    # =====================================================

    st.header("🔄 What-If Study Simulator")

    simulated_hours = st.slider(
        "Change Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=float(study_hours),
        step=0.5
    )

    simulated_data = pd.DataFrame(
        [[
            simulated_hours,
            previous_score,
            attendance_rate
        ]],
        columns=[
            study_col,
            score_col,
            attendance_col
        ]
    )

    simulated_probability = (
        model.predict_proba(simulated_data)[0][1] * 100
    )

    simulated_prediction = model.predict(simulated_data)[0]

    if simulated_prediction == 1:
        st.success(
            f"With {simulated_hours:.1f} study hours → PASS"
        )
    else:
        st.error(
            f"With {simulated_hours:.1f} study hours → FAIL"
        )

    st.metric(
        "Predicted Pass Probability",
        f"{simulated_probability:.2f}%"
    )

# =========================================================
# 7-DAY PERSONALIZED STUDY PLANNER
# =========================================================

st.header("📅 Personalized 7-Day Study Planner")

if target_score is None:
    st.info(
        "👆 Enter your Target Score above to generate "
        "your personalized 7-day study plan."
    )

else:
    daily_hours = recommended_hours

    st.success(
        f"📚 Study Hours for Your Plan: "
        f"{daily_hours:.1f} hours/day"
    )

    subjects = [
        "Weak Topic Revision",
        "Concept Learning",
        "Problem Solving",
        "Practice Questions",
        "Revision",
        "Mock Test + Error Analysis",
        "Full Revision"
    ]

    for i in range(7):
        st.write(
            f"**Day {i + 1}:** "
            f"{subjects[i]} — approximately "
            f"{daily_hours:.1f} hours"
        )

# =========================================================
# FINAL PERSONALIZED MESSAGE
# =========================================================

st.markdown("---")

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    pass_probability = st.session_state["pass_probability"]

    if prediction == 1 and pass_probability >= 75:

        st.header("🎉 YOU'RE ON THE RIGHT TRACK!")

        st.write(
            "Keep your study routine consistent, "
            "follow your weekly plan, and continue revising."
        )

        st.markdown(
            "### 📚 Keep Learning • Keep Improving"
        )

    else:

        st.header("💪 YOU CAN IMPROVE!")

        st.write(
            "Use the recommendations above, "
            "increase your preparation gradually, "
            "and track your progress regularly."
        )

        st.markdown(
            "### 📚 Plan • Practice • Improve"
        )

else:

    st.header("🎓 Start Your Performance Analysis")

    st.write(
        "Enter your study hours, previous exam score, "
        "and attendance rate above, then click the "
        "Predict Student Performance button."
    )

    st.markdown(
        "### 📚 Learn • Practice • Improve"
    )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h3>"🎓 SmartScore AI"</h3>
        <p>Student Performance Prediction & Personalized Study Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)
