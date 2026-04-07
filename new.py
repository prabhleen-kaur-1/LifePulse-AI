import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Model training
def train_model():
    csv_path = 'heart.csv'
    if not os.path.exists(csv_path):
        csv_path = 'heartproject/heart.csv'
    # st.write(f"Using file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # st.write("CSV file loaded successfully")
    
    X = df.drop(['target'], axis=1)
    y = df["target"]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    model.fit(x_train, y_train)
    # st.write("Model trained")

    acc = model.score(x_val, y_val)
    print(f"\n[SYSTEM] Model Accuracy: {acc * 100:.2f}%\n")

    return model

st.session_state.model = train_model()

# UI 
st.set_page_config(page_title="Heart Predictor", layout="wide")

st.title("HEART DISEASE PREDICTION SYSTEM")
st.divider()

col1, spacer, col2 = st.columns([1, 0.3, 1.2])

with col1:
    st.subheader("Patient Clinical Data")

    age = st.number_input("Age", 1, 120, 50)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
    
    cp = st.selectbox("Chest Pain Type", [0,1,2,3],
        format_func=lambda x: ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"][x])

    trestbps = st.number_input("Resting BP", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0,1],
        format_func=lambda x: "<120 mg/dl" if x==0 else "≥120 mg/dl")

    restecg = st.selectbox("Resting ECG", [0,1,2],
        format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])

    thalach = st.number_input("Max Heart Rate", 50, 220, 150)

    exang = st.selectbox("Exercise Angina", [0,1],
        format_func=lambda x: "No" if x==0 else "Yes")

    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope", [0,1,2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    ca = st.slider("Major Vessels", 0, 3, 0)

    thal = st.selectbox("Thalassemia", [0,1,2],
        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    analyze_btn = st.button("RUN DIAGNOSTIC ANALYSIS")

# PDF
def create_pdf(text):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 750, "HEART HEALTH REPORT")

    p.setFont("Courier", 10)
    y = 650

    for line in text.split("\n"):
        p.drawString(50, y, line)
        y -= 15
        if y < 50:
            p.showPage()
            y = 650

    p.save()
    buffer.seek(0)
    return buffer
    
# Report
if analyze_btn:

    feature_list = ['age','gender','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']

    input_df = pd.DataFrame([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
                              slope, ca, thal]], columns=feature_list)

    prob = st.session_state.model.predict_proba(input_df)[0][1]
    risk_percent = round(prob * 100, 2)

    cp_label = ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"][cp]
    restecg_label = ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][restecg]
    slope_label = ["Upsloping", "Flat", "Downsloping"][slope]
    thal_label = ["Normal", "Fixed Defect", "Reversible Defect"][thal]

    if risk_percent >= 70:
        status = "Very High Risk! Immediate medical attention required."
    elif risk_percent >= 40:
        status = "Moderate Risk. Lifestyle changes recommended."
    else:
        status = "Low Risk. Maintain healthy habits."

    report_text = f"""
    HEART HEALTH REPORT

    -----------------------------
    PATIENT DETAILS
    -----------------------------
    Age: {age} years
    Gender: {"Male" if gender == 0 else "Female"}
    Chest Pain Type: {cp_label}
    Resting BP: {trestbps} mm Hg
    Cholesterol: {chol} mg/dl
    Fasting Blood Sugar: {"≥120 mg/dl" if fbs == 1 else "<120 mg/dl"}
    Resting ECG: {restecg_label}
    Max Heart Rate: {thalach} bpm
    Exercise Angina: {"Yes" if exang == 1 else "No"}
    ST Depression: {oldpeak}
    Slope: {slope_label}
    Major Vessels: {ca}
    Thalassemia: {thal_label}

    -----------------------------
    DIAGNOSIS RESULT 
    -----------------------------
    Risk Percentage: {risk_percent}%

    {status}
    """
    with col2:
        st.subheader("Diagnostic Report")
        st.text(report_text)
        
        pdf = create_pdf(report_text.replace("<br>", "\n").replace("<hr>", ""))
        st.download_button("DOWNLOAD PDF", pdf, file_name="report.pdf")

else:
    with col2:
        st.info("Enter data and click analysis.")
        