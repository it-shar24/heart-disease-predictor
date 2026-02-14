#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Risk Prediction")
st.write("Enter patient details to assess heart disease risk")

# ------------------ Load Pipeline Safely ------------------
try:
    pipeline = joblib.load("heart_disease_pipeline.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please check 'heart_disease_pipeline.pkl'")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------ Inputs ------------------
try:
    age = st.number_input("Age", min_value=1, max_value=120)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "0 - Asymptomatic",
            "1 - Typical Angina",
            "2 - Atypical Angina",
            "3 - Non-anginal Pain"
        ]
    )
    cp_val = int(cp[0])

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600)

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs_val = 1 if fbs == "Yes" else 0

    restecg = st.selectbox(
        "Resting ECG Results",
        [
            "0 - Normal",
            "1 - ST-T Wave Abnormality",
            "2 - Left Ventricular Hypertrophy"
        ]
    )
    restecg_val = int(restecg[0])

    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0

    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0)

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "0 - Upsloping",
            "1 - Flat",
            "2 - Downsloping"
        ]
    )
    slope_val = int(slope[0])

    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

    thal = st.selectbox(
        "Thalassemia",
        [
            "1 - Normal",
            "2 - Fixed Defect",
            "3 - Reversible Defect"
        ]
    )
    thal_val = int(thal[0])

except Exception as e:
    st.error(f"Error while taking inputs: {e}")
    st.stop()

# ------------------ Prediction ------------------
if st.button("Predict Risk"):
    try:
        input_data = np.array([[
            age,
            sex_val,
            cp_val,
            trestbps,
            chol,
            fbs_val,
            restecg_val,
            thalach,
            exang_val,
            oldpeak,
            slope_val,
            ca,
            thal_val
        ]])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        result = "High Risk" if prediction == 1 else "Low Risk"

        if prediction == 1:
            st.error(f"High Risk of Heart Disease\n\nRisk Probability: {probability:.2f}")
        else:
            st.success(f"Low Risk of Heart Disease\n\nRisk Probability: {probability:.2f}")

        # ------------------ Save to Excel Safely ------------------
        data = {
            "Age": age,
            "Sex": sex,
            "Chest Pain Type": cp,
            "Resting BP": trestbps,
            "Cholesterol": chol,
            "FBS": fbs,
            "Rest ECG": restecg,
            "Max Heart Rate": thalach,
            "Exercise Angina": exang,
            "Oldpeak": oldpeak,
            "Slope": slope,
            "CA": ca,
            "Thal": thal,
            "Prediction": result,
            "Risk Probability": round(probability, 2)
        }

        file_name = "heart_predictions.xlsx"

        if os.path.exists(file_name):
            df_existing = pd.read_excel(file_name)
            df_final = pd.concat([df_existing, pd.DataFrame([data])], ignore_index=True)
            df_final.to_excel(file_name, index=False)
        else:
            pd.DataFrame([data]).to_excel(file_name, index=False)

        st.info("Patient data saved successfully")

    except ValueError as ve:
        st.error(f"Invalid input value: {ve}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# In[ ]:




