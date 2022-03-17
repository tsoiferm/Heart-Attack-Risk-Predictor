import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps_final.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

svm = data["model"]

def show_predict_page():

    st.set_page_config(page_title = 'Heart Attack Risk Predictor', page_icon = "https://clipart.info/images/ccovers/1484710002Heart-PNG-clipart-min.png")

    st.title("Heart Attack Risk Prediction")

    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("                                                                                                                            ")

    with col2:    
        st.image("https://clipart.info/images/ccovers/1484710002Heart-PNG-clipart-min.png", width = 450)
    
    with col3:
        st.write("                                                                    ")

    st.write("""### We need some information to predict your risk of a heart attack.""")


    age = st.slider("Age", 25,85)
    sex = st.slider("Sex (0 = Female, 1 = Male", 0,1)
    cp = st.slider("Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic)", 0,3)
    trestbps = st.slider("Resting Blood Pressure", 85,220)
    chol = st.slider("Serum Cholestoral (mg/dl)", 115,580)
    fbs = st.slider("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", 0,1)
    restecg = st.slider("Resting Electrocardiographic Results (0 = Normal, 1 = Having ST-T Wave Abnormality, 2 = Showing Probable or Definite Left Ventricular Hypertrophy by Estes' Criteria)", 0,2)
    thalach = st.slider("Maximum Heart Rate Achieved", 65,220)
    exang = st.slider("Exercise Induced Angina (0 = No, 1 = Yes)", 0,1)
    oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0,7.0)
    slope = st.slider("Slope of the Peak Exercise ST Segment", 0,2)
    ca = st.slider("Number of Major Vessels (0-4) Colored by Flourosopy", 0,4)
    thal = st.slider("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect, 3 = Irreversable Defect)", 0,3)

    predict_risk = st.button("Predict Risk")
    if predict_risk:
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart = pd.read_csv(r"C:\Users\twsoi\OneDrive - stevens.edu\COVID Severity Predictor\heart.csv")
        columns_list = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        i = 0
        for name in columns_list:
            X[0][i] = X[0][i] / heart[name].abs().max()
            i+=1
        
        the_risk = svm.predict(X)
        st.subheader(f"The estimated risk is (0 = No/Less Chance of a Heart Attack, 1 = More Chance of a Heart Attack): {the_risk[0]}")



