import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.title("Income Prediction Classification App")

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Naive Bayes", "KNN", "Random Forest", "XGBoost"]
)

model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "KNN": "model/KNN.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    model = joblib.load(model_files[model_choice])
    predictions = model.predict(data)

    st.write("Predictions:", predictions)
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, predictions))
    st.write("Classification Report")
    st.text(classification_report(y_test, predictions))
