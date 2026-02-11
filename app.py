import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.title("Income Prediction Classification App")

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    model = joblib.load(f,"model/{model_choice}.pkl")
    predictions = model.predict(data)

    st.write("Predictions:", predictions)
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, predictions))
    st.write("Classification Report")
    st.text(classification_report(y_test, predictions))
