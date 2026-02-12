import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scaler
from sklearn.preprocessing import StandardScaler

# Streamlit UI
st.title("BITS Pilani ML Assignment 2 - Classification Models")

st.sidebar.header("Dataset Options")
use_default = st.sidebar.checkbox("Use Adult Income Dataset (UCI)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])

# Load dataset
if use_default:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a dataset or select the default Adult Income dataset to proceed.")
        st.stop()

st.write("### Dataset Preview")
st.write(df.head())

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)

target_column = st.sidebar.text_input("Enter Target Column Name", value="income")

if target_column and target_column in df.columns:
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if categorical
    if y.dtype == "object":
        y = pd.factorize(y)[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (important for Logistic Regression, KNN, etc.)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(np.unique(y)) == 2 else np.nan
    except:
        auc = np.nan
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write("### Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**AUC:** {auc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**MCC:** {mcc:.4f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Download button for metrics
    metrics_df = pd.DataFrame({
        "Accuracy": [acc],
        "AUC": [auc],
        "Precision": [prec],
        "Recall": [rec],
        "F1 Score": [f1],
        "MCC": [mcc]
    })
    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Metrics as CSV",
        data=csv,
        file_name="model_metrics.csv",
        mime="text/csv"
    )
else:
    st.warning("Please enter a valid target column name.")
