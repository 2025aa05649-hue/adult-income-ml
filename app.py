import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from model.logistic_regression import train_and_evaluate as lr
from model.decision_tree import train_and_evaluate as dt
from model.knn import train_and_evaluate as knn
from model.naive_bayes import train_and_evaluate as nb
from model.random_forest import train_and_evaluate as rf
from model.xgboost_model import train_and_evaluate as xgb

st.set_page_config(page_title="Adult Income Classification")
st.title("💼 Adult Income Classification")

# Dataset source selection
data_option = st.radio(
    "Select dataset source:",
    ("Load from UCI URL", "Upload CSV file")
)

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

# Target column input
target_column = st.text_input(
    "Target Column Name",
    value="income",
    help="Enter the name of your target/label column (default: income)"
)

df = None
if data_option == "Load from UCI URL":
    st.info("Loading Adult Income dataset from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head(10), width='stretch')

elif data_option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload Adult Income CSV (Test Data Only)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        st.dataframe(df.head(10), width='stretch')

if df is not None:
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found in dataset.")
    else:
        # Features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column].apply(lambda x: 1 if str(x).strip() in [">50K", "1"] else 0)

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.button("Run Model"):
            model_map = {
                "Logistic Regression": lr,
                "Decision Tree": dt,
                "KNN": knn,
                "Naive Bayes": nb,
                "Random Forest": rf,
                "XGBoost": xgb
            }

            y_pred, metrics = model_map[model_name](
                X_train, X_test, y_train, y_test
            )

            st.subheader("📊 Evaluation Metrics")
            metrics_df = pd.DataFrame(metrics, index=["Score"]).T
            # Round metrics to 4 decimal places for better readability
            metrics_df_display = metrics_df.round(4)
            st.table(metrics_df_display)

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC", f"{metrics['AUC']:.4f}")
            with col3:
                st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["<=50K", ">50K"],
                        yticklabels=["<=50K", ">50K"],
                        ax=ax, cbar_kws={"label": "Count"},
                        annot_kws={"size": 14, "weight": "bold"})
            ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
            ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
            st.pyplot(fig)

            st.subheader("📑 Classification Report")
            report = classification_report(y_test, y_pred,
                                           target_names=["<=50K", ">50K"],
                                           output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df_display = report_df.round(4)
            st.table(report_df_display)
