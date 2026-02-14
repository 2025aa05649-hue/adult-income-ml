# ML Assignment 2 - Multiple Classifiers + Streamlit App

## a) Problem statement
Build and compare multiple machine-learning classification models on **one public dataset**, compute evaluation metrics, and demonstrate the models in an interactive **Streamlit** web application.  
The app supports uploading test CSV data, selecting a model, viewing evaluation metrics, and visualizing a confusion matrix/classification report.

---

## b) Dataset description
**Dataset:** *Adult Income Dataset* (UCI Machine Learning Repository).  
- **Task:** Binary classification (income >50K vs ≤50K).  
- **Size:** 48,842 instances and 14 features.  
- **Features:** Demographic and employment attributes.  
- **Target encoding:** `1` → income >50K, `0` → income ≤50K.  

---

## c) Models used + evaluation metrics
Models implemented:
- Logistic Regression  
- Decision Tree  
- kNN  
- Naive Bayes  
- Random Forest  
- XGBoost  

Metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

---

## Dependencies & Installation

### Requirements
The project depends on the following libraries:

- `pandas==2.2.0`  
- `numpy==1.26.4`  
- `scikit-learn==1.4.0`  
- `xgboost==2.0.3`  
- `matplotlib==3.8.2`  
- `seaborn==0.13.2`  
- `streamlit==1.31.0`  
- `joblib==1.3.2`  

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/username/ml-assignment2.git
cd ml-assignment2
pip install -r requirements.txt