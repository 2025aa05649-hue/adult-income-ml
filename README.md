# Adult Income Classification using Machine Learning

## **a. Problem Statement**

The goal of this project is to predict whether a person earns more than $50,000 per year based on demographic and employment-related attributes such as age, education, occupation, work class, and hours worked per week. This is a binary classification problem where accurate prediction can help in socio-economic analysis and policy planning.

## **b. Dataset Description**

The dataset used in this project is the **Adult Income dataset** from the UCI Machine Learning Repository (Kaggle version). It contains more than **48,000 records** and **14 attributes** describing individuals, including age, education, marital status, occupation, race, gender, hours worked per week, and capital gain/loss.

The target variable is **`income`**, which has two classes:

* `<=50K` → income less than or equal to $50,000
* `>50K` → income greater than $50,000

The dataset satisfies the assignment requirement of having more than **500 instances** and more than **12 features**.

## **c. Models Used and Performance Comparison**

Six machine learning classification models were trained and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model                 | Accuracy     | AUC          | Precision    | Recall       | F1 Score     | MCC          |
| ------------------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Logistic Regression      | 0.851264     | 0.905249     | 0.742213     | 0.586837     | 0.655442     | 0.568558     |
| Decision Tree            | 0.818405     | 0.750716     | 0.624305     | 0.619533     | 0.621910     | 0.502425     |
| K-Nearest Neighbors      | 0.817586     | 0.830759     | 0.642750     | 0.547771     | 0.591472     | 0.477526     |
| Naive Bayes              | 0.598424     | 0.848864     | 0.368984     | 0.937580     | 0.529560     | 0.376022     |
| Random Forest (Ensemble) | 0.852697     | 0.900382     | 0.734151     | 0.609766     | 0.666203     | 0.576798     |
| XGBoost (Ensemble)       | **0.872351** | **0.926159** | **0.782942** | **0.650955** | **0.710874** | **0.634277** |

## **d. Model Observations**

| ML Model            | Observation about Model Performance                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Provides strong baseline performance with high AUC, indicating good class separability.                           |
| Decision Tree       | Performs reasonably well but is prone to overfitting, resulting in lower AUC.                                     |
| K-Nearest Neighbors | Achieves moderate accuracy but is sensitive to feature scaling and noise.                                         |
| Naive Bayes         | Produces very high recall but low precision, meaning it predicts many false positives.                            |
| Random Forest       | Offers better stability and accuracy by combining multiple decision trees.                                        |
| XGBoost             | Achieves the **best overall performance** across all metrics, making it the most suitable model for this dataset. |


If you want, send me your **Streamlit live link** and I will verify it before you upload your final PDF 🚀
