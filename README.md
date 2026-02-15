# 📘 Machine Learning Assignment 2 – Adult Income Classification

## 1. Problem Statement
The goal of this project is to build and evaluate multiple machine learning classification models on a real-world dataset. The models are compared using standard evaluation metrics, and the best-performing model is deployed in an interactive Streamlit web application.

---

## 2. Dataset Description
- **Dataset Source:** UCI Adult Income Dataset  
- **Instances:** ~32,000 rows  
- **Features:** 14 attributes (mix of categorical and numerical)  
- **Target Variable:** `income` (binary: `<=50K` or `>50K`)  
- **Preprocessing:**  
  - Missing values handled (`?` replaced with NaN and dropped/encoded)  
  - Label encoding applied to categorical features  
  - Standard scaling applied to numerical features  

---

## 3. Models Used
We implemented the following six classification models:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 4. Comparison Table of Evaluation Metrics

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1   | MCC   |
|----------------------|----------|-------|-----------|--------|------|-------|
| Logistic Regression  | 0.8279   | 0.8608| 0.7246    | 0.4598 |0.5626|0.4806 |
| Decision Tree        | 0.8125   | 0.7532| 0.6047    | 0.6390 |0.6214|0.4973 |
| KNN                  | 0.8340   | 0.8569| 0.6711    | 0.6091 |0.6386|0.5322 |
| Naive Bayes          | 0.8081   | 0.8644| 0.7044    | 0.3495 |0.4672|0.3994 |
| Random Forest        | 0.8592   | 0.9108| 0.7406    | 0.6390 |0.6861|0.5986 |
| XGBoost              | 0.8762   | 0.9286| 0.7761    | 0.6830 |0.7266|0.6492 |

---

## 5. Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Achieved strong accuracy (82.8%) and high AUC (0.86), indicating good overall discrimination ability. However, recall (0.46) was relatively low, meaning the model missed a significant portion of positive cases. Precision (0.72) was decent, showing it was more conservative in predicting the positive class. Overall, Logistic Regression provided a balanced baseline but struggled with recall. |
| Decision Tree        | Delivered slightly lower accuracy (81.2%) compared to Logistic Regression, but recall (0.64) was higher, meaning it captured more positive cases. Precision (0.60) was moderate, showing some trade-off between false positives and false negatives. AUC (0.75) was weaker, reflecting less robust separation between classes. Overall, the Decision Tree offered better recall but at the cost of precision and generalization. |
| KNN                  | Produced high accuracy (83.4%) with a strong AUC (0.86), showing good class separation. Precision (0.67) and recall (0.61) were fairly balanced, resulting in a solid F1 score (0.64). MCC (0.53) was also strong, indicating better overall correlation between predictions and true labels. KNN demonstrated a good balance between precision and recall, making it a competitive model for this dataset. |
| Naive Bayes          | Achieved decent accuracy (80.8%) and surprisingly high AUC (0.86), showing strong separation capability. However, recall (0.35) was very low, meaning the model missed many positive cases. Precision (0.70) was solid, indicating it was conservative in predicting positives. The low F1 score (0.47) and MCC (0.39) highlight its weakness in handling complex feature interactions. Overall, Naive Bayes was fast and simple but underperformed in recall compared to other models. |
| Random Forest        | Delivered excellent performance with high accuracy (85.9%) and AUC (0.91), showing strong class separation. Precision (0.74) and recall (0.64) were well-balanced, leading to a solid F1 score (0.69). MCC (0.60) was also the highest among all tested models up to this point, indicating strong correlation between predictions and true labels. Overall, Random Forest proved to be a robust ensemble method, outperforming simpler models in both accuracy and balance. |
| XGBoost              | Achieved the best overall performance with the highest accuracy (87.6%) and AUC (0.93), demonstrating excellent class separation. Precision (0.78) and recall (0.68) were both strong, resulting in the highest F1 score (0.73) and MCC (0.65) among all models. XGBoost showed superior balance between precision and recall, making it the most reliable and effective model for this dataset. |

---

## 6. Streamlit App Deployment
- **App Features:**  
  - Dataset upload option (CSV)  
  - Model selection dropdown  
  - Display of evaluation metrics  
  - Confusion matrix visualization  
  - Classification report table  

- **Deployment:**  
  - Hosted on Streamlit Community Cloud  
  - Accessible via GitHub integration  

---

## 7. Submission Checklist
- [x] GitHub repository with source code, requirements.txt, README.md  
- [x] Streamlit app deployed and link shared  
- [x] Screenshot of execution on BITS Virtual Lab  
- [x] README.md included in submission PDF  

---

## 8. Conclusion
Among all six models, **XGBoost delivered the best performance**, achieving the highest accuracy, AUC, F1 score, and MCC. Random Forest also performed strongly, while Logistic Regression and KNN provided solid baselines. Naive Bayes was the weakest due to poor recall, highlighting its limitations on complex datasets. Overall, ensemble methods (Random Forest and XGBoost) proved most effective for the Adult Income classification problem.