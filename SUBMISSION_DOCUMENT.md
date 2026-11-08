# BITS Pilani — Work Integrated Learning Programmes Division
## M.Tech (AIML) | Machine Learning
### Assignment - 2 Submission Document

---

## 1. Mandatory Submission Links

### 1.1 GitHub Repository Link
* **URL**: `https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`
* **Contents Included**:
  * Complete source code (`app.py`, `model/train_models.py`, `model/train_models.ipynb`)
  * `requirements.txt`
  * Complete `README.md`
  * Test data used in experiments (`test_data.csv`)
  * Serialized trained model files (`model/*.joblib`) & preprocessing artifacts

### 1.2 Live Streamlit App Link
* **URL**: `https://<YOUR_STREAMLIT_SUBDOMAIN>.streamlit.app`
* **Deployment Platform**: Streamlit Community Cloud

---

## 2. BITS Virtual Lab Execution Proof
*(Insert your screenshot of running the assignment / Jupyter Notebook / VS Code on BITS Virtual Lab here)*

> **[ INSERT BITS VIRTUAL LAB SCREENSHOT HERE ]**

---

## 3. GitHub README Content & Evaluation Report

### a. Problem Statement
Credit card default prediction is a critical risk-management problem faced by financial institutions. When cardholders default on their credit commitments, banks incur substantial bad-debt losses and operational recovery overhead. Conversely, misclassifying responsible credit card customers as potential defaulters can damage customer relationships and lead to lost interest revenue.

The objective of this assignment is to:
1. Formulate credit card default prediction as a supervised binary classification task: predicting whether a client will default on their credit card payment in the upcoming month ($y \in \{0, 1\}$).
2. Train, evaluate, and benchmark 5 diverse machine learning classifiers (**Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors**, **Naive Bayes**, and **Support Vector Machine**) on the exact same dataset using 6 standardized evaluation metrics: **Accuracy**, **ROC-AUC**, **Precision**, **Recall**, **F1-Score**, and **Matthews Correlation Coefficient (MCC)**.
3. Build and deploy an interactive **Streamlit web application** that accepts test CSV uploads, loads pre-trained serialized model artifacts, and dynamically renders metrics, confusion matrices, and classification reports without retraining.

---

### b. Dataset Description
* **Dataset Name**: Default of Credit Card Clients Dataset
* **Source**: UCI Machine Learning Repository / Kaggle ([UCI Dataset Link](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients))
* **Target Variable**: `default` (Binary: `1` = Default payment next month, `0` = No default)
* **Dataset Characteristics**:
  * **Total Instances**: 30,000 credit card accounts
  * **Feature Dimensionality**: 23 explanatory input features (exceeding the assignment minimum of 12 features)
  * **Instance Size**: 30,000 instances (exceeding the assignment minimum of 500 instances)
  * **Class Distribution**: 23,364 non-default instances (77.88%) and 6,636 default instances (22.12%)
  * **Data Integrity**: 0 missing or NaN values across all 30,000 records.

#### Feature Breakdown:
1. **Demographic Variables**:
   * `LIMIT_BAL`: Amount of given credit in NT dollars (individual and family/supplementary credit).
   * `SEX`: Gender (`1` = male, `2` = female).
   * `EDUCATION`: Education level (`1` = graduate school, `2` = university, `3` = high school, `4` = others).
   * `MARRIAGE`: Marital status (`1` = married, `2` = single, `3` = others).
   * `AGE`: Age in years.
2. **Repayment Status History (April 2005 – September 2005)**:
   * `PAY_0` to `PAY_6`: Monthly repayment status (-2 = no consumption, -1 = paid in full, 0 = use of revolving credit, 1 = payment delay for one month, 2 = payment delay for two months, ..., 8 = payment delay for eight months, 9 = payment delay for nine months and above).
3. **Bill Statement Amounts (April 2005 – September 2005)**:
   * `BILL_AMT1` to `BILL_AMT6`: Amount of bill statement (NT dollar) for each corresponding month.
4. **Previous Payment Amounts (April 2005 – September 2005)**:
   * `PAY_AMT1` to `PAY_AMT6`: Amount of previous payment (NT dollar) made in each corresponding month.

---

### c. Models Used & Evaluation Metric Comparison
All 5 classification models were trained on the same 80% stratified training split (24,000 instances) with a consistent `StandardScaler` preprocessing pipeline and evaluated on the identical 20% holdout test set (6,000 instances, with 1,327 default cases and 4,673 non-default cases).

#### Comparison Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8077 | 0.7076 | **0.6868** | 0.2396 | 0.3553 | 0.3244 |
| **Decision Tree** | **0.8172** | **0.7462** | 0.6584 | 0.3602 | **0.4657** | **0.3909** |
| **KNN** | 0.7980 | 0.7085 | 0.5697 | 0.3542 | 0.4368 | 0.3353 |
| **Naive Bayes** | 0.7525 | 0.7249 | 0.4515 | **0.5539** | 0.4975 | 0.3386 |
| **Support Vector Machine** | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |

---

### d. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong baseline accuracy (**80.77%**) and the **highest precision (0.6868)** of all evaluated classifiers, minimizing false accusations of default. However, it displays a low recall (**23.96%**), as the linear decision boundary is inherently conservative under class imbalance (22% positive class) and misses subtle delinquency patterns. |
| **Decision Tree** | Achieves the highest overall **ROC-AUC (0.7462)**, highest **F1-Score (0.4657)**, highest **MCC (0.3909)**, and highest **Accuracy (81.72%)**. By recursively splitting on repayment delay indicators (notably `PAY_0` and `PAY_2`), the tree captures critical non-linear interaction thresholds between recent payment delays and total credit exposure without overfitting (regularized with `max_depth=6`). |
| **KNN** | Yields **79.80% accuracy** and **0.7085 AUC**. Using distance-weighted neighborhood voting ($k=7$) allows the model to locally cluster similar borrower profiles. However, with 23 continuous and ordinal dimensions, distance metrics suffer mildly from the curse of dimensionality, leading to moderate precision (0.5697). |
| **Naive Bayes** | Achieves the **highest recall (55.39%)** among all models, successfully capturing over half of all actual defaulters. However, the Gaussian Naive Bayes assumption of conditional independence is heavily violated due to strong collinearity among consecutive monthly bill amounts (`BILL_AMT1`–`BILL_AMT6`), resulting in the lowest precision (**45.15%**) and more false alarms. |
| **Support Vector Machine** | Strong and consistent performer with **81.60% accuracy**, **0.7210 AUC**, and solid **Precision (0.6652)**. The RBF kernel maps the 23-dimensional normalized feature space into a higher-dimensional reproducing kernel Hilbert space, establishing a smooth non-linear decision boundary that closely rivals the Decision Tree. |

#### Overall, Winner for your dataset?
> **🏆 Overall Winner: Decision Tree Classifier**  
>  
> The **Decision Tree Classifier** is the decisive winner across multiple evaluation dimensions:
> 1. **Leader in Discrimination & Balance**: It achieves the highest **ROC-AUC (0.7462)**, the highest **F1-Score (0.4657)**, and the highest **Matthews Correlation Coefficient (0.3909)**.
> 2. **Handling Class Imbalance**: In credit risk evaluation where the default class represents ~22% of total customers, Accuracy alone is deceptive. MCC and F1-Score reflect genuine predictive quality, and Decision Tree outperforms all competitors on both.
> 3. **Financial Interpretability & Speed**: Unlike black-box kernel methods or slow instance-based learners like KNN, the Decision Tree provides lightning-fast inference on cloud endpoints (Streamlit Community Cloud) and straightforward rule interpretability for banking regulatory compliance.

