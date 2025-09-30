# 💳 FraudSense - Fraud Detection in Financial Transactions

A **machine learning pipeline** for detecting fraudulent transactions in financial datasets, deployed via a **Streamlit web app** for real-time prediction. Demonstrates **EDA, feature engineering, rare-event modeling, and deployment**.

---

## 🚀 Features

* Trained **logistic regression model** on **6M+ transactions**, handling extreme class imbalance.
* Achieved **94% recall** on rare fraud cases (~0.13% of data).
* Built a **Streamlit web app** for real-time fraud prediction.
* Conducted **EDA and anomaly detection**, identifying **cash-out** and **transfer** as high-risk transaction types.
* Implemented **feature engineering and preprocessing pipelines** using scikit-learn.

---

## 📊 Dataset

* Dataset: [Kaggle Fraud Detection Dataset](#)
* **6M+ rows**, 11 columns:

  * `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
* Fraud cases are extremely rare: **0.13%**, highlighting **class imbalance** challenges.

---

## 🔧 Technologies Used

* **Python**
* **Pandas, NumPy** – Data manipulation
* **Seaborn, Matplotlib** – Data visualization
* **Scikit-learn** – Modeling and preprocessing
* **Streamlit** – Web app deployment
* **Joblib** – Model serialization

---

## 📝 Project Workflow

1. **Data Analysis & Visualization**

   * Explored transaction types, fraud rates, and balance anomalies.
   * Identified top senders/receivers and high-risk transactions.

2. **Feature Engineering & Preprocessing**

   * Created pipelines for **numerical scaling** and **categorical encoding**.
   * Handled **class imbalance** with `class_weight='balanced'`.

3. **Model Training & Evaluation**

   * Split data: 70% train / 30% test
   * Logistic regression trained and evaluated using **classification report** and **confusion matrix**.

4. **Deployment with Streamlit**

   * Built interactive app for **real-time fraud predictions**.
   * Users input transaction details to instantly check if a transaction is fraudulent.

---

## 🎯 Usage

1. Clone the repo:

```bash
git clone <repository-url>
cd fraud-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run fraud_detection.py
```

4. Open the displayed URL in a browser to test transactions.

---

## 📂 Project Structure

```
fraud-detection/
│
├── fraud_detection.py               # Streamlit app
├── fraud_detection_pipeline.pickle  # Trained ML pipeline
├── analysis.ipynb                   # Jupyter notebook for EDA and modeling
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## ⚡ Author

**Nishant Kaushik** – https://github.com/nishantkaushik20
