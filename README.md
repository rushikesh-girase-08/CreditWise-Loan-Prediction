
# CreditWise Loan Prediction 💳

## 📌 Project Overview

CreditWise Loan Prediction is a Machine Learning project that predicts whether a loan application is likely to be **Approved or Rejected** based on an applicant's financial information, CIBIL score, income, loan amount, and asset values.

The project uses the **XGBoost Classifier** and a **Streamlit web application** to provide an interactive interface for loan prediction.

## 🎯 Objectives

- Predict loan approval status using Machine Learning.
- Analyze applicant financial information and credit scores.
- Build a user-friendly web application using Streamlit.
- Apply data preprocessing and categorical feature encoding using Scikit-learn.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- NumPy
- Joblib
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
CreditWise-Loan-Prediction/
│
├── app.py
├── project.ipynb
├── creditwise_model.pkl
├── loan_approval_dataset.csv
├── requirements.txt
└── README.md
```

## 📊 Dataset Information

The dataset contains information about loan applicants, including:

- Number of dependents
- Education
- Self-employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential assets value
- Commercial assets value
- Luxury assets value
- Bank asset value
- Loan status

**Target Variable:** `loan_status`

## 🤖 Machine Learning Algorithm

This project uses **XGBoost (Extreme Gradient Boosting)**, a powerful supervised Machine Learning algorithm used for classification tasks.

The machine learning workflow includes:

1. Data loading and exploration
2. Data preprocessing
3. Encoding categorical features
4. Splitting data into training and testing sets
5. Training the XGBoost Classifier
6. Evaluating model performance
7. Saving the trained model using Joblib
8. Integrating the model with Streamlit

A Scikit-learn Pipeline is used to combine preprocessing and model prediction.

## 📈 Model Performance

The model achieved the following results during testing:

| Metric | Score |
|---|---:|
| Training Accuracy | 99.91% |
| Testing Accuracy | 98.24% |

*Note: These are the results obtained during the project's reported evaluation. Performance may vary with different datasets or train-test splits.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rushikesh-girase-08/CreditWise-Loan-Prediction.git
```

### 2. Open the project folder

```bash
cd CreditWise-Loan-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🖥️ Application Features

- Interactive loan prediction form
- Applicant and financial information inputs
- CIBIL score and asset details
- XGBoost-based loan status prediction
- User-friendly Streamlit interface

## 🔮 Future Improvements

- Deploy the application online
- Add more model evaluation metrics
- Improve model explainability
- Test performance on additional datasets
- Add more applicant risk analysis features

## 👨‍💻 Author

**Rushikesh Girase**

GitHub: [rushikesh-girase-08](https://github.com/rushikesh-girase-08)

## 📜 Disclaimer

This project is developed for educational and demonstration purposes only. Predictions should not be used as the sole basis for real-world lending decisions.
