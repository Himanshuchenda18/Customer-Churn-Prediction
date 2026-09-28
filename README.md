# Telecom Customer Churn Prediction using machine learning models

An end-to-end machine learning application for predicting whether a telecom customer is likely to churn. The project covers data preprocessing, feature engineering, class-imbalance handling, ensemble machine learning, model evaluation, model serialization, and deployment using Streamlit.

## Project Overview

Customer churn is an important business problem for telecom companies because losing existing customers can directly affect revenue.

This project uses supervised machine learning to analyze customer information and predict whether a customer is likely to leave the telecom service.

The complete workflow is:

```text
Customer Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Class Imbalance Handling
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Hyperparameter Optimization
       ↓
Final AdaBoost Model
       ↓
Model Serialization
       ↓
Streamlit Deployment
       ↓
Real-Time Churn Prediction
```

---

## Key Features

* Exploratory Data Analysis
* Data cleaning and preprocessing
* Feature engineering
* Categorical feature encoding
* Feature scaling
* Class imbalance handling
* SMOTE and other imbalance techniques
* Multiple machine learning models
* Ensemble learning
* Model evaluation using classification metrics
* Cross-validation
* Hyperparameter optimization
* AdaBoost-based final prediction model
* Model serialization using Joblib/Pickle
* Interactive Streamlit application
* Real-time customer churn prediction

---

## Technologies Used

| Technology       | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| Python           | Main programming language                      |
| Pandas           | Data manipulation and preprocessing            |
| NumPy            | Numerical operations                           |
| Scikit-learn     | Machine learning, preprocessing and evaluation |
| Imbalanced-learn | Class imbalance techniques such as SMOTE       |
| AdaBoost         | Final ensemble classification model            |
| XGBoost          | Gradient boosting experimentation              |
| CatBoost         | Gradient boosting experimentation              |
| LightGBM         | Gradient boosting experimentation              |
| Joblib/Pickle    | Saving and loading the trained model           |
| Streamlit        | Interactive web application                    |
| Matplotlib       | Data visualization                             |
| Git/GitHub       | Version control and project hosting            |

---

# Machine Learning Problem

This is a:

**Supervised Binary Classification Problem**

The model predicts one of two possible outcomes:

```text
0 → Customer does not churn

1 → Customer churns
```

The target variable is:

```text
Churn
```

The model learns patterns from historical customer information and uses those patterns to predict churn for new customers.

---

# Dataset

The project uses telecom customer information containing demographic, service, contract and billing-related attributes.

Typical features include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

The target variable is:

```text
Churn
```

---

# Data Preprocessing

## 1. Data Cleaning

The dataset is first inspected for:

* Missing values
* Incorrect data types
* Duplicate records
* Invalid values
* Categorical and numerical columns

For example, `TotalCharges` may contain values that need to be converted from strings into numerical values.

---

## 2. Removing Identifier Columns

Customer identifiers such as:

```text
customerID
```

are not useful predictive features because they identify a customer rather than describe customer behavior.

Therefore, identifier columns can be removed before model training.

---

# Feature Engineering

Feature engineering is used to create more useful representations of the original data.

For example, customer tenure can be divided into groups:

```text
1–12 months
13–24 months
25–36 months
37–48 months
49–60 months
61–72 months
```

This can help the model capture different customer lifecycle patterns.

---

# Categorical Encoding

Machine learning algorithms generally require numerical input.

Categorical variables such as:

```text
Contract
InternetService
PaymentMethod
```

are converted into numerical representations using techniques such as one-hot encoding.

For example:

```text
Contract

Month-to-month
One year
Two year
```

can become:

```text
Contract_Month-to-month
Contract_One year
Contract_Two year
```

This avoids incorrectly treating categories as having a numerical order.

---

# Feature Scaling

Numerical features can have very different ranges.

For example:

```text
Tenure:          1–72
MonthlyCharges:  20–120
TotalCharges:    20–8000
```

Scaling can transform these features into comparable ranges.

Techniques experimented with include:

* StandardScaler
* MinMaxScaler

An important principle is that the scaler should be fitted using the training data rather than the complete dataset to avoid data leakage.

---

# Class Imbalance

One of the important challenges in churn prediction is class imbalance.

Typically, the number of customers who do not churn is significantly larger than the number who churn.

For example:

```text
No Churn
██████████████████████████████████████

Churn
██████████
```

If a model simply predicts "No Churn" for almost everyone, it can still achieve relatively high accuracy while failing to identify actual churners.

Therefore, accuracy alone is not sufficient.

---

# Imbalance Handling Techniques

Different techniques can be experimented with to improve minority-class learning.

## SMOTE

SMOTE stands for:

**Synthetic Minority Over-sampling Technique**

Instead of simply duplicating minority examples, SMOTE creates synthetic examples based on existing minority samples.

Conceptually:

```text
Existing Churn Customers
          ↓
       SMOTE
          ↓
Synthetic Churn Customers
          ↓
More Balanced Training Data
```

---

## ADASYN

ADASYN is another oversampling technique.

It focuses more on difficult minority examples that are harder for the model to classify.

---

## SMOTEENN

SMOTEENN combines:

```text
SMOTE
+
Edited Nearest Neighbours
```

SMOTE creates additional minority examples while ENN can remove noisy or ambiguous examples.

---

## Sample Weighting / Class Weighting

Another approach is to give more importance to minority-class observations during model training.

This allows the algorithm to penalize mistakes on churn customers more strongly.

---

# Machine Learning Models

Multiple classification algorithms can be compared during experimentation.

## Decision Tree

A Decision Tree creates a sequence of decision rules.

Example:

```text
Contract = Month-to-month?
       ↓
     Yes
       ↓
Tenure < 12 months?
       ↓
     Yes
       ↓
Higher Churn Risk
```

It is simple and interpretable but can overfit if not controlled.

---

## Random Forest

Random Forest combines multiple decision trees.

```text
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┤
Tree 4 ──┤──→ Final Prediction
Tree 5 ──┘
```

It generally provides better stability than a single decision tree.

---

## AdaBoost

AdaBoost stands for:

**Adaptive Boosting**

It builds weak learners sequentially.

Conceptually:

```text
Weak Model 1
      ↓
Identify difficult examples
      ↓
Weak Model 2
      ↓
Focus on difficult examples
      ↓
Weak Model 3
      ↓
Final Combined Model
```

The final prediction combines the outputs of multiple weak learners.

The trained AdaBoost model is saved as:

```text
ada_boost_churn_model.pkl
```

---

## XGBoost

XGBoost is a gradient boosting algorithm that builds trees sequentially and tries to improve the errors made by previous trees.

---

## CatBoost

CatBoost is a gradient boosting framework that provides strong support for categorical data and is often useful for datasets containing many categorical features.

---

## LightGBM

LightGBM is another gradient boosting framework designed for efficient and scalable training.

---

# Model Evaluation

Models are evaluated using multiple metrics rather than relying only on accuracy.

Important metrics include:

### Accuracy

Measures the overall percentage of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

### Precision

Answers:

> Of the customers predicted as churners, how many actually churned?

```text
Precision = TP / (TP + FP)
```

---

### Recall

Answers:

> Of all the customers who actually churned, how many did the model identify?

```text
Recall = TP / (TP + FN)
```

Recall is particularly important when missing a potential churn customer is costly.

---

### F1 Score

F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

It is useful when the dataset is imbalanced and both false positives and false negatives matter.

---

# Confusion Matrix

The confusion matrix contains four outcomes:

```text
                    Predicted
                  No       Yes

Actual No         TN       FP

Actual Yes        FN       TP
```

Where:

* **TP** = correctly predicted churn
* **TN** = correctly predicted non-churn
* **FP** = predicted churn but customer did not churn
* **FN** = predicted non-churn but customer actually churned

---

# Cross-Validation

Cross-validation is used to obtain a more reliable estimate of model performance.

For example, with 5-fold cross-validation:

```text
Dataset
   ↓
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
```

The model is trained and evaluated multiple times using different training and validation portions.

This helps determine whether the model performs consistently rather than only performing well on one particular split.

---

# Hyperparameter Optimization

Machine learning algorithms contain parameters that control their behavior.

The project can use approaches such as:

* RandomizedSearchCV
* Optuna

These techniques search for parameter combinations that improve model performance.

---

# Model Serialization

After training the final model, it is saved to disk.

Example:

```python
joblib.dump(model, "ada_boost_churn_model.pkl")
```

This creates:

```text
ada_boost_churn_model.pkl
```

The `.pkl` file contains the serialized trained Python model object.

The application can then load it without retraining:

```python
model = joblib.load("ada_boost_churn_model.pkl")
```

This is important for deployment because the deployed application should normally perform predictions using the already-trained model.

---

# Streamlit Application

The trained model is integrated into a Streamlit interface.

The application flow is:

```text
User
 ↓
Enter Customer Information
 ↓
Streamlit
 ↓
Create Input DataFrame
 ↓
Apply Required Preprocessing
 ↓
Load Trained AdaBoost Model
 ↓
Prediction
 ↓
Churn / No Churn
 ↓
Prediction Probability
```

The application can provide both:

```text
Prediction
+
Probability
```

For example:

```text
Prediction: Customer is likely to churn

Churn Probability: 82%
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/pik1989/Cancer-Risk-Prediction.git
```

Move into the project directory:

```bash
cd Cancer-Risk-Prediction
```

---

## 2. Create a Virtual Environment

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install python3-venv python3-full -y
```

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, the terminal should show something similar to:

```text
(.venv) user@server:~/Cancer-Risk-Prediction$
```

---

## 3. Install Dependencies

Install the Python packages:

```bash
pip3 install -r requirements.txt
```

Or, after activating the virtual environment:

```bash
pip install -r requirements.txt
```

---

# Run the Application

For local development:

```bash
python3 -m streamlit run app.py
```

Streamlit normally starts the application on port:

```text
8501
```

---

# Deployment on Ubuntu/Linux Server

Update the system:

```bash
sudo apt update
sudo apt-get update
sudo apt upgrade -y
```

Install required system utilities:

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```

Install Python tools:

```bash
sudo apt install python3-pip python3-venv python3-full -y
```

Clone the repository:

```bash
git clone https://github.com/pik1989/Cancer-Risk-Prediction.git
```

Enter the project:

```bash
cd ~/Cancer-Risk-Prediction
```

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Temporary Run

To run Streamlit interactively:

```bash
python3 -m streamlit run app.py
```

This is useful during development and testing.

---

# Background / Persistent Run

To run Streamlit in the background:

```bash
nohup python3 -m streamlit run app.py > streamlit.log 2>&1 &
```

Explanation:

```text
nohup
  ↓
Prevents the process from stopping when the terminal session closes

python3 -m streamlit run app.py
  ↓
Starts the Streamlit application

> streamlit.log
  ↓
Stores application output in a log file

2>&1
  ↓
Redirects errors into the same log file

&
  ↓
Runs the application in the background
```

---

# Server Deployment

For a remote Ubuntu server where the Streamlit application needs to be accessed externally, Streamlit can be configured to listen on the server interface.

Example:

```bash
nohup python3 -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  > streamlit.log 2>&1 &
```

The server/firewall must also allow the selected port.

---

# Check Whether Streamlit Is Running

Use:

```bash
ps aux | grep streamlit
```

You can also inspect the log:

```bash
cat streamlit.log
```

Or:

```bash
tail -f streamlit.log
```

---

# Stop the Application

Find the process:

```bash
ps aux | grep streamlit
```

Then terminate the corresponding process:

```bash
kill <PROCESS_ID>
```

---

# Project Structure

A typical project structure is:

```text
Cancer-Risk-Prediction/
│
├── app.py
├── requirements.txt
├── ada_boost_churn_model.pkl
├── README.md
├── DEPLOYMENT.md
├── .gitignore
│
├── data/
│
├── notebooks/
│
├── src/
│
└── ...
```

The exact folders may vary depending on the final project organization.

---

# Important Files

### `app.py`

Contains the Streamlit application and prediction interface.

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `ada_boost_churn_model.pkl`

Contains the serialized trained AdaBoost churn prediction model.

### `README.md`

Contains the main project documentation.

### `DEPLOYMENT.md`

Contains deployment and server setup instructions.

### `.gitignore`

Specifies files and folders that should not be committed to GitHub.

---

# Recommended `.gitignore`

The virtual environment should **not** be pushed to GitHub.

Create a `.gitignore` file containing:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
*.log
.DS_Store
```

Do not commit API keys, passwords, tokens or other secrets.

---

# Model File and GitHub

The trained model:

```text
ada_boost_churn_model.pkl
```

can be included in the repository if the Streamlit application requires it and the file is reasonably sized.

If the model is too large for normal GitHub storage, use Git LFS or another model/artifact storage solution.

Never commit:

```text
API keys
Passwords
.env files
Private credentials
Database credentials
```

---

# Future Improvements

Possible improvements include:

* Build a complete Scikit-learn Pipeline
* Add automated model retraining
* Add SHAP-based model explainability
* Add model monitoring
* Add data-drift detection
* Add authentication to the application
* Expose predictions through a FastAPI REST API
* Containerize the application using Docker
* Deploy using a cloud platform
* Add CI/CD using GitHub Actions

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Imbalance Handling
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Hyperparameter Optimization
 ↓
Model Serialization
 ↓
Streamlit Deployment
 ↓
Customer Churn Prediction
```

The project combines machine learning experimentation with practical deployment, allowing a trained AdaBoost model to be used through an interactive Streamlit application.
