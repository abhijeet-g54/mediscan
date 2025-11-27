#  MediScan — AI-Powered Symptom Analyzer
**A demonstration project for symptom-based disease prediction and triage assistance.**  
*MediScan is an educational prototype and **not** a substitute for professional medical diagnosis.*

---

##  Overview
MediScan analyzes user-entered symptoms (with optional **duration** and **intensity**) and predicts the most likely conditions using a trained machine-learning model.

The system can:
- Ask clarifying questions  
- Display confidence scores  
- Show safety/triage warnings  

---

##  Features
- **Streamlit UI** (`app/main_app.py`)
- **Symptom normalization + fuzzy matching** (`utils/feature_engineering.py`)
- **Row → wide-format data converter** (`utils/data_preprocessing.py`)
- **Model training pipeline** → outputs `models/symptom_disease_model.pkl`
- **Clarifying-question rules** (`models/clarifying_questions.json`)
- **Optional SHAP-based explanations**
- **Local session feedback + downloadable reports**

---

##  Tech Stack
- **Python**
- **Streamlit** (frontend)
- **scikit-learn**, **XGBoost** (modeling)
- **pandas**, **numpy** (data handling)
- **fuzzywuzzy** (text normalization)
- **SHAP** (optional)

---

## 📁 Repository Structure

MediScan/

├── app/ # Streamlit frontend

│ └── main_app.py

├── models/ # Saved model & question rules (ignored in repo)

├── utils/ # Preprocessing and feature engineering

├── data/ # Raw/processed datasets (keep out of repo)

├── train_model.py # ML training pipeline

├── requirements.txt # Dependencies

├── README.md # Documentation

├── .gitignore # Ignore rules

└── LICENSE # Optional license


---

##  Installation & Setup

> **These commands use Windows PowerShell.**  
> macOS/Linux users can use equivalent shell commands.


### Clone the repository

git clone https://github.com/abhijeet-g54/mediscan
cd mediscan


### Setup & Environment

Create a Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1
If PowerShell blocks execution
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


### Install Dependencies
3️⃣ Install Required Packages
pip install --upgrade pip
pip install -r requirements.txt



Notes

If xgboost or shap fail to install → remove them temporarily.
The app works without SHAP (you only lose explanations).



### Dataset Format
Required Structure (Wide Format)
fever,cough,headache,abdominal_pain,sore_throat,...,Disease
1,0,1,0,0,...,Flu

Requirements

Each symptom column = 0 or 1
Disease = target label
Each row = one training sample
Place your dataset here
data/processed/disease_symptom_dataset.csv
or:
data/raw/disease_symptom_dataset.csv
Row-Wise Symptom Format (List Input)
If your data looks like:
["fever", "cough", "fatigue"]


Convert it using:

utils/data_preprocessing.py

---

## Train the Model


Run Training
python train_model.py
What Happens During Training
Loads processed dataset
Encodes disease labels
Trains:
XGBoost
RandomForest
Builds a soft-voting ensemble
Evaluates model
Saves trained model to:
models/symptom_disease_model.pkl
Model File Contains
model — trained ensemble
label_encoder
features — canonical symptom list
Additional Output
data/processed/feature_list.txt

---

## Run the Application

Launch Streamlit App
python -m streamlit run app/main_app.py

Default URL
http://localhost:8501

App Workflow

Enter symptoms
Add duration & intensity
Click Analyze
(If needed) answer clarifying question

View:
Top-3 probable conditions
Confidence scores
Triage warnings
Downloadable summary

---

## Development Notes

utils/feature_engineering.py
Contains:
Symptom normalization
Fuzzy matching
Feature-vector builder
models/clarifying_questions.json
Defines:
Standard clarifiers
Targeted clarifiers
app/main_app.py
Expects:
Trained model
Label encoder
Feature list

⚠️ Ensure consistent symptom names across dataset, preprocessing, and UI.

---

## Improving Model Accuracy

Recommended Enhancements
Larger, cleaner dataset
Add meta-features (age, sex, comorbidities)
Handle class imbalance (weights / SMOTE)
Interaction features
Hyperparameter tuning (RandomizedSearchCV)
Active learning from user feedback
Rule-based expert red-flag checks

---

## Security, Privacy & Ethics

Guidelines
Do not store PII
Display a clear medical disclaimer
Encrypt stored feedback
Respect dataset licenses
Sanitize external data inputs

---

## Contributing

Contribution Workflow
Fork the repository
Create a feature branch:
git checkout -b feature/your-feature
Commit & push
Open a Pull Request
Avoid Committing-
Datasets
Model binaries
Virtual environments



