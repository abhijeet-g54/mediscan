MediScan — AI Symptom Analyzer

MediScan is a demonstration AI assistant for symptom analysis and disease prediction. It accepts user-entered symptoms (with duration and intensity), asks clarifying questions when needed, and returns the top probable conditions with confidence scores and triage guidance.

Important: MediScan is an educational prototype and not a replacement for professional medical advice.

Features

Interactive Streamlit frontend (app/main_app.py)

Symptom normalization and fuzzy matching (utils/feature_engineering.py)

Preprocessing helper to convert row-wise symptom data to wide-format (utils/data_preprocessing.py)

Model training script that produces models/symptom_disease_model.pkl (train_model.py)

Clarifying question rules (models/clarifying_questions.json)

Optional SHAP-based explainability (if shap is installed)

Local session feedback and downloadable text summary for each analysis

Tech stack

Python

Streamlit (frontend)

scikit-learn, XGBoost (modeling)

pandas, numpy (data handling)

fuzzywuzzy (fuzzy matching)

SHAP (optional explainability)

Repository structure
MediScan/
├── app/                     # Streamlit UI (main_app.py)
├── models/                  # Saved model + clarifiers (not committed)
├── utils/                   # preprocessing & feature utilities
├── data/                    # datasets (raw & processed) — keep out of repo
├── train_model.py           # Model training script
├── requirements.txt         # Dependencies
├── README.md                # This file
├── .gitignore               # Recommended ignore rules
└── LICENSE                  # Optional (e.g., MIT)

Installation and setup

The instructions below use Windows PowerShell. Commands are similar on macOS/Linux with appropriate shell changes.

Clone the repository and change directory:

git clone https://github.com/abhijeet-g54/MediScan.git
cd MediScan


Create and activate a virtual environment:

python -m venv venv
.\venv\Scripts\Activate.ps1


If PowerShell prevents execution of the activation script, run once (as Administrator):

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt


Notes:

If xgboost or shap are problematic to install on your system, you may remove them from requirements.txt temporarily and install later. The app will still run without SHAP (explanation parts will be skipped).

Ensure you install packages inside the activated virtual environment.

Dataset format

When you are ready to train, provide a dataset in wide-format.

Place your dataset at data/processed/disease_symptom_dataset.csv (preferred) or data/raw/disease_symptom_dataset.csv.

The CSV must include a Disease column (string) and one binary column per canonical symptom. Example header:

fever,cough,headache,abdominal_pain,sore_throat,...,Disease


Each row corresponds to one training example. Symptom columns contain 0 or 1 indicating absence/presence.

If your data is row-wise (a list of symptoms per row), use the helper in utils/data_preprocessing.py to convert to wide format.

Training the model

After placing your processed dataset, run:

python train_model.py


What the script does:

Loads the processed CSV (data/processed/disease_symptom_dataset.csv) or fallback data/raw/disease_symptom_dataset.csv

Encodes the target disease labels

Trains an XGBoost model and a RandomForest model

Builds a soft-voting ensemble of the two models

Evaluates on a held-out test set and prints metrics

Saves a package to models/symptom_disease_model.pkl containing:

"model": the trained ensemble

"label_encoder": the LabelEncoder instance

"features": ordered list of symptom features

Writes data/processed/feature_list.txt listing feature names

If xgboost is not available, edit train_model.py to use only RandomForest or install a compatible xgboost wheel for your Python version.

Running the app

After training (or if you already have models/symptom_disease_model.pkl), run the Streamlit app:

python -m streamlit run app\main_app.py


Open the URL printed by Streamlit (typically http://localhost:8501
).

Enter symptoms using the search box or suggestion buttons.

Add duration and intensity for each symptom, then click Analyze.

If the model is uncertain or top predictions are close, a single clarifying question may be asked.

The app shows top-3 predicted conditions with confidence, a triage warning for red-flag symptoms or low confidence, and an option to download a text summary.

Development notes

utils/feature_engineering.py contains normalization logic (mapping user text to canonical tokens) and the function build_feature_vector which converts symptoms + duration + intensity into the numeric feature vector used at inference time.

models/clarifying_questions.json (and models/clarifying_questions.json) contain default clarifying questions and targeted pairwise disambiguation questions.

app/main_app.py expects the model package models/symptom_disease_model.pkl to include the model, label encoder, and features; ensure these are present after training.

The app uses the canonical feature list (features) to build vectors. If your symptom dictionary or dataset uses different canonical names, align them before training.

Improving accuracy (suggested next steps)

Expand and clean training data: combine multiple reliable datasets and curate samples.

Add patient meta-features: age, sex, comorbidities (columns added to dataset and model).

Class balancing: use class weights or oversampling (SMOTE) for rare disease classes.

Feature engineering: add interaction features for clinically significant symptom pairs.

Hyperparameter tuning: use RandomizedSearchCV for XGBoost and RandomForest.

Active learning: collect (consented) user feedback and confirmed labels to retrain periodically.

Expert rules: add deterministic red-flag rules and verified symptom-disease mappings.

Security, privacy, and ethics

Do not store personally identifiable information (PII) unless you implement secure storage and obtain explicit consent.

Display a clear disclaimer in the UI: MediScan is not a medical diagnosis tool.

If you plan to collect feedback or user-confirmed diagnoses, implement a consent workflow and secure storage (encryption at rest).

Validate and sanitize any external data sources and respect dataset licenses.

Contributing

Contributions are welcome. Suggested contribution process:

Fork the repository.

Create a feature branch: git checkout -b feature/your-feature

Commit changes with clear messages and push to your fork.

Open a pull request describing the change and rationale.

Include unit tests where applicable and avoid committing datasets or model binaries.

.gitignore recommendations

Add a .gitignore file to exclude virtual environments, data, and model binaries:

# Python
__pycache__/
*.pyc
venv/
.env
.ipynb_checkpoints/

# Data & Models
data/raw/
data/processed/
models/*.pkl

# IDEs
.vscode/
.idea/

# System
.DS_Store
Thumbs.db