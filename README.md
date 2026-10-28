# Heart Disease Risk Prediction System

An end-to-end Machine Learning application that predicts a model-estimated heart disease classification from clinical indicators using a **K-Nearest Neighbors (KNN)** model. The project combines a trained machine learning model with an interactive Streamlit dashboard and Plotly visualizations.

> **Medical Disclaimer:** This is an educational project, not a medical device. Its predictions must not be used to diagnose heart disease or make healthcare decisions. Always consult a qualified healthcare professional for medical advice.

---

## Project Overview

Building a machine learning application involves more than training a model. It also requires consistent data preprocessing, reliable model inference, meaningful evaluation, and a user-friendly interface.

This project demonstrates an end-to-end workflow for integrating a trained KNN classifier into a Streamlit application with interactive visualizations.

The application allows users to enter clinical indicators and view the model's output alongside comparisons with the dataset.

---

## Features

- **KNN Classification:** Predicts a model-estimated heart disease classification.
- **Interactive Streamlit Dashboard:** Collects user inputs through an interactive web interface.
- **Plotly Visualizations:** Displays a prediction gauge, patient-vital comparisons, and a 3D patient-data visualization.
- **Feature Alignment:** Aligns input columns with the feature columns expected by the saved model.
- **Model Artifact Loading:** Uses Joblib to load the trained model, scaler, and expected feature columns.
- **Dataset Comparison:** Compares selected input values against dataset averages.
- **Educational ML Workflow:** Demonstrates model integration, preprocessing, and interactive visualization.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preparation |
| NumPy | Numerical computing and dependency support |
| Scikit-learn | Machine learning and preprocessing |
| Streamlit | Interactive web application |
| Plotly | Interactive charts and visualizations |
| Joblib | Loading saved model artifacts |

---

## Project Structure

```text
Heart-Disease-Risk-Prediction-System/
├── app.py
├── heart.csv
├── requirements.txt
├── KNN_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
└── README.md
```

### File Descriptions

- `app.py` — Main Streamlit application containing the input form, prediction logic, and visualizations.
- `heart.csv` — Dataset used for patient-data comparisons and model development.
- `requirements.txt` — Python dependencies required to run the application.
- `KNN_heart_model.pkl` — Saved KNN model.
- `heart_scaler.pkl` — Saved feature scaler used during inference.
- `heart_columns.pkl` — Saved feature-column schema expected by the model.
- `README.md` — Project documentation and setup instructions.

**Note:** The three `.pkl` filenames above are the artifact paths referenced in `app.py`. Ensure that these files actually exist in the repository before running the application. If they are not committed, document how to generate them.

---

## Dataset

- **Dataset:** Heart Failure Prediction Dataset
- **Source:** [Kaggle — Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Local file:** `heart.csv`
- **Task:** Binary classification using clinical indicators.

### Features

| Feature | Description |
|---|---|
| `Age` | Age of the individual |
| `Sex` | Recorded sex category |
| `ChestPainType` | Type of chest pain |
| `RestingBP` | Resting blood pressure |
| `Cholesterol` | Recorded cholesterol level |
| `FastingBS` | Fasting blood sugar indicator |
| `RestingECG` | Resting electrocardiogram result |
| `MaxHR` | Maximum recorded heart rate |
| `ExerciseAngina` | Exercise-induced angina indicator |
| `Oldpeak` | ST depression measure |
| `ST_Slope` | Slope of the peak exercise ST segment |

The dataset's target column is `HeartDisease`.

Check the original Kaggle page for licensing, attribution, and citation requirements before redistributing the dataset.

---

## How It Works

1. **Collect inputs:** Users enter clinical indicators through the Streamlit form.
2. **Prepare features:** The application maps numerical inputs and categorical selections into a feature DataFrame.
3. **Align columns:** Missing expected columns are added, and the input columns are reordered according to the saved feature schema.
4. **Scale features:** The saved scaler transforms the input using the previously fitted scaling parameters.
5. **Generate prediction:** The trained KNN model predicts the classification.
6. **Calculate model output:** If supported, the application obtains a class probability using `predict_proba()`.
7. **Visualize results:** Streamlit and Plotly display the model output and available patient-data comparisons.

The application loads the model and preprocessing artifacts using Joblib, so the artifacts must be compatible with the training workflow.

---

## Getting Started

### Prerequisites

Install the following:

- Python compatible with the pinned package versions.
- Git.
- The project source code.
- The dataset and required saved model artifacts.

### 1. Clone the Repository

```bash
git clone https://github.com/UjjwalGupta1117/Heart-Disease-Risk-Prediction-System.git
cd Heart-Disease-Risk-Prediction-System
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

Upgrade pip and install the project's requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

### Troubleshooting

**Missing model file**

If you see a `FileNotFoundError` mentioning `KNN_heart_model.pkl`, `heart_scaler.pkl`, or `heart_columns.pkl`, check that the corresponding file exists in the project directory.

**Feature mismatch**

If the model raises an input-shape or feature mismatch error, verify that the training and inference workflows use the same encoding, feature names, and column order.

**Dependency compatibility**

If a saved model fails to load, check the Scikit-learn version used during training against the installed version. Reproducing the training environment may be necessary.

**Missing dataset**

If `heart.csv` is unavailable, the application cannot display its dataset-comparison and 3D visualization charts.

---

## Model Evaluation

Model performance should be evaluated using a held-out test set.

Useful classification metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Breakdown of correct and incorrect classifications.

For a health-related classification project, accuracy alone is not enough to establish model quality.

Evaluation results should be calculated and verified before being published. No independently verified performance results are reported in this README.

---

## Known Limitations

- The model's predictions depend on the dataset and training process.
- Manually constructed one-hot encoded inputs must match the model's training feature schema.
- A model probability is not automatically a medically calibrated risk estimate.
- Dataset averages are descriptive comparisons, not clinical reference ranges.
- The model has not been established as a clinically validated diagnostic tool.
- The application's actual performance depends on the saved model and preprocessing artifacts.

---

## Future Improvements

- Integrate preprocessing and classification into a single scikit-learn `Pipeline`.
- Add input validation and clearer error messages.
- Compare KNN against Logistic Regression and tree-based classifiers.
- Add cross-validation and hyperparameter tuning.
- Display a verified classification report and confusion matrix.
- Add automated tests for preprocessing and prediction.
- Improve deployment documentation and artifact management.
- Add more prominent medical disclaimers inside the dashboard.

---

## Responsible Use

This application is intended for educational and demonstration purposes only.

- A model output is not a clinical diagnosis.
- A predicted class or probability is not a validated individual heart disease risk assessment.
- Do not use this application to make medical or treatment decisions.
- Predictions may be inaccurate or biased and may not generalize to other populations.
- Consult a qualified healthcare professional for medical concerns.

---

## Author

**Ujjwal Gupta**

GitHub: [@UjjwalGupta1117](https://github.com/UjjwalGupta1117)

Project repository: [Heart Disease Risk Prediction System](https://github.com/UjjwalGupta1117/Heart-Disease-Risk-Prediction-System)

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting the dataset.
- [Scikit-learn](https://scikit-learn.org/) for machine learning and preprocessing tools.
- [Streamlit](https://streamlit.io/) for the application framework.
- [Plotly](https://plotly.com/) for interactive visualizations.
- [Pandas](https://pandas.pydata.org/) for data manipulation.
