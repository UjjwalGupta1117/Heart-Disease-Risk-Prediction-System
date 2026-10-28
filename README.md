# Heart Disease Risk Prediction System

An end-to-end Machine Learning application that predicts a model-estimated heart disease risk category from clinical indicators using a K-Nearest Neighbors (KNN) model. The project combines a trained ML model with an interactive Streamlit dashboard and Plotly visualizations.

> **Medical Disclaimer:** This is an educational project, not a medical device. It must not be used to diagnose heart disease or make healthcare decisions.

---

## Project Overview

Training a machine learning model is only one part of building an ML application. The application also needs consistent preprocessing, reliable inference, meaningful evaluation, and a clear user experience.

This project explores the complete workflow—from preparing data and training a model to integrating it into an interactive web application.

---

## Features

- **KNN prediction model:** Produces a model-estimated risk category from user-provided clinical indicators.
- **Interactive Streamlit dashboard:** Provides a web interface for entering inputs and viewing results.
- **Plotly visualizations:** Includes a risk-score gauge, vital-sign comparisons, and an interactive 3D patient-data visualization.
- **Consistent preprocessing:** Aligns feature scaling and one-hot encoding between training and inference.
- **Model integration:** Uses Joblib to load the trained model and any saved preprocessing objects.
- **Input-to-feature mapping:** Maps UI inputs to the feature schema expected by the model.

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib

---

## How It Works

1. **Prepare the data:** Inspect and clean the dataset and select the model features.
2. **Split the data:** Create training and test sets before fitting preprocessing steps.
3. **Preprocess features:** Fit the scaler and encoders on training data, then apply the fitted transformations consistently.
4. **Train the model:** Train a K-Nearest Neighbors classifier on the processed training data.
5. **Save model artifacts:** Save the trained model and required preprocessing objects with Joblib.
6. **Run the application:** Load the saved artifacts, collect user inputs in Streamlit, apply the same transformations, and display the model output and visualizations.

---

## Key Learning Challenges

1. **Data leakage during scaling:**
   The scaler must be fitted only on the training data. The fitted scaler is then used to transform training and test data. Fitting preprocessing on the full dataset can expose information from the test set and lead to overly optimistic evaluation.

2. **Feature mismatch:**
   Using `drop_first=True` for one-hot encoding removes a reference category. If encoded columns at inference time do not match the columns used during training, predictions may fail or become unreliable. The application must preserve the expected feature names, order, and encoding.

3. **Unexpected predictions:**
   The initial form defaults repeatedly produced high-risk predictions. Investigating this highlighted how default input values, feature distributions, scaling, and KNN's distance-based behavior can influence predictions. Form defaults should be chosen carefully and should not imply a medically meaningful baseline.

---

## Dataset

- **Source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features Included:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.
- **Note:** Please check the dataset page on Kaggle for its license and citation requirements.

---

## Getting Started

### Prerequisites

- Python installed
- Git installed
- The project source code and required saved model artifacts

### 1. Clone the repository

```bash
git clone [https://github.com/UjjwalGupta1117/Heart-Disease-Risk-Prediction-System.git](https://github.com/UjjwalGupta1117/Heart-Disease-Risk-Prediction-System.git)
cd Heart-Disease-Risk-Prediction-System
