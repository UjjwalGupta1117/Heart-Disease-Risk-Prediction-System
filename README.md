Heart Disease Risk Prediction System
An end-to-end Machine Learning application that predicts heart disease risk from clinical indicators using a K-Nearest Neighbors (KNN) model. The project combines a trained ML model with an interactive Streamlit dashboard and Plotly visualizations.
Disclaimer: This project is for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose heart disease or make healthcare decisions.

Project Overview
Training a machine learning model is only one part of building an ML application. The application also needs consistent preprocessing, reliable inference, and a clear user experience.

This project explores the complete workflow—from preparing data and training a model to integrating it into an interactive web application.
Features
- KNN prediction model: Predicts a model-estimated risk category from user-provided clinical indicators.
- Interactive Streamlit dashboard: Provides a web interface for entering patient data and viewing results.
- Plotly visualizations: Includes a risk-score gauge, vital-sign comparisons, and an interactive 3D patient-data visualization.
- Consistent preprocessing: Aligns feature scaling and one-hot encoding between training and inference.
- Model integration: Uses Joblib to load the trained model and supporting preprocessing objects, where saved.
- Input-to-feature mapping: Maps UI inputs to the feature schema expected by the model.
Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib
How It Works
1. Prepare the data — inspect and clean the dataset and select the model features.
2. Split the data — create training and test sets before fitting preprocessing steps.
3. Preprocess features — fit the scaler and any encoders on training data, then apply the fitted transformations consistently.
4. Train the model — train a K-Nearest Neighbors classifier on the processed training data.
5. Save model artifacts — save the trained model and any required preprocessing objects with Joblib.
6. Run the application — load the saved artifacts, collect user inputs in Streamlit, transform inputs using the same feature schema, and display the model output and visualizations.
Key Learning Challenges
1. Data leakage during scaling
The scaler must be fitted only on the training data. The fitted scaler is then used to transform both training and test data. Fitting preprocessing on the full dataset can expose information from the test set and lead to overly optimistic evaluation.
2. Feature mismatch
Using drop_first=True for one-hot encoding removes a reference category. If the encoded columns at inference time do not match the columns used during training, predictions may fail or become incorrect. The application must preserve the expected feature names, order, and encoding.
3. Unexpected predictions
The initial form defaults repeatedly produced high-risk predictions. Investigating this highlighted how default input values, feature distributions, scaling, and KNN's distance-based behavior can influence predictions. Defaults should be chosen carefully and should not imply a medically meaningful baseline.
Getting Started
Prerequisites
- Python installed
- Git installed
- The project source code and required saved model artifacts
1. Clone the repository
Replace <your-github-username> with your GitHub username:
git clone https://github.com/<your-github-username>/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction
2. Create and activate a virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
If the repository contains a requirements.txt file:
pip install -r requirements.txt
Otherwise, install the main libraries:
pip install pandas scikit-learn streamlit plotly joblib
4. Run the Streamlit app
Use the actual path to your Streamlit entry-point file. For example, if the file is named app.py:
streamlit run app.py
The terminal will display a local URL to open in your browser.
Before publishing: Update the entry-point filename and any folder or artifact paths to match your actual project structure. Include the required model and preprocessing artifacts, or document how to generate them.

Suggested Project Structure
Your actual filenames may differ. Adjust this example to match your repository.
heart-disease-risk-prediction/
├── app.py
├── train_model.py
├── data/
│   └── README.md
├── models/
│   ├── knn_model.joblib
│   └── scaler.joblib
├── requirements.txt
├── .gitignore
└── README.md
Only include files and artifacts that exist in your project. If the scaler or encoder is stored inside a single saved pipeline, document that instead of listing separate files.
Evaluation and Reliability
A responsible ML workflow should evaluate more than whether the app runs.
- Evaluate the model on a held-out test set that was not used to fit preprocessing or the model.
- Report appropriate classification metrics, such as precision, recall, F1-score, and a confusion matrix.
- If the positive class represents heart disease, explain which label is treated as positive.
- Check that training and inference use the same feature names, ordering, encoding, and scaling.
- Test missing, invalid, and out-of-range inputs.
- Avoid describing model output as a confirmed diagnosis or calibrated medical probability unless that interpretation has been properly validated.
Note: Add your actual evaluation results here once verified. Do not publish assumed or placeholder metrics as measured results.
Future Improvements
- Add automated tests for preprocessing and prediction.
- Validate form inputs and handle missing or invalid values clearly.
- Save preprocessing and the classifier together in a Scikit-learn Pipeline.
- Add model evaluation reports and explain the limitations of the dataset.
- Improve accessibility and dashboard usability.
- Document the dataset source, permitted use, and feature definitions.
What I Learned
This project reinforced that a working ML application needs more than a trained model. Reliable preprocessing, consistent inputs, careful debugging, meaningful evaluation, and thoughtful user experience are all important parts of the workflow.
Author
Created as part of my Day 49 of my AI/ML Pivot journey—from model training toward building practical, end-to-end machine learning applications.
If you find this project useful, feel free to explore the code and share constructive feedback.
