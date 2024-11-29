# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st

# Load the Dataset
file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv(file_path, header=None, names=column_names)

# Preprocessing and Data Cleaning
data.drop('ID', axis=1, inplace=True)  # Drop ID column
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})  # Encode Diagnosis (M=1, B=0)
X = data.drop('Diagnosis', axis=1)  # Features
y = data['Diagnosis']  # Target variable

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Balance data with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train AdaBoost Model
ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), 
    n_estimators=50,
    learning_rate=1.0,
    algorithm="SAMME",
    random_state=42
)
ada_classifier.fit(X_train_smote, y_train_smote)

# Streamlit Layout
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🔬", layout="centered")
st.title("🔬 Breast Cancer Prediction App")

# Add a header section with a subtitle
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #FF4B4B;">Predict Whether a Tumor is Malignant or Benign</h2>
        <p style="color: #555;">Enter the values for key features to predict the diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

# Input form for features
st.markdown("<h3 style='color: #4B88FF;'>Enter Tumor Characteristics</h3>", unsafe_allow_html=True)

user_inputs = {}
cols = st.columns(2)  # Split input fields into two columns for better design

for i, feature in enumerate(X.columns):
    with cols[i % 2]:  # Alternate columns
        user_inputs[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, value=1.0, format="%.2f")

# Prediction button
if st.button("Make Prediction", help="Click to predict whether the tumor is malignant or benign"):
    # Scale user inputs
    user_input_array = np.array(list(user_inputs.values())).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input_array)
    
    # Predict using AdaBoost model
    prediction = ada_classifier.predict(user_input_scaled)
    prediction_text = "Malignant" if prediction == 1 else "Benign"
    prediction_color = "#FF4B4B" if prediction == 1 else "#4CAF50"

    # Display result with styled text
    st.markdown(f"""
        <div style="text-align: center; margin-top: 30px;">
            <h2 style="color: {prediction_color};">The tumor is {prediction_text}.</h2>
        </div>
        """, unsafe_allow_html=True)

# Add footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #999;">Powered by Streamlit & AdaBoost Model | © 2024</p>
    """, unsafe_allow_html=True)
