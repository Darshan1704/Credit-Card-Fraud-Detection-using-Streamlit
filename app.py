import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Streamlit title and description
st.title("Credit Card Fraud Detection Model")
# Display an image at the top
st.image("credit_card_image.jpg", use_column_width=True)
st.sidebar.title("App Description")
st.sidebar.write("Welcome to our Credit Card Fraud Detection application. "
                 "This tool helps identify fraudulent credit card transactions.")

# Load your dataset (replace with your dataset path)
data = pd.read_csv('Fraud.csv')

# Data Preprocessing: Label Encoding for non-numeric columns
label_encoder = LabelEncoder()
categorical_columns = ['type', 'nameOrig', 'nameDest']

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Separate legitimate and fraudulent transactions
legit = data[data.isFraud == 0]
fraud = data[data.isFraud == 1]

# Perform class balancing (you can choose undersampling or oversampling)
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns=["isFraud"], axis=1)
y = data["isFraud"]
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Create a sidebar section for model selection
st.sidebar.subheader("Select Classification Model and Hyperparameters")

# Model selection
model_name = st.sidebar.selectbox("Select Classification Model", ("Logistic Regression", "SVM", "K-Nearest Neighbors"))
model = None

if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "SVM":
    model = SVC()
elif model_name == "K-Nearest Neighbors":
    model = KNeighborsClassifier()

# Hyperparameter selection
if model:
    if st.sidebar.checkbox("Tune Hyperparameters"):
        if model_name == "Logistic Regression":
            C = st.sidebar.slider("Regularization Parameter (C)", 0.001, 100.0, 1.0)
            model = LogisticRegression(C=C)

        if model_name == "SVM":
            C = st.sidebar.slider("Regularization Parameter (C)", 0.001, 100.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
            model = SVC(C=C, kernel=kernel)

        if model_name == "K-Nearest Neighbors":
            n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Training model
if model:
    model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.write(f"Selected Model: {model_name}")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")

st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for the user to enter feature values
feature_names = X.columns
input_features = []

for feature_name in feature_names:
    input_value = st.number_input(f"Enter {feature_name}", value=0.0)
    input_features.append(input_value)

# Create a button to submit input and get prediction
submit = st.button("Submit")
prediction = None  # Initialize prediction

if submit:
    # Convert input features to a NumPy array
    features = np.array(input_features).reshape(1, -1)
    # Make prediction
    if model:
        prediction = model.predict(features)

if prediction is not None:
    if prediction[0] == 0:
        st.markdown("<p style='color: green; font-size: 36px;'>Legitimate transaction</p>", unsafe_allow_html=True)
        st.markdown(":smile:", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red; font-size=36px;'>Fraudulent transaction</p>", unsafe_allow_html=True)
        st.markdown(":cry:", unsafe_allow_html=True)

# Add a checkbox in the sidebar to show/hide the model comparison
show_model_comparison = st.sidebar.checkbox("Show Model Comparison")

# Model Comparison Section (at the bottom of the page)
if show_model_comparison:
    st.write("Model Comparison")

    # Create a list of models for comparison
    models = ["Logistic Regression", "SVM", "K-Nearest Neighbors"]

    # Create lists for training and testing accuracies (modify with actual values)
    train_accuracies = [train_acc, train_acc, train_acc]
    test_accuracies = [test_acc, test_acc, test_acc]

    fig_comparison, ax_comparison = plt.subplots(figsize=(10, 6))
    ax_comparison.bar(models, train_accuracies, color='b', alpha=0.7, label='Training Accuracy')
    ax_comparison.bar(models, test_accuracies, color='g', alpha=0.7, label='Testing Accuracy')
    ax_comparison.set_xlabel("Model")
    ax_comparison.set_ylabel("Accuracy")
    ax_comparison.set_title("Model Comparison")
    ax_comparison.legend()
    st.pyplot(fig_comparison)
