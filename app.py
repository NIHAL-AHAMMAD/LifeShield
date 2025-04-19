from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Data Formation (random data)
data = {
    "Age": [18, 24, 32, 45, 67, 21, 29, 38, 54, 16],
    "Gender": [0, 1, 1, 0, 1, 0, 1, 0, 0, 1],   # Male=0, Female=1
    "Mental_health_history": [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "Social_isolation": [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    "Abuse": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "Financial_stability": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "Suicide": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],  # Suicide risk
    "Theft": [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],  # Crime type 1
    "Assault": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],  # Crime type 2
    "Fraud": [0, 0, 0, 1, 1, 0, 1, 0, 1, 0],  # Crime type 3
    "Vandalism": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # Crime type 4
    "Cybercrime": [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]  # Crime type 5
}

# Create DataFrame
data_sample = pd.DataFrame(data)

# Features and Targets
x = data_sample[["Age", "Gender", "Mental_health_history", "Social_isolation", "Abuse", "Financial_stability"]].dropna()
y_suicide = data_sample["Suicide"].dropna()
y_theft = data_sample["Theft"].dropna()
y_assault = data_sample["Assault"].dropna()
y_fraud = data_sample["Fraud"].dropna()
y_vandalism = data_sample["Vandalism"].dropna()
y_cybercrime = data_sample["Cybercrime"].dropna()

# Fit Models for each category
suicide_model = LinearRegression()
theft_model = LinearRegression()
assault_model = LinearRegression()
fraud_model = LinearRegression()
vandalism_model = LinearRegression()
cybercrime_model = LinearRegression()

suicide_model.fit(x, y_suicide)
theft_model.fit(x, y_theft)
assault_model.fit(x, y_assault)
fraud_model.fit(x, y_fraud)
vandalism_model.fit(x, y_vandalism)
cybercrime_model.fit(x, y_cybercrime)

# Error finding techniques
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def sum_of_squared_residuals(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def inverse_square_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    return np.sum(1 / (residuals ** 2 + 1e-10))  # Adding a small value to avoid division by zero

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = 0 if request.form['gender'].lower() == 'm' else 1
    mental_health = 1 if request.form['mental_health'] == 'yes' else 0
    social_isolation = 1 if request.form['social_isolation'] == 'yes' else 0
    abuse = 1 if request.form['abuse'] == 'yes' else 0
    financial_stability = 1 if request.form['financial_stability'] == 'yes' else 0

    new_data = np.array([[age, gender, mental_health, social_isolation, abuse, financial_stability]])

    new_suicide_prediction = (suicide_model.predict(new_data) >= 0.5).astype(int)
    new_theft_prediction = (theft_model.predict(new_data) >= 0.5).astype(int)
    new_assault_prediction = (assault_model.predict(new_data) >= 0.5).astype(int)
    new_fraud_prediction = (fraud_model.predict(new_data) >= 0.5).astype(int)
    new_vandalism_prediction = (vandalism_model.predict(new_data) >= 0.5).astype(int)
    new_cybercrime_prediction = (cybercrime_model.predict(new_data) >= 0.5).astype(int)

    predictions = {
        "suicide": new_suicide_prediction[0],
        "theft": new_theft_prediction[0],
        "assault": new_assault_prediction[0],
        "fraud": new_fraud_prediction[0],
        "vandalism": new_vandalism_prediction[0],
        "cybercrime": new_cybercrime_prediction[0]
    }

    # Calculate error metrics (only once)

    # Calculate error metrics
    mae = mean_absolute_error(y_suicide, suicide_model.predict(x))
    mse = mean_squared_error(y_suicide, suicide_model.predict(x))
    ssr = sum_of_squared_residuals(y_suicide, suicide_model.predict(x))
    inv_sq_res = inverse_square_residuals(y_suicide, suicide_model.predict(x))

    # Display Predictions
    prevention_steps = []
    if new_suicide_prediction == 1:
        prevention_steps.append("- Suicide: High risk. Seek immediate help from professionals and support systems.")
    if new_theft_prediction == 1:
        prevention_steps.append("- Theft: High risk. Secure your environment and seek guidance.")
    if new_assault_prediction == 1:
        prevention_steps.append("- Assault: High risk. Address anger issues through counseling.")
    if new_fraud_prediction == 1:
        prevention_steps.append("- Fraud: High risk. Strengthen moral integrity and avoid manipulation.")
    if new_vandalism_prediction == 1:
        prevention_steps.append("- Vandalism: High risk. Channel energies into positive outlets.")
    if new_cybercrime_prediction == 1:
        prevention_steps.append("- Cybercrime: High risk. Focus on ethical technology practices.")
    if (
        new_suicide_prediction == 0
        and new_theft_prediction == 0
        and new_assault_prediction == 0
        and new_fraud_prediction == 0
        and new_vandalism_prediction == 0
        and new_cybercrime_prediction == 0
    ):
        prevention_steps.append("- No major risks detected. Maintain positive habits and social connections.")

    return render_template('result_final_updated.html', predictions=predictions, prevention_steps=prevention_steps)

if __name__ == '__main__':
    app.run(debug=True)
