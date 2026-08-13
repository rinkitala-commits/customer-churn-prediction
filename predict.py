import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("models/churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


def predict_churn(customer_data):
    """
    Predict whether a customer is likely to churn.
    """

    # Convert input dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Preprocess customer data
    customer_processed = preprocessor.transform(
        customer_df
    )

    # Make prediction
    prediction = model.predict(
        customer_processed
    )[0]

    # Get churn probability
    probability = model.predict_proba(
        customer_processed
    )[0][1]

    if prediction == 1:
        result = "Likely to Churn"
    else:
        result = "Not Likely to Churn"

    return result, probability


# Example customer
customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.50,
    "TotalCharges": 966.00
}

result, probability = predict_churn(customer)

print("\n===== CUSTOMER CHURN PREDICTION =====")
print("Prediction:", result)
print(f"Churn Probability: {probability:.2%}")