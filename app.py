import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("models/churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Customer Churn Prediction")
st.markdown(
    """
    Predict whether a customer is likely to churn using
    Machine Learning.
    """
)
st.write(
    "Enter customer information below to predict "
    "the likelihood of churn."
)

st.info(
    "💡 The model uses customer demographics, services, "
    "contract details, and billing information to estimate "
    "churn probability."
)

st.divider()

st.header("👤 Customer Information")

# Customer details
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=1.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly_charges * tenure,
    step=10.0
)

st.divider()

# Prediction button
if st.button("🔮 Predict Churn", type="primary"):

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer_df = pd.DataFrame([customer_data])

    # Preprocess
    customer_processed = preprocessor.transform(
        customer_df
    )

    # Prediction
    prediction = model.predict(
        customer_processed
    )[0]

    probability = model.predict_proba(
        customer_processed
    )[0][1]

    st.subheader("👤 Customer Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Tenure", f"{tenure} months")

    with summary_col2:
        st.metric("Monthly Charges", f"₹{monthly_charges:.2f}")

    with summary_col3:
        st.metric("Contract", contract)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Likely to Churn")
        st.warning(
            "This customer shows a higher estimated risk of churn."
        )
    else:
        st.success("✅ Not Likely to Churn")
        st.info(
            "This customer shows a lower estimated risk of churn."
        )
        st.subheader("🎯 Churn Probability")

        st.progress(
            float(probability)
        )

        st.write(
            f"Estimated Churn Probability: "
            f"**{probability:.2%}**"
        )

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )
    st.progress(
        probability,
        text=f"Churn Risk: {probability:.2%}"
    )
    if probability < 0.30:
        risk_level = "🟢 Low Risk"
    elif probability < 0.60:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    st.subheader("🎯 Risk Assessment")
    st.write(risk_level)

    st.divider()

    st.subheader("🤖 About This Model")

    st.write(
        "This application uses a machine learning classification model "
        "trained on customer service and billing information to estimate "
        "the probability of customer churn."
    )

    st.caption(
        "Prediction is an estimate generated by the trained ML model "
        "and should not be treated as a guarantee."
    )