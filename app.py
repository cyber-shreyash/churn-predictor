import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("telco_churn_model.pkl")

# -----------------------------
# Title
# -----------------------------

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Enter the customer's information to estimate their probability of churn."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    seniorcitizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col3:
    phoneservice = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

# -----------------------------
# Internet & Services
# -----------------------------

st.subheader("🌐 Internet & Services")

col1, col2, col3 = st.columns(3)

with col1:
    multiplelines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    internetservice = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    onlinesecurity = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    onlinebackup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:
    deviceprotection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

with col3:
    techsupport = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

col1, col2 = st.columns(2)

with col1:
    streamingtv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streamingmovies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# -----------------------------
# Billing Information
# -----------------------------

st.subheader("💳 Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

with col3:
    paperlessbilling = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

col1, col2 = st.columns(2)

with col1:
    monthlycharges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col2:
    totalcharges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------

if st.button(
    "🔍 Predict Churn",
    use_container_width=True
):

    # -------------------------
    # Binary Encoding
    # -------------------------

    gender_value = 1 if gender == "Male" else 0
    seniorcitizen_value = 1 if seniorcitizen == "Yes" else 0
    partner_value = 1 if partner == "Yes" else 0
    dependents_value = 1 if dependents == "Yes" else 0
    phoneservice_value = 1 if phoneservice == "Yes" else 0
    onlinesecurity_value = 1 if onlinesecurity == "Yes" else 0
    onlinebackup_value = 1 if onlinebackup == "Yes" else 0
    deviceprotection_value = 1 if deviceprotection == "Yes" else 0
    techsupport_value = 1 if techsupport == "Yes" else 0
    streamingtv_value = 1 if streamingtv == "Yes" else 0
    streamingmovies_value = 1 if streamingmovies == "Yes" else 0
    paperlessbilling_value = 1 if paperlessbilling == "Yes" else 0

    # -------------------------
    # Create 28 Features
    # -------------------------

    input_data = {
        "gender": gender_value,
        "seniorcitizen": seniorcitizen_value,
        "partner": partner_value,
        "dependents": dependents_value,
        "tenure": tenure,
        "phoneservice": phoneservice_value,
        "onlinesecurity": onlinesecurity_value,
        "onlinebackup": onlinebackup_value,
        "deviceprotection": deviceprotection_value,
        "techsupport": techsupport_value,
        "streamingtv": streamingtv_value,
        "streamingmovies": streamingmovies_value,
        "paperlessbilling": paperlessbilling_value,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,

        "multiplelines_No":
            1 if multiplelines == "No" else 0,

        "multiplelines_No phone service":
            1 if multiplelines == "No phone service" else 0,

        "multiplelines_Yes":
            1 if multiplelines == "Yes" else 0,

        "internetservice_DSL":
            1 if internetservice == "DSL" else 0,

        "internetservice_Fiber optic":
            1 if internetservice == "Fiber optic" else 0,

        "internetservice_No":
            1 if internetservice == "No" else 0,

        "contract_Month-to-month":
            1 if contract == "Month-to-month" else 0,

        "contract_One year":
            1 if contract == "One year" else 0,

        "contract_Two year":
            1 if contract == "Two year" else 0,

        "paymentmethod_Bank transfer (automatic)":
            1 if paymentmethod == "Bank transfer (automatic)" else 0,

        "paymentmethod_Credit card (automatic)":
            1 if paymentmethod == "Credit card (automatic)" else 0,

        "paymentmethod_Electronic check":
            1 if paymentmethod == "Electronic check" else 0,

        "paymentmethod_Mailed check":
            1 if paymentmethod == "Mailed check" else 0
    }

    # -------------------------
    # DataFrame
    # -------------------------

    input_df = pd.DataFrame([input_data])

    # -------------------------
    # Prediction
    # -------------------------

    probability = model.predict_proba(input_df)[0][1]

    # Final threshold selected during your testing
    threshold = 0.55

    prediction = 1 if probability >= threshold else 0

    # -------------------------
    # Display Result
    # -------------------------

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    # Probability bar
    st.progress(float(probability))

    st.caption(
        "Prediction threshold: 55%"
    )