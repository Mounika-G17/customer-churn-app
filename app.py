import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Title
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below:")

# Sidebar Inputs (Professional UI)
st.sidebar.header("Customer Inputs")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)

contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# Create input dataframe
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment,
}

input_df = pd.DataFrame([input_dict])

# Apply encoding
input_df = pd.get_dummies(input_df)

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Predict button
if st.button("Predict"):

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    prob = model.predict_proba(input_df)[0][1]

    # Result
    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer will not churn")

    # Probability display
    st.metric(label="Churn Probability", value=f"{prob:.2f}")

    # Explanation
    st.info("""
📊 Prediction Logic:
- Low tenure → Higher chance of churn
- High monthly charges → Higher risk
- Month-to-month contract → More churn
- Long-term contract → Less churn
""")
