import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below:")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# Create input dataframe
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}

input_df = pd.DataFrame([input_dict])

# Apply same encoding
input_df = pd.get_dummies(input_df)

# Match columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer will churn")
    else:
        st.success("✅ Customer will not churn")