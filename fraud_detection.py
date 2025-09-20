import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title("Fraud Detection Application")
st.markdown("Please enter the transaction details and use the predict button to check if the transaction is fraudulent or not.")

st.divider()

transaction_type = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT'])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
old_balance_orig = st.number_input("Old Balance (Sender)", min_value=0.0, value= 10000.0)
new_balance_orig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
old_balance_dest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
new_balance_dest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [old_balance_orig],
        'newbalanceOrig': [new_balance_orig],
        'oldbalanceDest': [old_balance_dest],
        'newbalanceDest': [new_balance_dest]
    })

    prediction = model.predict(input_data)[0]
    
    st.subheader(f"Prediction Result: '{int(prediction)}'")

    if prediction == 1:
        st.error("The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("The transaction is predicted to be NON-FRAUDULENT.")