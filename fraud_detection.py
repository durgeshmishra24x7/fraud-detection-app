import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('fraud_detection_model.pkl')

# --- Page Config ---
st.set_page_config(
    page_title="Fraud Detection Application",
    page_icon="💳",
    layout="centered",
)

# --- Custom Banking Theme ---
card_style = """
    <style>
    /* App background and font */
    .stApp {
        background: linear-gradient(135deg, #e6f0fa 0%, #f7fbff 100%);
        color: #1a237e;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }
    /* Card-like container for form */
    .stForm {
        background: #ffffffcc;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,64,128,0.08);
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
    }
    /* Header styling */
    .st-emotion-cache-10trblm {
        color: #004080 !important;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #b3c6e0;
    }
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #004080 0%, #1976d2 100%);
        color: #fff;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,64,128,0.10);
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1976d2 0%, #004080 100%);
        color: #fff;
    }
    .predict-btn button {
        width: 100%;
    }
    /* Input fields */
    .stNumberInput input, .stSelectbox div {
        border-radius: 6px;
        border: 1px solid #b3c6e0;
        background: #f4f8fb;
        font-size: 16px;
    }
    /* Divider styling */
    .st-emotion-cache-1avcm0n {
        border-top: 2px solid #1976d2;
        margin: 1.5rem 0;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
"""

st.markdown(card_style, unsafe_allow_html=True)

# --- Header ---
st.title("🏦 Transaction Fraud Detector")
st.markdown(
    """
    Welcome to the **Bank Fraud Detection Dashboard**  
    Enter transaction details below and click **Predict**  
    to check if the transaction is **Fraudulent** or **Legitimate**.
    """,
    unsafe_allow_html=True,
)

st.divider()

# --- Input Form ---
with st.form("fraud_form"):
    st.subheader("💼 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ['PAYMENT', 'TRANSFER', 'CASH_OUT']
        )
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )
        old_balance_orig = st.number_input(
            "Old Balance (Sender)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:
        new_balance_orig = st.number_input(
            "New Balance (Sender)",
            min_value=0.0,
            value=9000.0,
            step=500.0
        )
        old_balance_dest = st.number_input(
            "Old Balance (Receiver)",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
        new_balance_dest = st.number_input(
            "New Balance (Receiver)",
            min_value=0.0,
            value=0.0,
            step=100.0
        )

    # Wrap submit button in a unique container
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    submit = st.form_submit_button("Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
if submit:
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [old_balance_orig],
        'newbalanceOrig': [new_balance_orig],
        'oldbalanceDest': [old_balance_dest],
        'newbalanceDest': [new_balance_dest]
    })

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error("⚠️ The transaction is predicted to be **FRAUDULENT**.")
    else:
        st.success("✅ The transaction is predicted to be **LEGITIMATE**.")
