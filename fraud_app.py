import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('models/fraud_detection_pipeline.pkl')

st.title('💳 FraudSense -- Fraud Detection Prediction App')
st.markdown('Enter transaction details below and click Predict.')

transaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input('Old Balance (Sender)', min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input('New Balance (Sender)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error('⚠️ This transaction **may be fraudulent**!')
    else:
        st.success('✅ This transaction looks safe.')
