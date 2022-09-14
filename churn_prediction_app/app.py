from optparse import Values
from statistics import mode
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd


def load_model():
    return load('model/churn_model_v1.jb')

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout='centered',
    page_icon="🛃"
)

with st.form('form1', clear_on_submit=False):
    cscore = st.number_input('Credit Score', min_value= 0, max_value=1000, value=600)
    location = st.radio('Location',['France', 'Spain', 'Germany'])
    gender = st.radio('Gender',['Female','Male'])
    age = st.number_input("Age", min_value=18, max_value=100, value=33)
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=2)
    balance = st.number_input("Balance", min_value=0.0, max_value=9999999.0, value=100.0, step=.5)
    num_products = st.radio('Num of Products', [1,2,3,4])
    has_card = st.checkbox('Has Credit card', value = True)
    is_active = st.checkbox("Is Active member", value = True)
    est_salary = st.slider("Estimated Salary",min_value=1000.0, max_value=999999.0, value=2000.0,step=.5)
    btn = st.form_submit_button("Predict Customer Churn Status")

if btn:
    xinput = [{
        'CreditScore':cscore,
        'Geography': location,        
        'Gender':gender,              
        'Age':age,         
        'Tenure': tenure,          
        'Balance': balance,              
        'NumOfProducts': num_products,         
        'HasCrCard': int(has_card),            
        'IsActiveMember': int(is_active),       
        'EstimatedSalary':est_salary,
    }]
    xinput = pd.DataFrame(xinput)
    model = load_model()
    pred = model.predict(xinput)
    st.markdown('# will leave' if pred[0] == 0 else '# stay')