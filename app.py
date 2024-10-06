import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

model = load_model('model.h5')

with open('ohe_cb_person_default_on_file.pkl', 'rb') as file:
    ohe_default = pickle.load(file)

with open('ohe_loan_grade.pkl', 'rb') as file:
    ohe_loan_grade = pickle.load(file)

with open('ohe_person_home_ownership.pkl', 'rb') as file:
    ohe_home_ownership = pickle.load(file)

with open('ohe_loan_intent.pkl', 'rb') as file:
    ohe_loan_intent = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit App
st.title("Loan Approval Prediction")
age = st.slider('Age', 18, 80)
income = st.slider("Income", 10000, 1000000)
home_ownership = st.selectbox("Home Ownership", ohe_home_ownership.categories_[0]) #categorical
emp_length = st.slider('Employement Length', 0, 20)
loan_intent = st.selectbox("Loan Intent", ohe_loan_intent.categories_[0]) #categorical
loan_grade = st.selectbox("Loan Grade", ohe_loan_grade.categories_[0]) #categorical
loan_amt = st.slider("Loan Amount", 1000, 10000)
loan_interest_rate = st.slider("Interest rate", 2, 20)
loan_percent_income = st.slider("Loan % Income", 1, 20)
cb_person_default_on_file = st.selectbox("Default on file", ohe_default.categories_[0]) #categorical
cb_person_cred_hist_length = st.slider("Credict History length", 0, 20)

#Preparing the input data
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_emp_length': [emp_length],
    'loan_amnt': [loan_amt],
    'loan_int_rate': [loan_interest_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]  
})

## Adding the one hot encoder
home_owner_encode = ohe_home_ownership.transform([[home_ownership]])
home_owner_df = pd.DataFrame(home_owner_encode, columns=ohe_home_ownership.get_feature_names_out(['person_home_ownership']))

loan_intent_encode = ohe_loan_intent.transform([[loan_intent]])
loan_intent_df = pd.DataFrame(loan_intent_encode, columns=ohe_loan_intent.get_feature_names_out(['loan_intent']))

loan_grade_encode = ohe_loan_grade.transform([[loan_grade]])
loan_grade_df = pd.DataFrame(loan_grade_encode, columns=ohe_loan_grade.get_feature_names_out(['loan_grade']))

person_default = ohe_default.transform([[cb_person_default_on_file]])
person_default_df = pd.DataFrame(person_default, columns=ohe_default.get_feature_names_out(['cb_person_default_on_file']))

input_data = pd.concat([input_data.reset_index(drop=True), home_owner_df, loan_intent_df, loan_grade_df, person_default_df], axis=1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
predict_proba = prediction[0][0]

if predict_proba > 0.5:
    st.write(f"Customer Loan Approved. {predict_proba}")

else:
    st.write(f"Customer Loan not Approved. {predict_proba}")