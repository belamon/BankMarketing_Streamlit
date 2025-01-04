import streamlit as st
import pandas as pd 
import joblib

#load the model 
model = joblib.load("LGMBbest_modelexport.pkl")

#Title of the App
st.title("Bank Marketing Campaign Prediction")

#Sidebar for user input 
st.sidebar.header("Customer Input Features")

#Function to get use input from the sidebar 

def get_user_input():
    age = st.sidebar.slider("Age", 18, 100, 30)
    job = st.sidebar.selectbox("Job", ["white-collar", "blue-collar", "retired", "self-employed", "not-working", "unknown"])
    balance = st.sidebar.number_input("Account Balance ($)", min_value = -5000, max_value=2000, value=0)
    housing = st.sidebar.selectbox("Has Housing Loan?", ["Yes", "No"])
    loan = st.sidebar.selectbox("Has Loan?", ["Yes", "No"])
    contact = st.sidebar.selectbox("Contact Type", ["Celullar", "Telephone", "Unknown"])
    month = st.sidebar.selectbox("Last Conctact", ["Jan", "Feb", "March", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    campaign = st.sidebar.slider("Mount of Contact Performed", min_value=1, max_value=4000, value=0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ["Successful", "Non-Successful", "Unknown"])

    #input dictionary 
    input_data = {
        "age": age,
        "job": job,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "campaign": campaign,
        "poutcome": poutcome

    }

    return pd.DataFrame([input_data])

#Get the user inout 
input_df = get_user_input()

#Display the user input 
st.subheader("Customer Input Features")
st.write(input_df)

#Make prediction 
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    #Display prediction result 
    st.subheader("Prediction")
    st.write("The customer wll subscribe to a term deposit" if prediction[0] == "yes" else "The customer will not subscribe to a term deposit")

    st.subheader("Prediction Probability")
    st.write(f"Probability of subscribing :{prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not subscribing: {prediction_proba[0][0]:.2f}")

