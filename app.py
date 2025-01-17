import streamlit as st
import pandas as pd
import joblib

# Load the pipeline (preprocessor + model)
pipeline = joblib.load("LGMBbest_modelexport1.pkl")

# Title of the app
st.title("Bank Marketing Campaign Prediction")

# Sidebar for user input
st.sidebar.header("Customer Input Features")

# Function to get user input
def get_user_input():
    # Collect numerical inputs
    age = st.sidebar.slider("Age", 18, 100, 30)  
    balance = st.sidebar.number_input("Account Balance ($)",min_value= -6847,
    max_value= 66663,
    value=0,
    step=100)  
    campaign = st.sidebar.slider("Number of Contacts", 1, 3400, 15)  

    # Collect categorical inputs
    job = st.sidebar.selectbox(
        "Job Type",
        ["white-collar ", "blue-collar", "retired ", "self-employed ", "not-working ", "unknown"])
    housing = st.sidebar.selectbox("Has Housing Loan?", ["yes", "no"])  
    loan = st.sidebar.selectbox("Has Personal Loan?", ["yes", "no"]) 
    contact = st.sidebar.selectbox("Contact Type", ["cellular", "telephone", "unknown"])  
    month = st.sidebar.selectbox(
        "Last Contact Month",
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],  
    )
    poutcome = st.sidebar.selectbox(
        "Previous Campaign Outcome",
        ["success", "not-success", "unknown"], 
    )

    # Create input data as a dictionary
    input_data = {
        "age": age,
        "balance": balance,
        "campaign": campaign,
        "job": job,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
    }

    # Return as DataFrame
    return pd.DataFrame([input_data])

# Get user input
input_df = get_user_input()

# Debugging: Display the input DataFrame
st.subheader("Customer Input Features")
st.write(input_df)

# Prediction
if st.button("Predict"):
    try:
        # Pass the raw input to the pipeline
        prediction = pipeline.predict(input_df)  # Encoded output (0 or 1)
        prediction_proba = pipeline.predict_proba(input_df)  # Probabilities

        # Display prediction result
        st.subheader("Prediction")
        st.write(
            "The customer will likely to subscribe to a term deposit"
            if prediction[0] == 1
            else "The customer will likely not subscribe to a term deposit"
        )

        # Display prediction probabilities
        st.subheader("Prediction Probability")
        st.write(f"Probability of subscribing: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of not subscribing: {prediction_proba[0][0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
