import streamlit as st
import pandas as pd
import pickle

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict_churn(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    # Create a DataFrame with the same columns and order as the training data
    data = {'credit_score': [credit_score],
            'country': [country],
            'gender': [gender],
            'age': [age],
            'tenure': [tenure],
            'balance': [balance],
            'products_number': [products_number],
            'credit_card': [credit_card],
            'active_member': [active_member],
            'estimated_salary': [estimated_salary]}
    df = pd.DataFrame(data)

    # Apply one-hot encoding (must match the training data)
    df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

    # Ensure all columns from training data are present, fill missing with 0
    # This step is crucial if new data might not have all categories from training
    # For simplicity in this example, we assume all categories are present in the input form.
    # In a real application, you'd need to handle this more robustly.
    # For now, we'll just scale the available columns.

    # Select and order columns to match the training data before scaling
    # Based on the training data, the columns after one-hot encoding and dropping customer_id were:
    # 'credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male'
    
    # Create a dictionary to hold the data with all possible columns
    processed_data = {}
    for col in ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']:
        processed_data[col] = df[col].iloc[0]

    # Handle one-hot encoded columns - ensure they exist and are in the correct order
    for country_col in ['country_Germany', 'country_Spain']:
        processed_data[country_col] = df.get(country_col, 0).iloc[0] # Use .get() with default 0

    for gender_col in ['gender_Male']:
         processed_data[gender_col] = df.get(gender_col, 0).iloc[0] # Use .get() with default 0

    # Convert the dictionary to a DataFrame with explicit column order
    processed_df = pd.DataFrame([processed_data], columns=['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male'])


    # Scale the numerical features
    scaled_data = scaler.transform(processed_df)
    scaled_df = pd.DataFrame(scaled_data, columns=processed_df.columns)


    # Make prediction
    prediction = model.predict(scaled_df)
    return prediction[0]

# Streamlit App
st.title("Bank Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

# Input fields
credit_score = st.slider("Credit Score", 350, 850, 650)
country = st.selectbox("Country", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.slider("Age", 18, 92, 38)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Balance", value=0.0)
products_number = st.slider("Number of Products", 1, 4, 1)
credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input("Estimated Salary", value=0.0)


if st.button("Predict Churn"):
    result = predict_churn(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)
    if result == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")
