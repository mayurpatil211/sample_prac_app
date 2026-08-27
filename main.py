import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🍷 Wine Quality Predictor")

st.write("Enter the wine properties below:")

fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

if st.button("Predict"):

    user_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    # Scale input
    user_data = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data)

    st.success(f"Predicted Wine Quality: {prediction[0]}")