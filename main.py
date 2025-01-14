import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Yard Occupancy Ratio (YOR) Prediction")

# Step 1: Input Data
st.header("1. Masukkan Data Historis")

# Sample input fields for historical data
arrival_rate = st.number_input("Masukkan rata-rata kedatangan kontainer per hari:", min_value=0, value=50)
departure_rate = st.number_input("Masukkan rata-rata keberangkatan kontainer per hari:", min_value=0, value=40)
yard_capacity = st.number_input("Masukkan kapasitas yard (jumlah total kontainer yang dapat ditampung):", min_value=0, value=500)

# Step 2: Input Time Period for Prediction
st.header("2. Pilih Periode Waktu Prediksi")

# Input for number of days for prediction
days_to_predict = st.number_input("Pilih jumlah hari untuk prediksi YOR:", min_value=1, value=30)

# Step 3: Model Choice
st.header("3. Pilih Model Prediksi")

# Radio button for model selection
model_choice = st.radio("Pilih model untuk prediksi YOR:", ("Regresi Linier", "Model Dummy"))

# Step 4: Run Simulation
st.header("4. Jalankan Prediksi")

# Simulate prediction based on model choice
if model_choice == "Regresi Linier":
    # Create dummy historical data for training
    # Randomly generate arrival and departure rates
    historical_data = pd.DataFrame({
        'Arrival Rate': np.random.randint(40, 60, 100),
        'Departure Rate': np.random.randint(30, 50, 100),
        'Yard Occupancy Ratio': np.random.uniform(0.3, 0.9, 100)
    })

    # Define features and target
    X = historical_data[['Arrival Rate', 'Departure Rate']]
    y = historical_data['Yard Occupancy Ratio']

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make prediction based on user input
    input_data = pd.DataFrame({
        'Arrival Rate': [arrival_rate],
        'Departure Rate': [departure_rate]
    })

    predicted_yor = model.predict(input_data)[0]

    # Display result
    st.write(f"Prediksi Yard Occupancy Ratio (YOR) untuk {days_to_predict} hari ke depan adalah: {predicted_yor:.2f}")

    # Generate and display estimation plot
    st.subheader("Ilustrasi Grafik Estimasi YOR")
    plt.figure(figsize=(8, 6))

    # Estimasi berdasarkan input pengguna
    estimated_yor = predicted_yor * np.ones(days_to_predict)

    plt.plot(range(1, days_to_predict + 1), estimated_yor, label='Prediksi YOR', color='blue', linestyle='--')
    plt.xlabel("Hari")
    plt.ylabel("Yard Occupancy Ratio")
    plt.title("Estimasi Yard Occupancy Ratio (YOR) untuk 30 Hari Ke Depan")
    plt.legend()
    st.pyplot()

    # Calculate MSE for model evaluation
    mse = mean_squared_error(y, model.predict(X))
    st.write(f"Mean Squared Error (MSE) model: {mse:.2f}")

else:
    st.write("Model dummy: Hasil prediksi YOR hanya berdasarkan nilai rata-rata historis.")
    st.write(f"Prediksi YOR: {(arrival_rate - departure_rate) / yard_capacity:.2f}")

# Step 5: Export Results
st.header("5. Ekspor Hasil Prediksi")

# Provide export option
if st.button("Export Hasil Prediksi"):
    result = {
        "Hari untuk Prediksi": days_to_predict,
        "Prediksi YOR": predicted_yor if model_choice == "Regresi Linier" else (arrival_rate - departure_rate) / yard_capacity
    }
    df_result = pd.DataFrame([result])
    df_result.to_csv("prediksi_yor.csv", index=False)
    st.write("Hasil telah diekspor ke 'prediksi_yor.csv'.")
