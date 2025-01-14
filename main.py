import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Contoh model
import os

def main():
    # --- Judul Aplikasi ---
    st.title('Yard Occupancy Ratio Forecast')

    # --- Print working directory ---
    st.write(os.getcwd())

    # --- Membaca Data ---
    data_yor = pd.read_excel('data/data_yor.xlsx', sheet_name='YOR')
    data_kapal = pd.read_excel('data/data_kapal.xlsx', sheet_name='Data kapal')

    # --- Data Cleaning ---
    # (Sesuaikan dengan kondisi data Anda)
    data_yor.dropna(inplace=True)
    data_kapal.dropna(inplace=True)

    # --- Data Preprocessing ---
    # Scaling (Import dan Export terpisah)
    scaler_import = MinMaxScaler()
    data_yor['Import YOR_scaled'] = scaler_import.fit_transform(data_yor[['Import YOR']])

    scaler_export = MinMaxScaler()
    data_yor['Export YOR_scaled'] = scaler_export.fit_transform(data_yor[['Export YOR']])

    scaler_kapal = StandardScaler()
    data_kapal['Total_scaled'] = scaler_kapal.fit_transform(data_kapal[['Total']])  # Ganti 'Total' dengan nama kolom yang sesuai

    # Encoding (jika ada kolom kategorikal di data_kapal)
    # data_kapal = pd.get_dummies(data_kapal, columns=['Nama Kolom Kategorikal'])

    # Feature Engineering
    data_gabungan = pd.merge(data_yor, data_kapal, on='Date')  # Pastikan nama kolom tanggal sama di kedua file

    # --- Eksplorasi Data ---
    st.subheader('Analisis Deskriptif')
    st.write(data_yor.describe())
    st.write(data_kapal.describe())

    st.subheader('Visualisasi Data')
    fig, ax = plt.subplots()
    ax.plot(data_yor['Date'], data_yor['Import YOR'], label='Import YOR')  # Pastikan nama kolom tanggal benar
    ax.plot(data_yor['Date'], data_yor['Export YOR'], label='Export YOR')  # Pastikan nama kolom tanggal benar
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('YOR')
    ax.set_title('Tren YOR')
    ax.legend()
    st.pyplot(fig)

    # --- Korelasi ---
    st.subheader('Korelasi')
    st.write(data_gabungan.corr())

    # --- Pembagian Data (Import dan Export terpisah) ---
    X_import = data_gabungan[['Import YOR_scaled', 'Total_scaled']] 
    y_import = data_gabungan['Import YOR']
    X_train_import, X_test_import, y_train_import, y_test_import = train_test_split(
        X_import, y_import, test_size=0.2, random_state=42
    )

    X_export = data_gabungan[['Export YOR_scaled', 'Total_scaled']] 
    y_export = data_gabungan['Export YOR']
    X_train_export, X_test_export, y_train_export, y_test_export = train_test_split(
        X_export, y_export, test_size=0.2, random_state=42
    )

    # --- Pemodelan (Contoh dengan Linear Regression) ---
    st.subheader('Pemodelan')
    # Model Import
    model_import = LinearRegression()
    model_import.fit(X_train_import, y_train_import)

    # Model Export
    model_export = LinearRegression()
    model_export.fit(X_train_export, y_train_export)

    # --- Evaluasi dan Prediksi ---
    # ... (Tambahkan kode untuk evaluasi dan prediksi di sini) ...

if __name__ == '__main__':
    main()
