import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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
    # Scaling
    scaler = MinMaxScaler()
    data_yor['YOR_scaled'] = scaler.fit_transform(data_yor[['YOR']])

    scaler = StandardScaler()
    data_kapal['Total_scaled'] = scaler.fit_transform(data_kapal[['Total']])  # Ganti 'Total' dengan nama kolom yang sesuai

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
    ax.plot(data_yor['Date'], data_yor['YOR'])  # Pastikan nama kolom tanggal benar
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('YOR')
    ax.set_title('Tren YOR')
    st.pyplot(fig)

    # --- Korelasi ---
    st.subheader('Korelasi')
    st.write(data_gabungan.corr())

    # --- Pembagian Data ---
    X = data_gabungan[['YOR_scaled', 'Total_scaled']]  # Ganti 'Total_scaled' dengan nama kolom yang sesuai
    y = data_gabungan['YOR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Lanjut ke Pemodelan ---
    # ...

if __name__ == '__main__':
    main()
