import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# --- Judul Aplikasi ---
st.title('Yard Occupancy Ratio Forecast')

# --- Membaca Data ---
data_yor = pd.read_excel('data/data_yor.xlsx', sheet_name='YOR')
data_kapal = pd.read_excel('data/data_kapal.xlsx', sheet_name='Data kapal')

# --- Data Cleaning dan Preprocessing ---
# ... (kode yang sama seperti sebelumnya)

# --- Eksplorasi Data ---
st.subheader('Analisis Deskriptif')
st.write(data_yor.describe())
st.write(data_kapal.describe())

st.subheader('Visualisasi Data')
fig, ax = plt.subplots()
ax.plot(data_yor['Date'], data_yor['YOR'])
ax.set_xlabel('Tanggal')
ax.set_ylabel('YOR')
ax.set_title('Tren YOR')
st.pyplot(fig)

# --- Lanjut ke Pemodelan ---
# ...
