import os
st.write(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Membaca Data ---
data_yor = pd.read_excel('data_yor.xlsx', sheet_name='YOR')
data_kapal = pd.read_excel('data_kapal.xlsx', sheet_name='Data kapal')

# --- 2. Data Cleaning ---
# (Sesuaikan dengan kondisi data Anda)
data_yor.dropna(inplace=True)
data_kapal.dropna(inplace=True)

# --- 3. Data Preprocessing ---
# Scaling
scaler = MinMaxScaler()
data_yor['YOR_scaled'] = scaler.fit_transform(data_yor[['YOR']])

scaler = StandardScaler()
data_kapal['Total_scaled'] = scaler.fit_transform(data_kapal[['Total']])  # Ganti 'Total' dengan nama kolom yang sesuai

# Encoding (jika ada kolom kategorikal di data_kapal)
# data_kapal = pd.get_dummies(data_kapal, columns=['Nama Kolom Kategorikal'])

# Feature Engineering
data_gabungan = pd.merge(data_yor, data_kapal, on='Date')  # Pastikan nama kolom tanggal sama di kedua file

# --- 4. Eksplorasi Data ---
# Analisis Deskriptif
print(data_yor.describe())
print(data_kapal.describe())

# Visualisasi
plt.plot(data_yor['Date'], data_yor['YOR'])  # Ganti 'Date' dengan nama kolom tanggal di data_yor
plt.xlabel('Tanggal')
plt.ylabel('YOR')
plt.title('Tren YOR')
plt.show()

# Korelasi
print(data_gabungan.corr())

# --- 5. Pembagian Data ---
X = data_gabungan[['YOR_scaled', 'Total_scaled']]  # Ganti 'Total_scaled' dengan nama kolom yang sesuai
y = data_gabungan['YOR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Lanjut ke Pemodelan ---
# (Kita akan bahas di langkah berikutnya)
