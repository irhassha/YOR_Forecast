import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    # --- Judul Aplikasi ---
    st.title('Yard Occupancy Ratio Forecast')

    # --- Data Dummy ---
    np.random.seed(42)  # Untuk reproducibility
    n_samples = 100  # Jumlah sampel data
    dates = pd.date_range('2023-01-01', periods=n_samples)
    data_yor = pd.DataFrame({
        'Date': dates,
        'Import YOR': np.random.rand(n_samples) * 0.6,  # Import YOR antara 0-60%
        'Export YOR': np.random.rand(n_samples) * 0.8  # Export YOR antara 0-80%
    })
    data_kapal = pd.DataFrame({
        'Date': dates,
        'Total Teus': np.random.randint(500, 3000, size=n_samples)  # Total Teus antara 500-3000
    })

    # --- Data Preprocessing ---
    # Scaling (Import dan Export terpisah)
    scaler_import = MinMaxScaler()
    data_yor['Import YOR_scaled'] = scaler_import.fit_transform(data_yor[['Import YOR']])

    scaler_export = MinMaxScaler()
    data_yor['Export YOR_scaled'] = scaler_export.fit_transform(data_yor[['Export YOR']])

    # Scaling untuk kolom 'Total Teus' di data_kapal
    scaler_kapal = StandardScaler()
    data_kapal['Total Teus_scaled'] = scaler_kapal.fit_transform(data_kapal[['Total Teus']]) 

    # Encoding (jika ada kolom kategorikal di data_kapal)
    # data_kapal = pd.get_dummies(data_kapal, columns=['Nama Kolom Kategorikal'])

    # Feature Engineering
    data_gabungan = pd.merge(data_yor, data_kapal, on='Date')

    # --- Eksplorasi Data ---
    st.subheader('Analisis Deskriptif')
    st.write(data_yor.describe())
    st.write(data_kapal.describe())

    st.subheader('Visualisasi Data')
    fig, ax = plt.subplots()
    ax.plot(data_yor['Date'], data_yor['Import YOR'], label='Import YOR')
    ax.plot(data_yor['Date'], data_yor['Export YOR'], label='Export YOR')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('YOR')
    ax.set_title('Tren YOR')
    ax.legend()
    st.pyplot(fig)

    # --- Korelasi ---
    st.subheader('Korelasi')
    st.write(data_gabungan.corr())

    # --- Pembagian Data (Import dan Export terpisah) ---
    X_import = data_gabungan[['Import YOR_scaled', 'Total Teus_scaled']] 
    y_import = data_gabungan['Import YOR']
    X_train_import, X_test_import, y_train_import, y_test_import = train_test_split(
        X_import, y_import, test_size=0.2, random_state=42
    )

    X_export = data_gabungan[['Export YOR_scaled', 'Total Teus_scaled']] 
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

    # --- Evaluasi ---
    st.subheader('Evaluasi Model')
    # Prediksi pada data uji
    y_pred_import = model_import.predict(X_test_import)
    y_pred_export = model_export.predict(X_test_export)

    # Menghitung MAE dan RMSE
    mae_import = mean_absolute_error(y_test_import, y_pred_import)
    rmse_import = np.sqrt(mean_squared_error(y_test_import, y_pred_import))
    mae_export = mean_absolute_error(y_test_export, y_pred_export)
    rmse_export = np.sqrt(mean_squared_error(y_test_export, y_pred_export))

    st.write(f'MAE Import: {mae_import:.4f}')
    st.write(f'RMSE Import: {rmse_import:.4f}')
    st.write(f'MAE Export: {mae_export:.4f}')
    st.write(f'RMSE Export: {rmse_export:.4f}')

    # --- Prediksi ---
    # ... (Tambahkan kode untuk prediksi di sini) ...

if __name__ == '__main__':
    main()
