import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Container Forecast", layout="wide")
st.title("📦 Forecast Jumlah Container Masuk")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, delimiter=';')
        df['GATE IN'] = pd.to_datetime(df['GATE IN'], format='%d/%m/%Y %H:%M')
        
        df_daily = df.groupby(df['GATE IN'].dt.date).size().reset_index(name='container_count')
        df_daily['GATE IN'] = pd.to_datetime(df_daily['GATE IN'])
        ts = df_daily.set_index('GATE IN')['container_count']

        # Pilih horizon forecast
        forecast_days = st.slider("Pilih berapa hari ke depan untuk forecast", min_value=7, max_value=90, value=30)

        # Split data
        if len(ts) > forecast_days:
            train = ts[:-forecast_days]
            test = ts[-forecast_days:]
        else:
            train = ts
            test = None

        # Fit ARIMA
        model = ARIMA(train, order=(5, 1, 2))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        # Plot
        st.subheader("📊 Hasil Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ts.plot(ax=ax, label='Data Aktual')
        pd.Series(forecast.values, index=forecast_index).plot(ax=ax, label='Forecast')
        ax.legend()
        st.pyplot(fig)

        # Tampilkan error
        if test is not None and len(test) == forecast_days:
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            st.markdown(f"**📉 MAE:** {mae:.2f} | **RMSE:** {rmse:.2f}")

        # Tampilkan tabel forecast
        st.subheader("📋 Tabel Forecast")
        st.dataframe(pd.DataFrame({
            'Tanggal': forecast_index,
            'Forecast Jumlah Container': forecast.values
        }))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
else:
    st.info("Silakan upload file CSV berisi data container.")
