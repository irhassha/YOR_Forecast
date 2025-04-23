import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Container Forecast", layout="wide")
st.title("📦 Forecast Jumlah Container Masuk")

# Load file dari GitHub (ganti URL dengan path file abati di GitHub)
@st.cache_data
def load_data():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/EXPORT%20DATA%202024.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE IN'] = pd.to_datetime(df['GATE IN'], format='%d/%m/%Y %H:%M')
    return df

try:
    df = load_data()

    df_daily = df.groupby(df['GATE IN'].dt.date).size().reset_index(name='container_count')
    df_daily['GATE IN'] = pd.to_datetime(df_daily['GATE IN'])
    ts = df_daily.set_index('GATE IN')['container_count']

    # Pilih tanggal awal dan akhir forecast
    st.subheader("📅 Pilih Periode Forecast")
    start_date = st.date_input("Tanggal mulai forecast", value=ts.index[-1] + pd.Timedelta(days=1), min_value=ts.index[-1] + pd.Timedelta(days=1))
    end_date = st.date_input("Tanggal akhir forecast", value=start_date + pd.Timedelta(days=30), min_value=start_date)

    forecast_days = (end_date - start_date).days + 1

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
    forecast_index = pd.date_range(start=start_date, periods=forecast_days)

    # Plot
    st.subheader("📊 Hasil Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))
    ts.plot(ax=ax, label='Data Aktual')
    pd.Series(forecast.values, index=forecast_index).plot(ax=ax, label='Forecast')
    ax.legend()
    st.pyplot(fig)

    # Tampilkan error sebagai persentase
    if test is not None and len(test) == forecast_days:
        mae = mean_absolute_error(test, forecast)
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** {mape:.2f}%")

    # Tampilkan tabel forecast
    st.subheader("📋 Tabel Forecast")
    st.dataframe(pd.DataFrame({
        'Tanggal': forecast_index,
        'Forecast Jumlah Container': forecast.values
    }))

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
