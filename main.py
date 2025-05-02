
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Container Forecast", layout="wide")
st.title("📦 Forecast Jumlah Container Masuk dan Keluar")

@st.cache_data
def load_data_in():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/EXPORT%2024-25.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE IN'] = pd.to_datetime(df['GATE IN'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['GATE IN'])
    return df

@st.cache_data
def load_data_out():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/IMPORT%2024-25.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE OUT'] = pd.to_datetime(df['GATE OUT'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['GATE OUT'])
    return df

try:
    df_in = load_data_in()
    df_out = load_data_out()

    tab1, tab2 = st.tabs(["Container Masuk", "Container Keluar"])

    for label, df, gate_col, tab in [("Masuk", df_in, 'GATE IN', tab1), ("Keluar", df_out, 'GATE OUT', tab2)]:
        with tab:
            st.subheader(f"📅 Forecast Container {label}")

            df_daily = df.groupby(df[gate_col].dt.date).size().reset_index(name='container_count')
            df_daily[gate_col] = pd.to_datetime(df_daily[gate_col])
            ts = df_daily.set_index(gate_col)['container_count']

            start_date = st.date_input(f"Tanggal mulai forecast ({label})", value=ts.index[-1] + pd.Timedelta(days=1), key=f"start_{label}")
            end_date = st.date_input(f"Tanggal akhir forecast ({label})", value=start_date + pd.Timedelta(days=30), key=f"end_{label}")
            forecast_days = (end_date - start_date).days + 1

            train = ts[:-forecast_days] if len(ts) > forecast_days else ts
            test = ts[-forecast_days:] if len(ts) > forecast_days else None

            model = ARIMA(train, order=(5, 1, 2))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=forecast_days)
            forecast_index = pd.date_range(start=start_date, periods=forecast_days)

            st.subheader(f"📊 Hasil Forecast Container {label}")
            fig, ax = plt.subplots(figsize=(12, 5))
            ts.plot(ax=ax, label='Data Aktual')
            pd.Series(forecast.values, index=forecast_index).plot(ax=ax, label='Forecast')
            ax.legend()
            st.pyplot(fig)

            if test is not None and len(test) == forecast_days:
                forecast_series = pd.Series(forecast.values, index=forecast_index)
                test_aligned = test.reindex(forecast_series.index)

                combined = pd.concat([test_aligned, forecast_series], axis=1)
                combined.columns = ['actual', 'forecast']
                combined = combined.dropna()

                if not combined.empty:
                    mae = mean_absolute_error(combined['actual'], combined['forecast'])
                    nonzero_mask = combined['actual'] != 0
                    if nonzero_mask.sum() > 0:
                        mape = np.mean(np.abs((combined['actual'][nonzero_mask] - combined['forecast'][nonzero_mask]) / combined['actual'][nonzero_mask])) * 100
                        st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** {mape:.2f}%**")
                    else:
                        st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** Tidak bisa dihitung (semua nilai aktual = 0)**")
                else:
                    st.warning("Tidak cukup data yang bisa dibandingkan untuk menghitung MAE dan MAPE.")

            st.subheader(f"📋 Tabel Forecast Container {label}")
            st.dataframe(pd.DataFrame({
                'Tanggal': forecast_index,
                'Forecast Jumlah Container': forecast.values
            }))

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
