import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Container Forecast", layout="wide")
st.title("📦 Forecast Jumlah Container Masuk dan Keluar")

# ========================
# Load Data Masuk
# ========================
@st.cache_data
def load_data_in():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/EXPORT%20DATA%202024.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE IN'] = pd.to_datetime(df['GATE IN'], format='%d/%m/%Y %H:%M')
    return df

# ========================
# Load Data Keluar
# ========================
@st.cache_data
def load_data_out():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/IMPORT%20DATA%20FILTERED%202024.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE OUT'] = pd.to_datetime(df['GATE OUT'], format='%d/%m/%Y %H:%M')
    return df

try:
    df_in = load_data_in()
    df_out = load_data_out()

    tab1, tab2 = st.tabs(["Container Masuk", "Container Keluar"])

    for label, df, gate_col in [("Masuk", df_in, 'GATE IN'), ("Keluar", df_out, 'GATE OUT')]:
        with tab1 if label == "Masuk" else tab2:
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
                mae = mean_absolute_error(test, forecast)
                nonzero_mask = test != 0
                if nonzero_mask.sum() > 0:
                    mape = np.mean(np.abs((test[nonzero_mask] - forecast[nonzero_mask]) / test[nonzero_mask])) * 100
                    st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** {mape:.2f}%**")
                else:
                    st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** Tidak bisa dihitung (semua nilai aktual = 0)**")

            st.subheader(f"📋 Tabel Forecast Container {label}")
            st.dataframe(pd.DataFrame({
                'Tanggal': forecast_index,
                'Forecast Jumlah Container': forecast.values
            }))

            st.subheader(f"🔮 Forecasted Receiving Trends by Service (per DAY category) - Container {label}")
            if 'SERVICE OUT' in df.columns and 'DAY' in df.columns:
                service_day_actual = df.pivot_table(index='SERVICE OUT', columns='DAY', values=gate_col, aggfunc='count', fill_value=0)
                service_total_actual = service_day_actual.sum(axis=1)
                service_day_percentage = service_day_actual.div(service_total_actual, axis=0) * 100

                forecast_total = forecast.sum()
                service_share = service_total_actual / service_total_actual.sum()
                forecast_service_total = service_share * forecast_total

                forecast_service_day = pd.DataFrame()
                for service in service_day_percentage.index:
                    for day in service_day_percentage.columns:
                        forecast_value = forecast_service_total[service] * service_day_percentage.loc[service, day] / 100
                        forecast_service_day.loc[service, f"DAY {day}"] = forecast_value

                forecast_service_day_percentage = forecast_service_day.div(forecast_service_day.sum(axis=1), axis=0) * 100

                st.dataframe(forecast_service_day_percentage.style.format("{:.1f}%"))
            else:
                st.info("Kolom SERVICE atau DAY tidak ditemukan di dataset.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
