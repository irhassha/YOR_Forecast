import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Container Forecast", layout="wide")
st.title("📦 Forecast Jumlah Container Masuk dan Keluar")

# ========================
# Load Data Masuk
# ========================
@st.cache_data
def load_data_in():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/EXPORT%20DATA%202024-2025.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE IN'] = pd.to_datetime(df['GATE IN'], format='%d/%m/%Y %H:%M', errors='coerce')
    return df.dropna(subset=['GATE IN'])

# ========================
# Load Data Keluar
# ========================
@st.cache_data
def load_data_out():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/IMPORT%20DATA%20FILTER%202024-2025.csv"
    df = pd.read_csv(url, delimiter=';')
    df['GATE OUT'] = pd.to_datetime(df['GATE OUT'], format='%d/%m/%Y %H:%M', errors='coerce')
    return df.dropna(subset=['GATE OUT'])

# ========================
# Fungsi untuk simpan ke Excel
# ========================
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

try:
    df_in = load_data_in()
    df_out = load_data_out()

    tab1, tab2 = st.tabs(["Container Masuk", "Container Keluar"])

    for label, df, gate_col, service_col in [
        ("Masuk", df_in, 'GATE IN', 'SERVICE'),
        ("Keluar", df_out, 'GATE OUT', 'SERVICE OUT')
    ]:
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

            df_forecast = pd.DataFrame({
                'Tanggal': forecast_index,
                'Forecast Jumlah Container': forecast.values
            })

            st.subheader(f"📋 Tabel Forecast Container {label}")
            st.dataframe(df_forecast)

            excel_data = convert_df_to_excel(df_forecast)
            st.download_button(
                label="📥 Download Forecast (Excel)",
                data=excel_data,
                file_name=f"forecast_{label.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ========== AVERAGE TRENDS BY SERVICE ==========
            st.subheader(f"📊 Average Receiving Trends by Service - Container {label}")
            if service_col in df.columns and 'DAY' in df.columns:
                st.markdown("Pilih rentang tanggal untuk menghitung tren:")
                start_filter = st.date_input(f"Mulai sample tren ({label})", value=ts.index.min(), key=f"trend_start_{label}")
                end_filter = st.date_input(f"Akhir sample tren ({label})", value=ts.index.max(), key=f"trend_end_{label}")

                df_filtered = df[(df[gate_col].dt.date >= start_filter) & (df[gate_col].dt.date <= end_filter)]
                pivot = df_filtered.pivot_table(index=service_col, columns='DAY', values=gate_col, aggfunc='count', fill_value=0)
                total_per_service = pivot.sum(axis=1)
                trend_percentage = pivot.div(total_per_service, axis=0) * 100
                trend_percentage.columns = [f'DAY {int(c)}' for c in trend_percentage.columns]
                st.dataframe(trend_percentage.style.format("{:.1f}%"))
            else:
                st.info("Kolom SERVICE atau DAY tidak ditemukan di dataset.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
