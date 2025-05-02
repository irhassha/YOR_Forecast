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
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/EXPORT%20DATA%202024.csv"
    df = pd.read_csv(url, delimiter=';', on_bad_lines='skip')
    df['GATE IN'] = pd.to_datetime(df['GATE IN'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['GATE IN'])

# ========================
# Load Data Keluar
# ========================
@st.cache_data
def load_data_out():
    url = "https://github.com/irhassha/YOR_Forecast/raw/refs/heads/main/IMPORT%2024-25.csv"
    df = pd.read_csv(url, delimiter=';', on_bad_lines='skip')
    df['GATE OUT'] = pd.to_datetime(df['GATE OUT'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['GATE OUT'])

try:
    df_in = load_data_in()
    df_out = load_data_out()

    tab1, tab2 = st.tabs(["Container Masuk", "Container Keluar"])

    for label, df, gate_col, service_col in [("Masuk", df_in, 'GATE IN', 'SERVICE'), ("Keluar", df_out, 'GATE OUT', 'SERVICE OUT')]:
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

            if test is not None:
                forecast_series = pd.Series(forecast.values, index=forecast_index)
                test = test[test.index.isin(forecast_index)]
                forecast_series = forecast_series[forecast_series.index.isin(test.index)]

                combined = pd.concat([test, forecast_series], axis=1).dropna()
                combined.columns = ['actual', 'forecast']

                if not combined.empty:
                    mae = mean_absolute_error(combined['actual'], combined['forecast'])
                    nonzero_mask = combined['actual'] != 0
                    if nonzero_mask.sum() > 0:
                        mape = np.mean(np.abs((combined['actual'][nonzero_mask] - combined['forecast'][nonzero_mask]) / combined['actual'][nonzero_mask])) * 100
                        st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** {mape:.2f}%**")
                    else:
                        st.markdown(f"**📉 MAE:** {mae:.2f} | **MAPE:** Tidak bisa dihitung (semua nilai aktual = 0)**")
                else:
                    st.warning("Tidak ada data yang bisa dibandingkan untuk MAE/MAPE.")

            st.subheader(f"📋 Tabel Forecast Container {label}")
            forecast_df = pd.DataFrame({
                'Tanggal': forecast_index,
                'Forecast Jumlah Container': forecast.values
            })
            st.dataframe(forecast_df)

            # Download forecast table as Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                forecast_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button(
                label="📥 Download Forecast (Excel)",
                data=output.getvalue(),
                file_name=f"forecast_container_{label.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader(f"📊 Average Trends per Service (berdasarkan sample tanggal) - Container {label}")
            if service_col in df.columns and 'DAY' in df.columns:
                min_date, max_date = df[gate_col].min().date(), df[gate_col].max().date()
                selected_range = st.date_input(f"Pilih rentang tanggal untuk sample trend ({label})", [min_date, max_date], key=f"trend_{label}")
                if len(selected_range) == 2:
                    filtered = df[(df[gate_col].dt.date >= selected_range[0]) & (df[gate_col].dt.date <= selected_range[1])]
                    if not filtered.empty:
                        trend = filtered.pivot_table(index=service_col, columns='DAY', values=gate_col, aggfunc='count', fill_value=0)
                        total = trend.sum(axis=1)
                        trend_pct = trend.div(total, axis=0) * 100
                        st.dataframe(trend_pct.style.format("{:.1f}%"))
                    else:
                        st.warning("Tidak ada data untuk tanggal yang dipilih.")
            else:
                st.info("Kolom SERVICE atau DAY tidak ditemukan di dataset.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
