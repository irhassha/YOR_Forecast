import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Data Dummy ---
kapasitas_yard = 5000  # TEU

# Data service kapal window
data_service = {
    'service': ['Service ' + str(i + 1) for i in range(10)],
    'kapal_per_minggu': [2, 1, 3, 2, 1, 2, 1, 3, 2, 1],
    'kapasitas_kapal': [500, 1000, 300, 400, 800, 500, 700, 300, 400, 900],
    'persen_impor': [0.6, 0.7, 0.5, 0.8, 0.4, 0.6, 0.7, 0.5, 0.8, 0.4],
}
df_service = pd.DataFrame(data_service)

# Tren ekspor-impor truk
rata_rata_impor_truk = 200
std_impor_truk = 50
rata_rata_ekspor_truk = 150
std_ekspor_truk = 30

# --- Fungsi untuk Menghasilkan Data Simulasi ---


def generate_data_simulasi(df_service, skenario, n_hari):
    df = df_service.copy()

    # Simulasi jumlah kapal per service (distribusi Poisson)
    df['jumlah_kapal'] = df['kapal_per_minggu'].apply(
        lambda x: np.random.poisson(x * (n_hari // 7))
    )  # disesuaikan dengan jumlah hari

    # Simulasi delay kapal (distribusi uniform antara 0-2 hari)
    if skenario == 2:  # Hanya untuk skenario delay
        df['delay'] = np.random.uniform(0, 2, size=len(df))
    else:
        df['delay'] = 0

    # Simulasi jumlah container per kapal (distribusi normal)
    df['jumlah_container'] = df.apply(
        lambda row: np.random.normal(
            row['kapasitas_kapal'], row['kapasitas_kapal'] * 0.1
        ) * row['jumlah_kapal'],
        axis=1,
    )

    # Hitung jumlah container ekspor dan impor
    df['container_impor'] = df['jumlah_container'] * df['persen_impor']
    df['container_ekspor'] = df['jumlah_container'] * (1 - df['persen_impor'])

    return df


# --- Fungsi untuk Menghitung Yard Occupancy ---
def hitung_yard_occupancy(
    df, rata_rata_impor_truk, std_impor_truk, rata_rata_ekspor_truk, std_ekspor_truk, n_hari
):
    # Inisialisasi array untuk menyimpan yard occupancy harian
    yard_occupancy_impor = np.zeros(n_hari)
    yard_occupancy_ekspor = np.zeros(n_hari)

    for hari in range(n_hari):
        # Simulasi tren ekspor-impor truk (distribusi normal)
        impor_truk = np.random.normal(rata_rata_impor_truk, std_impor_truk)
        ekspor_truk = np.random.normal(rata_rata_ekspor_truk, std_ekspor_truk)

        # Hitung total container ekspor dan impor HARIAN
        total_impor = impor_truk
        total_ekspor = ekspor_truk

        # Akumulasi container dari kapal yang datang
        for index, row in df.iterrows():
            if hari >= row['delay'] and hari <= row['delay'] + 3:  # asumsi kapal sandar 3 hari
                total_impor += row['container_impor'] / 3  # bagi rata container per hari sandar
                total_ekspor += row['container_ekspor'] / 3

        yard_occupancy_impor[hari] = total_impor
        yard_occupancy_ekspor[hari] = total_ekspor

    return yard_occupancy_impor, yard_occupancy_ekspor


# --- Streamlit App ---
st.title("Prediksi Yard Occupancy")

# Input jumlah simulasi
n_simulasi = st.number_input(
    "Jumlah Simulasi", min_value=100, value=1000, step=100
)

# Input jumlah hari
n_hari = st.number_input("Jumlah Hari Prediksi", min_value=1, value=7, step=1)

# Pilihan skenario
skenario = st.selectbox(
    "Pilih Skenario",
    [
        "1. Kapal-kapal datang sesuai dengan windownya",
        "2. Jika terjadi delay di beberapa service kapal",
        "3. Jika dalam satu minggu ada 2 service kapal yang sama dikarenakan kapal di minggu sebelumnya delay",
        "4. Jika ada penambahan kapal diluar service yang 10 tadi",
    ],
)

# Tombol untuk menjalankan simulasi
if st.button("Jalankan Simulasi"):
    # Inisialisasi array untuk menyimpan hasil simulasi
    hasil_simulasi_impor = np.zeros((n_simulasi, n_hari))
    hasil_simulasi_ekspor = np.zeros((n_simulasi, n_hari))

    with st.spinner("Menjalankan simulasi..."):
        for i in range(n_simulasi):
            # Generate data simulasi
            df = generate_data_simulasi(
                df_service, skenario_mapping[skenario], n_hari
            )

            # Hitung yard occupancy
            impor, ekspor = hitung_yard_occupancy(
                df,
                rata_rata_impor_truk,
                std_impor_truk,
                rata_rata_ekspor_truk,
                std_ekspor_truk,
                n_hari,
            )

            hasil_simulasi_impor[i, :] = impor
            hasil_simulasi_ekspor[i, :] = ekspor

    # --- Analisis Hasil ---
    # Hitung statistik deskriptif (rata-rata, deviasi standar)
    rata_rata_impor = np.mean(hasil_simulasi_impor, axis=0)
    std_impor = np.std(hasil_simulasi_impor, axis=0)
    rata_rata_ekspor = np.mean(hasil_simulasi_ekspor, axis=0)
    std_ekspor = np.std(hasil_simulasi_ekspor, axis=0)

    # --- Visualisasi ---
    st.subheader("Visualisasi Hasil")

    # Visualisasi yard occupancy ekspor dan impor per hari
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_hari + 1), rata_rata_ekspor, label="Ekspor (Rata-rata)")
    ax.fill_between(
        range(1, n_hari + 1),
        rata_rata_ekspor - std_ekspor,
        rata_rata_ekspor + std_ekspor,
        alpha=0.2,
    )
    ax.plot(range(1, n_hari + 1), rata_rata_impor, label="Impor (Rata-rata)")
    ax.fill_between(
        range(1, n_hari + 1),
        rata_rata_impor - std_impor,
        rata_rata_impor + std_impor,
        alpha=0.2,
    )

    ax.set_xlabel("Hari")
    ax.set_ylabel("Yard Occupancy (TEU)")
    ax.set_title(f"Yard Occupancy per Hari (Skenario {skenario})")
    ax.legend()
    st.pyplot(fig)

    # --- Output ---
    st.subheader("Output")

    st.write("Rata-rata Yard Occupancy per Hari:")
    st.write("Ekspor:", rata_rata_ekspor)
    st.write("Impor:", rata_rata_impor)

    st.write("\nDeviasi Standar Yard Occupancy per Hari:")
    st.write("Ekspor:", std_ekspor)
    st.write("Impor:", std_impor)
