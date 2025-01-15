import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta

# --- Data Dummy ---
kapasitas_yard = 5000  # TEU

# Data service kapal window
data_service = {
    "service": ["Service " + str(i + 1) for i in range(10)],
    "kapal_per_minggu": [2, 1, 3, 2, 1, 2, 1, 3, 2, 1],
    "kapasitas_kapal": [500, 1000, 300, 400, 800, 500, 700, 300, 400, 900],
    "persen_impor": [0.6, 0.7, 0.5, 0.8, 0.4, 0.6, 0.7, 0.5, 0.8, 0.4],
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
    df["jumlah_kapal"] = df["kapal_per_minggu"].apply(
        lambda x: np.random.poisson(x * (n_hari // 7))
    )

    # Simulasi delay kapal (distribusi uniform antara 0-2 hari)
    if skenario == 2:  # Hanya untuk skenario delay
        df["delay"] = np.random.uniform(0, 2, size=len(df))
    else:
        df["delay"] = 0

    # --- Skenario 3: Kapal Susulan ---
    if skenario == 3:
        # Contoh: Tambahkan 1 kapal susulan untuk service 1 dan 3
        df.loc[[0, 2], "jumlah_kapal"] += 1

    # --- Skenario 4: Kapal Tambahan ---
    if skenario == 4:
        # Contoh: Tambahkan 2 kapal baru dengan data acak
        new_kapal = pd.DataFrame(
            {
                "service": ["Service 11", "Service 12"],
                "kapal_per_minggu": [1, 1],
                "kapasitas_kapal": np.random.randint(300, 1000, size=2),
                "persen_impor": np.random.rand(2),
                "jumlah_kapal": [1, 1],
                "delay": 0,
            }
        )
        df = pd.concat([df, new_kapal], ignore_index=True)

    # Simulasi jumlah container per kapal (distribusi normal)
    df["jumlah_container"] = df.apply(
        lambda row: np.random.normal(
            row["kapasitas_kapal"], row["kapasitas_kapal"] * 0.1
        )
        * row["jumlah_kapal"],
        axis=1,
    )

    # Hitung jumlah container ekspor dan impor
    df["container_impor"] = df["jumlah_container"] * df["persen_impor"]
    df["container_ekspor"] = df["jumlah_container"] * (1 - df["persen_impor"])

    return df


# --- Fungsi untuk Menghitung Yard Occupancy ---
def hitung_yard_occupancy(
    df,
    rata_rata_impor_truk,
    std_impor_truk,
    rata_rata_ekspor_truk,
    std_ekspor_truk,
    n_hari,
    existing_ekspor,
    existing_impor,
):
    # Inisialisasi array untuk menyimpan yard occupancy harian
    yard_occupancy_impor = np.zeros(n_hari)
    yard_occupancy_ekspor = np.zeros(n_hari)

    # Inisialisasi existing container
    yard_occupancy_impor[0] = existing_impor
    yard_occupancy_ekspor[0] = existing_ekspor

    for hari in range(1, n_hari):  # mulai dari hari ke-1 (besok)
        # Simulasi tren ekspor-impor truk (distribusi normal)
        impor_truk = np.random.normal(rata_rata_impor_truk, std_impor_truk)
        ekspor_truk = np.random.normal(rata_rata_ekspor_truk, std_ekspor_truk)

        # Hitung total container ekspor dan impor HARIAN
        total_impor = (
            impor_truk + yard_occupancy_impor[hari - 1]
        )  # tambahkan existing container
        total_ekspor = ekspor_truk + yard_occupancy_ekspor[hari - 1]

        # Akumulasi container dari kapal yang datang
        for index, row in df.iterrows():
            if (
                hari >= row["delay"] and hari <= row["delay"] + 3
            ):  # asumsi kapal sandar 3 hari
                total_impor += (
                    row["container_impor"] / 3
                )  # bagi rata container per hari sandar
                total_ekspor += row["container_ekspor"] / 3

        yard_occupancy_impor[hari] = total_impor
        yard_occupancy_ekspor[hari] = total_ekspor

    return yard_occupancy_impor, yard_occupancy_ekspor


# --- Streamlit App ---
st.title("Prediksi Yard Occupancy")

# Input kapasitas yard
kapasitas_yard_ekspor = st.number_input(
    "Kapasitas Yard Ekspor (TEU)", min_value=0, value=2500, step=100
)
kapasitas_yard_impor = st.number_input(
    "Kapasitas Yard Impor (TEU)", min_value=0, value=2500, step=100
)

# Input existing container
existing_ekspor = st.number_input(
    "Existing Container Ekspor (TEU)", min_value=0, value=500, step=100
)
existing_impor = st.number_input(
    "Existing Container Impor (TEU)", min_value=0, value=600, step=100
)

# Input jumlah simulasi
n_simulasi = st.number_input("Jumlah Simulasi", min_value=100, value=1000, step=100)

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

# Mapping skenario
skenario_mapping = {
    "1. Kapal-kapal datang sesuai dengan windownya": 1,
    "2. Jika terjadi delay di beberapa service kapal": 2,
    "3. Jika dalam satu minggu ada 2 service kapal yang sama dikarenakan kapal di minggu sebelumnya delay": 3,
    "4. Jika ada penambahan kapal diluar service yang 10 tadi": 4,
}

# Tombol untuk menjalankan simulasi
if st.button("Jalankan Simulasi"):
    # Inisialisasi array untuk menyimpan hasil simulasi
    hasil_simulasi_impor = np.zeros((n_simulasi, n_hari))
    hasil_simulasi_ekspor = np.zeros((n_simulasi, n_hari))

    with st.spinner("Menjalankan simulasi..."):
        for i in range(n_simulasi):
            # Generate data simulasi
            df = generate_data_simulasi(df_service, skenario_mapping[skenario], n_hari)

            # Hitung yard occupancy
            impor, ekspor = hitung_yard_occupancy(
                df,
                rata_rata_impor_truk,
                std_impor_truk,
                rata_rata_ekspor_truk,
                std_ekspor_truk,
                n_hari,
                existing_ekspor,
                existing_impor,
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

    # Set x-ticks to represent dates starting from tomorrow
    ax.set_xticks(range(1, n_hari + 1))
    ax.set_xticklabels(
        [
            (date.today() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, n_hari + 1)
        ],
        rotation=45,
    )

    st.pyplot(fig)

    # --- Output ---
    st.subheader("Output")

    # Membuat DataFrame untuk output
    tanggal = [(date.today() + timedelta(days=i)) for i in range(1, n_hari + 1)]
    df_output = pd.DataFrame(
        {
            "Tanggal": tanggal,
            "Rata-rata Ekspor (TEU)": rata_rata_ekspor,
            "Rata-rata Impor (TEU)": rata_rata_impor,
        }
    )

    # Menampilkan DataFrame
    st.dataframe(df_output.T)  # transpose agar tanggal ditampilkan menyamping

    st.write("\nDeviasi Standar Yard Occupancy per Hari:")
    st.write("Ekspor:", std_ekspor)
    st.write("Impor:", std_impor)
