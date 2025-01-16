import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
import requests
from bs4 import BeautifulSoup

# --- Fungsi untuk Menghitung Lama Sandar Kapal ---
def hitung_lama_sandar(row):
    total_bongkar_muat = row["jumlah bongkar"] + row["jumlah muat"]
    lama_sandar = total_bongkar_muat / (
        row["crane deployment"] * row["performance crane"]
    )
    return lama_sandar


# --- Fungsi untuk Menghasilkan Data Simulasi ---
def generate_data_simulasi(df_kapal, skenario, n_hari):
    df = df_kapal.copy()

    # Simulasi delay kapal (distribusi uniform antara 0-2 hari)
    if skenario == 2:  # Hanya untuk skenario delay
        df["delay"] = np.random.uniform(0, 2, size=len(df))
    else:
        df["delay"] = 0

    # --- Skenario 3: Kapal Susulan ---
    # ... (logika skenario 3 - perlu disesuaikan dengan data aktual)

    # --- Skenario 4: Kapal Tambahan ---
    # ... (logika skenario 4 - perlu disesuaikan dengan data aktual)

    # Hitung lama sandar kapal
    df["lama sandar"] = df.apply(hitung_lama_sandar, axis=1)

    return df


# --- Fungsi untuk Menghitung Yard Occupancy ---
def hitung_yard_occupancy(df, df_truk, n_hari, existing_ekspor, existing_impor):
    # Inisialisasi array untuk menyimpan yard occupancy harian
    yard_occupancy_impor = np.zeros(n_hari)
    yard_occupancy_ekspor = np.zeros(n_hari)

    # Inisialisasi existing container
    yard_occupancy_impor[0] = existing_impor
    yard_occupancy_ekspor[0] = existing_ekspor

    for hari in range(1, n_hari):  # mulai dari hari ke-1 (besok)
        # Hitung rata-rata ekspor dan impor truk HARIAN dari data truk
        tanggal_hari = (date.today() + timedelta(days=hari)).strftime("%d/%m/%Y")
        try:  # Tangani error jika tanggal tidak ditemukan di data truk
            rata_rata_ekspor_truk = (
                df_truk[df_truk["tanggal"] == tanggal_hari]["export"].values[0]
            )
            rata_rata_impor_truk = (
                df_truk[df_truk["tanggal"] == tanggal_hari]["import"].values[0]
            )
        except IndexError:
            rata_rata_ekspor_truk = 150  # Nilai default jika tanggal tidak ditemukan
            rata_rata_impor_truk = 200  # Nilai default jika tanggal tidak ditemukan

        # Hitung total container ekspor dan impor HARIAN
        total_impor = yard_occupancy_impor[hari - 1] + rata_rata_impor_truk
        total_ekspor = yard_occupancy_ekspor[hari - 1] + rata_rata_ekspor_truk

        # Akumulasi container dari kapal yang datang
        for index, row in df.iterrows():
            if hari >= row["delay"] and hari <= row["delay"] + row["lama sandar"]:
                total_impor += row["jumlah bongkar"] / row["lama sandar"]
                total_ekspor += row["jumlah muat"] / row["lama sandar"]

        # Kurangi total_impor dengan jumlah kontainer yang dibawa truk impor
        total_impor -= rata_rata_impor_truk

        # Kurangi total_ekspor dengan jumlah kontainer yang dimuat ke kapal (ekspor)
        for index, row in df.iterrows():
            if hari >= row["delay"] and hari <= row["delay"] + row["lama sandar"]:
                total_ekspor -= row["jumlah muat"] / row["lama sandar"]

        yard_occupancy_impor[hari] = total_impor
        yard_occupancy_ekspor[hari] = total_ekspor

    return yard_occupancy_impor, yard_occupancy_ekspor


# --- Fungsi untuk Mengambil Data Kapal dari Website ---
def ambil_data_kapal_website(status_kapal=["REGISTER"]):  # Terima list status
    # URL website
    url = "https://www.npct1.co.id/vessel-schedule"

    # Mengambil kode HTML website
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Mencari tabel jadwal kapal
    table = soup.find("table")

    # Mengambil data dari tabel dan menyimpannya dalam list of dictionaries
    data = []
    for row in table.find_all("tr")[1:]:  # Skip baris header
        columns = row.find_all("td")

        # if columns[status_index].text.strip() in status_kapal:
        data.append(
            {
                "Vessel Name": columns[0].text.strip(),
                "Voyage No": columns[1].text.strip(),
                "Service": columns[2].text.strip(),
                "ETA": columns[3].text.strip(),
                "ETD": columns[4].text.strip(),
                "status kapal": columns[5].text.strip(),
                "Closing Date": columns[6].text.strip(),
                "Status": columns[7].text.strip(),
            }
        )

    # Membuat DataFrame dari data yang diekstrak
    df_kapal = pd.DataFrame(data)

    return df_kapal


# --- Eksekusi Fungsi ---
if __name__ == "__main__":
    # Ambil data dengan status ACTIVE atau REGISTER
    df_kapal = ambil_data_kapal_website(status_kapal=["ACTIVE", "REGISTER"])

    # Tampilkan hasil
    print("\nData Kapal dengan Status ACTIVE atau REGISTER:")
    print(df_kapal)

    # Simpan ke file CSV (opsional)
    df_kapal.to_csv("data_kapal_filtered.csv", index=False)
    print("\nData disimpan ke 'data_kapal_filtered.csv'")


# --- Streamlit App ---
st.title("Prediksi Yard Occupancy")

# --- Upload Data Kapal ---
st.subheader("Upload Data Kapal")
upload_choice = st.radio(
    "Pilih sumber data kapal:", ("Upload dari Excel", "Ambil dari Website")
)

if upload_choice == "Upload dari Excel":
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
    if uploaded_file is not None:
        df_kapal = pd.read_excel(uploaded_file)  # Baca file Excel
    else:
        st.warning("Silakan upload data kapal terlebih dahulu.")
        st.stop()  # Hentikan eksekusi jika tidak ada file yang diupload
elif upload_choice == "Ambil dari Website":
    with st.spinner("Mengambil data kapal dari website..."):
        df_kapal = ambil_data_kapal_website()  # Tidak perlu parameter status

# --- Menampilkan Data Kapal ---
st.write("Data Kapal:")
st.write(df_kapal)  # Menampilkan data kapal yang diupload / diambil dari website

# --- Upload Data Truk ---
st.subheader("Upload Data Truk")
uploaded_file_truk = st.file_uploader("Pilih file Excel (Data Truk)", type="xlsx")

# ---  Perbaikan:  Cek apakah df_kapal sudah terdefinisi ---
if df_kapal is not None and uploaded_file_truk is not None:
    df_truk = pd.read_excel(uploaded_file_truk)  # Baca data truk
    st.write("Data Truk:")
    st.write(df_truk)  # Menampilkan data truk yang diupload

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
                df = generate_data_simulasi(
                    df_kapal, skenario_mapping[skenario], n_hari
                )

                # Hitung yard occupancy (gunakan df_truk)
                impor, ekspor = hitung_yard_occupancy(
                    df, df_truk, n_hari, existing_ekspor, existing_impor
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

        # Membuat list tanggal
        tanggal = [
            (date.today() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, n_hari + 1)
        ]

        # Membuat DataFrame untuk output
        df_output = pd.DataFrame(
            {
                "Rata-rata Ekspor (TEU)": rata_rata_ekspor,
                "Deviasi Standar Ekspor": std_ekspor,
                "Rata-rata Impor (TEU)": rata_rata_impor,
                "Deviasi Standar Impor": std_impor,
            },
            index=tanggal,  # Menggunakan tanggal sebagai index
        )

        # Menampilkan DataFrame
        st.dataframe(df_output.T)  # transpose agar mudah dibaca

        # --- Tabel Bongkar Muat per Hari ---
        st.subheader("Tabel Bongkar Muat per Hari")

        # Inisialisasi DataFrame untuk tabel bongkar muat
        df_bongkar_muat = pd.DataFrame(
            columns=["Tanggal", "Total Bongkar (TEU)", "Total Muat (TEU)"]
        )

        for hari in range(1, n_hari):
            tanggal_hari = (date.today() + timedelta(days=hari)).strftime("%Y-%m-%d")
            total_bongkar = 0
            total_muat = 0

            for index, row in df.iterrows():
                if hari >= row["delay"] and hari <= row["delay"] + row["lama sandar"]:
                    total_bongkar += row["jumlah bongkar"] / row["lama sandar"]
                    total_muat += row["jumlah muat"] / row["lama sandar"]

            # Tambahkan data ke DataFrame menggunakan .loc
            df_bongkar_muat.loc[len(df_bongkar_muat)] = {
                "Tanggal": tanggal_hari,
                "Total Bongkar (TEU)": total_bongkar,
                "Total Muat (TEU)": total_muat,
            }

        st.dataframe(df_bongkar_muat)

else:
    st.warning("Silakan upload data truk terlebih dahulu.")
