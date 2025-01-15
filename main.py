import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Data Dummy ---
kapasitas_yard = 5000  # TEU

# Data service kapal window
data_service = {
    'service': ['Service ' + str(i+1) for i in range(10)],
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
def generate_data_simulasi(df_service, skenario):
    df = df_service.copy()

    # Simulasi jumlah kapal per service (distribusi Poisson)
    df['jumlah_kapal'] = df['kapal_per_minggu'].apply(
        lambda x: np.random.poisson(x)
    )

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
        axis=1
    )

    # Hitung jumlah container ekspor dan impor
    df['container_impor'] = df['jumlah_container'] * df['persen_impor']
    df['container_ekspor'] = df['jumlah_container'] * (1 - df['persen_impor'])

    return df

# --- Fungsi untuk Menghitung Yard Occupancy ---
def hitung_yard_occupancy(df, rata_rata_impor_truk, std_impor_truk,
                          rata_rata_ekspor_truk, std_ekspor_truk):
    # Simulasi tren ekspor-impor truk (distribusi normal)
    impor_truk = np.random.normal(rata_rata_impor_truk, std_impor_truk)
    ekspor_truk = np.random.normal(rata_rata_ekspor_truk, std_ekspor_truk)

    # Hitung total container ekspor dan impor
    total_impor = df['container_impor'].sum() + impor_truk
    total_ekspor = df['container_ekspor'].sum() + ekspor_truk

    return total_impor, total_ekspor

# --- Simulasi Monte Carlo ---
n_simulasi = 1000  # Jumlah simulasi

# Inisialisasi array untuk menyimpan hasil simulasi
hasil_simulasi = np.zeros((n_simulasi, 4, 2))  # 4 skenario, 2 output (ekspor, impor)

for i in range(n_simulasi):
    for j, skenario in enumerate([1, 2, 3, 4]):
        # Generate data simulasi
        df = generate_data_simulasi(df_service, skenario)

        # Hitung yard occupancy
        impor, ekspor = hitung_yard_occupancy(
            df, rata_rata_impor_truk, std_impor_truk,
            rata_rata_ekspor_truk, std_ekspor_truk
        )

        hasil_simulasi[i, j, 0] = ekspor
        hasil_simulasi[i, j, 1] = impor

# --- Analisis Hasil ---
# Hitung statistik deskriptif (rata-rata, deviasi standar)
rata_rata_simulasi = np.mean(hasil_simulasi, axis=0)
std_simulasi = np.std(hasil_simulasi, axis=0)

# --- Visualisasi ---
# Contoh: Visualisasi yard occupancy ekspor untuk setiap skenario
plt.figure(figsize=(10, 6))
for j, skenario in enumerate([1, 2, 3, 4]):
    plt.hist(hasil_simulasi[:, j, 0], bins=20, alpha=0.5,
             label=f'Skenario {skenario}')
plt.xlabel('Yard Occupancy Ekspor (TEU)')
plt.ylabel('Frekuensi')
plt.title('Distribusi Yard Occupancy Ekspor untuk Setiap Skenario')
plt.legend()
plt.show()

# --- Output ---
print('Rata-rata Yard Occupancy:')
print(rata_rata_simulasi)

print('\nDeviasi Standar Yard Occupancy:')
print(std_simulasi)
