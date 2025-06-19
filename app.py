import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# Load the main model
with open('model_xgboost1.pkl', 'rb') as file:
    model = pickle.load(file)

# Definisikan nilai tinggi_mean secara langsung
# Ini adalah rata-rata tinggi badan yang digunakan saat model Anda dilatih
tinggi_mean = 86.0 # Nilai tinggi_mean diatur ke 86.0

# Tentukan urutan fitur yang diharapkan oleh model
# Coba dapatkan nama fitur dari model jika tersedia
if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
    fitur_model = list(model.feature_names_in_)
else:
    # Jika model tidak menyimpan feature_names_in_, gunakan urutan fitur yang di-hardcode ini
    # PASTIKAN urutan ini sesuai dengan fitur yang digunakan saat training model
    fitur_model = ['Umur (bulan)', 'Jenis Kelamin', 'Berat Badan (kg)',
                   'Tinggi Badan (cm)', 'Tinggi di atas rata-rata']
    st.warning("Model tidak berisi 'feature_names_in_'. Menggunakan urutan fitur yang di-hardcode. "
               "Pastikan urutan ini sesuai dengan fitur pelatihan model Anda.")


# Judul aplikasi
st.title("Prediksi Stunting pada Balita")
st.markdown("Masukkan data berikut untuk mengetahui prediksi status gizi:")

# Form input data pengguna
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
berat_badan = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, step=0.1, value=10.0)
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, step=0.1, value=80.0)

# Definisikan pemetaan label kelas
# PENTING: Konfirmasi label ini dan urutannya sesuai dengan kelas output model Anda.
class_labels = {
    0: "Severely Stunting",
    1: "Stunting",
    2: "Normal",
    3: "Tinggi"
}


# Proses input
if st.button("Prediksi"):
    # Ubah jenis kelamin ke numerik
    jk_numeric = 1 if jenis_kelamin == 'Laki-laki' else 0

    # Buat fitur 'tinggi di atas rata-rata' menggunakan nilai tinggi_mean yang sudah ditentukan
    tinggi_di_atas_rata = 1 if tinggi_badan > tinggi_mean else 0

    # Buat DataFrame input
    data_input = pd.DataFrame([{
        'Umur (bulan)': umur,
        'Jenis Kelamin': jk_numeric,
        'Berat Badan (kg)': berat_badan,
        'Tinggi Badan (cm)': tinggi_badan,
        'Tinggi di atas rata-rata': tinggi_di_atas_rata
    }])

    # Sesuaikan urutan fitur agar sama dengan saat training
    try:
        data_input = data_input[fitur_model]
    except KeyError as e:
        st.error(f"Error: Fitur hilang dalam data masukan atau urutan fitur tidak cocok: {e}. "
                 "Harap periksa daftar 'fitur_model' dan pastikan semua kolom ada dan dinamai dengan benar.")
        st.stop()

    # Prediksi
    prediksi_index = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]

    # Tampilkan hasil
    predicted_status = class_labels.get(prediksi_index, f"Kelas Tidak Dikenal: {prediksi_index}")
    st.success(f"Prediksi Status Gizi: **{predicted_status}**")

    st.write("Probabilitas:")
    prob_dict = {}
    for i, prob in enumerate(probabilitas):
        label = class_labels.get(i, f"Kelas {i}")
        prob_dict[label] = f"{round(prob * 100, 2)}%"
    st.write(prob_dict)
