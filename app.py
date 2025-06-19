import streamlit as st
import pandas as pd
import pickle
import xgboost  # pastikan xgboost ter‑install

# ------------------------------------------------------------------
# KONSTANTA GLOBAL
# ------------------------------------------------------------------
TINGGI_MEAN = 86.0  # cm – rata² tinggi balita pada data training

# ------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------
with open("model_xgboost1.pkl", "rb") as f:
    model = pickle.load(f)

# Cek & ambil urutan fitur
if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
    FEATURE_ORDER = list(model.feature_names_in_)
else:
    FEATURE_ORDER = [
        "Umur (bulan)",
        "Jenis Kelamin",
        "Berat Badan (kg)",
        "Tinggi Badan (cm)",
        "Tinggi di atas rata-rata",
    ]
    st.info(
        "Model tidak punya `feature_names_in_`. "
        "Menggunakan urutan fitur hard‑coded; pastikan sudah benar."
    )

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("Prediksi Stunting pada Balita")

umur = st.number_input("Umur (bulan)", 0, 60, 24)
jk_text = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
berat = st.number_input("Berat Badan (kg)", 2.0, 30.0, 10.0, 0.1)
tinggi = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, 80.0, 0.1)

# ------------------------------------------------------------------
# PREDIKSI
# ------------------------------------------------------------------
if st.button("Prediksi"):
    jk_num = 1 if jk_text == "Laki-laki" else 0
    tinggi_flag = 1 if tinggi > TINGGI_MEAN else 0

    X = pd.DataFrame(
        [[umur, jk_num, berat, tinggi, tinggi_flag]],
        columns=[
            "Umur (bulan)",
            "Jenis Kelamin",
            "Berat Badan (kg)",
            "Tinggi Badan (cm)",
            "Tinggi di atas rata-rata",
        ],
    )[FEATURE_ORDER]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label_map = {0: "Severely Stunting", 1: "Stunting", 2: "Normal", 3: "Tinggi"}
    st.success(f"**{label_map.get(pred, f'Kelas {pred}')}**")

    st.write({label_map.get(i, f"Kelas {i}"): f"{p*100:.2f}%" for i, p in enumerate(proba)})
