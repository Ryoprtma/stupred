import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# 1. Load the main model
with open('model_xgboost1.pkl', 'rb') as file:
    model = pickle.load(file)

# 2. DEFINE tinggi_mean HERE
# This is the average height your model used to create the 'Tinggi di atas rata-rata' feature.
# REPLACE 'YOUR_ACTUAL_MEAN_HEIGHT_VALUE' with the specific number (e.g., 75.0, 82.5, etc.)
tinggi_mean = YOUR_ACTUAL_MEAN_HEIGHT_VALUE # <--- THIS LINE IS ABSOLUTELY REQUIRED AND MUST HAVE A NUMBER

# 3. Determine feature order (as discussed before)
if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
    fitur_model = list(model.feature_names_in_)
else:
    fitur_model = ['Umur (bulan)', 'Jenis Kelamin', 'Berat Badan (kg)',
                   'Tinggi Badan (cm)', 'Tinggi di atas rata-rata']
    st.warning("Model does not contain 'feature_names_in_'. Using a hardcoded feature order. "
               "Please ensure this order matches your model's training features.")

# ... (rest of your Streamlit app code) ...

# Judul aplikasi
st.title("Prediksi Stunting pada Balita")
st.markdown("Masukkan data berikut untuk mengetahui prediksi status gizi:")

# Form input data pengguna
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
berat_badan = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, step=0.1, value=10.0)
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, step=0.1, value=80.0)

# Define the class labels mapping
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

    # Buat fitur 'tinggi di atas rata-rata'
    # This line now expects 'tinggi_mean' to be defined
    tinggi_di_atas_rata = 1 if tinggi_badan > tinggi_mean else 0 # THIS IS LINE 27 OR CLOSE TO IT

    # ... (rest of the prediction logic) ...
    data_input = pd.DataFrame([{
        'Umur (bulan)': umur,
        'Jenis Kelamin': jk_numeric,
        'Berat Badan (kg)': berat_badan,
        'Tinggi Badan (cm)': tinggi_badan,
        'Tinggi di atas rata-rata': tinggi_di_atas_rata
    }])

    try:
        data_input = data_input[fitur_model]
    except KeyError as e:
        st.error(f"Error: Missing feature in input data or feature order mismatch: {e}. "
                 "Please check the 'fitur_model' list and ensure all columns are present and correctly named.")
        st.stop()

    prediksi_index = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]

    predicted_status = class_labels.get(prediksi_index, f"Unknown Class Index: {prediksi_index}")
    st.success(f"Prediksi Status Gizi: **{predicted_status}**")

    st.write("Probabilitas:")
    prob_dict = {}
    for i, prob in enumerate(probabilitas):
        label = class_labels.get(i, f"Class {i}")
        prob_dict[label] = f"{round(prob * 100, 2)}%"
    st.write(prob_dict)
