import streamlit as st
import pandas as pd
import pickle
import xgboost # Make sure xgboost is installed

# --- GLOBAL CONSTANTS ---
# Mean height of toddlers from training data (in cm)
TINGGI_MEAN = 86.0

# --- LOAD MODEL ---
try:
    with open("model_xgboost1.pkl", "rb") as f:
        model = pickle.load(f)

    # Check and get feature order
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
        st.warning(
            "Model does not have `feature_names_in_`. "
            "Using hard-coded feature order; please ensure it's correct."
        )
except FileNotFoundError:
    st.error("Error: 'model_xgboost1.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- UI ---
st.set_page_config(
    page_title="Prediksi Stunting pada Balita",
    page_icon="👶",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 5px #ccc;
    }
    .stButton>button {
        background-color: #28B463;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        box-shadow: 2px 2px 5px #aaa;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #239B56;
        box-shadow: 3px 3px 8px #999;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
        box-shadow: 1px 1px 3px #eee;
    }
    .prediction-result {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-radius: 15px;
        background-color: #EBF5FB; /* Latar belakang tetap */
        border: 2px solid #A9CCE3;
        color: #000000; /* UBAH INI ke hitam untuk teks umum di dalam kotak */
        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
    }
    .prediction-result span { /* Teks hasil prediksi spesifik (misal: "Stunting") */
        color: #000000; /* UBAH INI juga ke hitam agar hasil utama terlihat jelas */
    }
    .stAlert {
        border-radius: 10px;
    }
    .stSuccess {
        background-color: #D4EDDA;
        color: #155724;
        border-color: #C3E6CB;
        font-weight: bold;
    }
    .stWarning {
        background-color: #FFF3CD;
        color: #856404;
        border-color: #FFECB5;
    }
    .stError {
        background-color: #F8D7DA;
        color: #721C24;
        border-color: #F5C6CB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-header'>👶 Prediksi Stunting pada Balita 👶</h1>", unsafe_allow_html=True)
st.write(
    "Aplikasi ini membantu memprediksi status gizi balita (stunting/normal) berdasarkan data antropometri."
)

st.markdown("---")

# Input section
with st.container():
    st.subheader("Data Balita")
    col1, col2 = st.columns(2)

    with col1:
        umur = st.slider("Umur (bulan)", 0, 60, 24, help="Usia balita dalam bulan.")
        berat = st.number_input("Berat Badan (kg)", 2.0, 30.0, 10.0, 0.1, help="Berat badan balita dalam kilogram.")

    with col2:
        jk_text = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], help="Pilih jenis kelamin balita.")
        tinggi = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, 80.0, 0.1, help="Tinggi badan balita dalam centimeter.")

st.markdown("---")

# --- PREDICTION ---
if st.button("Prediksi Status Gizi"):
    if umur == 0 and berat == 2.0 and tinggi == 30.0:
        st.warning("Mohon masukkan data balita yang valid untuk prediksi.")
    else:
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

        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            label_map = {0: "Severely Stunting", 1: "Stunting", 2: "Normal", 3: "Tinggi"}
            prediction_label = label_map.get(pred, f'Kelas {pred}')

            # Pastikan teks hasil prediksi terlihat jelas
            st.markdown(
                f"<div class='prediction-result'>Status Gizi: <br> <span>{prediction_label}</span></div>",
                unsafe_allow_html=True
            )

            st.subheader("Detail Probabilitas:")
            proba_data = {label_map.get(i, f"Kelas {i}"): f"{p*100:.2f}%" for i, p in enumerate(proba)}
            st.json(proba_data) # Use st.json for better display of dictionary

            st.info("Prediksi ini didasarkan pada model Machine Learning. Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis dan saran lebih lanjut.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("---")
st.caption("Aplikasi ini dibuat untuk tujuan edukasi dan informasi. Jangan digunakan sebagai pengganti nasihat medis profesional.")
