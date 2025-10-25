"Prediksi Stunting Pada Anak Balita Menggunakan Algoritma Extreme Gradient Boosting Dan Bayesian Optimization"
Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi status gizi balita berdasarkan data umur, jenis kelamin, dan tinggi badan.
Model yang digunakan adalah Extreme Gradient Boosting (XGBoost) yang dioptimasi menggunakan Bayesian Optimization (Optuna) untuk memperoleh performa terbaik.

Tujuan
Mengimplementasikan algoritma XGBoost untuk klasifikasi status gizi.
Melakukan optimasi hyperparameter menggunakan Optuna (Bayesian Optimization).
Menganalisis performa model melalui metrik akurasi, presisi, recall, dan f1-score.
Menyajikan hasil prediksi dalam bentuk aplikasi interaktif.

Dataset
Dataset diperoleh dari Kaggle https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows/code dengan atribut utama:
Umur (bulan)
Jenis Kelamin
Tinggi Badan (cm)
Status Gizi (label target)

Metodologi
Data Preprocessing (Pembersihan data, normalisasi, dan encoding).
Model Development (Implementasi XGBoost menggunakan XGBClassifier).
Hyperparameter Tuning (Optimasi menggunakan Optuna (Bayesian Optimization)).
Model Evaluation (Evaluasi dengan Confusion Matrix dan Classification Report).
Deployment (Model disimpan dalam format .sav dan diintegrasikan ke aplikasi Streamlit).

Hasil Eksperimen
Akurasi model setelah tuning: 0.9578
Algoritma terbaik: XGBoost + Bayesian Optimization
Kesimpulan: Optimasi dengan BO mampu meningkatkan performa model dibanding parameter default.

Akses Aplikasi 
https://stupred-sttuutututuut.streamlit.app/ 
