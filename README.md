"Prediksi Stunting Pada Anak Balita Menggunakan Algoritma Extreme Gradient Boosting Dan Bayesian Optimization"
Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi status gizi balita berdasarkan data umur, jenis kelamin, dan tinggi badan.
Model yang digunakan adalah Extreme Gradient Boosting (XGBoost) yang dioptimasi menggunakan Bayesian Optimization (Optuna) untuk memperoleh performa terbaik.

Tujuan
Mengetahui tingkat akurasi serta kemampuan algoritma Extreme Gradient Boosting (XGBoost) dalam melakukan prediksi status stunting pada anak balita berdasarkan data antropometri.
Menganalisis kontribusi Bayesian Optimization dan teknik SMOTE dalam meningkatkan akurasi, kestabilan, serta kualitas performa model XGBoost pada kondisi data yang tidak seimbang.

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
Akurasi model setelah tuning: 0.981
Algoritma terbaik: XGBoost + Bayesian Optimization
Kesimpulan: Optimasi dengan BO mampu meningkatkan performa model dibanding parameter default.

Akses Aplikasi 
https://stupred-sttuutututuut.streamlit.app/ 
