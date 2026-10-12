"""
Stage 2 - Tahap 6.3: Pembangunan & Pengujian Model Support Vector Machine (SVM)
Referensi: Pertemuan 8 - Klasifikasi.pdf

Instruksi:
File ini berisi kerangka dan arahan (panduan komentar) terstruktur 
yang mengadopsi format dan gaya modul kuliah "Pertemuan 8 - Klasifikasi".
Silakan isi bagian '# TODO' di bawah ini untuk mengimplementasikan kode Anda.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul machine learning dan metrik evaluasi dari scikit-learn
# (Sesuai modul Hal. 1 & 9: SVC, train_test_split, confusion_matrix, accuracy_score, dsb.)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP PATH & FOLDER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
# Menentukan path direktori secara relatif agar dinamis dan portable
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

# Path dataset (bisa memilih data transformasi stage-2 atau data mentah diabetes.csv)
transform_dir = project_root / "stage-2" / "4-data-transformation" / "output"
resample_dir  = project_root / "stage-2" / "5-resampling" / "output"
raw_data_path = project_root / "diabetes.csv"

# Folder output untuk menyimpan gambar/screenshot dan file metrik CSV
output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "Outcome"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEMBACA DATA (LOAD DATA)
# Sesuai Modul Hal. 1: dataframe = pd.read_excel(...) / pd.read_csv(...)
# ─────────────────────────────────────────────────────────────────────────────
# TODO: Muat dataset.
# Skenario Pipeline (Sama seperti Decision Tree di 6-1):
#   - Data uji selalu memakai: test-transformed.csv
#   - Tiga varian data train:
#       1) Original / Baseline : train-transformed.csv
#       2) SMOTE               : train-oversampling-smote.csv
#       3) RUS (Undersampling) : train-undersampling-rus.csv

# Contoh pemuatan data (aktifkan jika memakai pipeline 3 varian):
df_test = pd.read_csv(transform_dir / "test-transformed.csv")
datasets = {
    "Original": pd.read_csv(transform_dir / "train-transformed.csv"),
    "SMOTE":    pd.read_csv(resample_dir  / "train-oversampling-smote.csv"),
    "RUS":      pd.read_csv(resample_dir  / "train-undersampling-rus.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PEMODELAN DENGAN SVM (SUPPORT VECTOR MACHINE)
# Sesuai Modul Hal. 9 & 12: model = SVC(), model.fit(), y_pred = model.predict()
# ─────────────────────────────────────────────────────────────────────────────
print("PEMODELAN DENGAN SVM".center(75, "="))
# TODO: Latih model SVM (Support Vector Classifier).
# Tips: Untuk data tabular medis, kernel='rbf' atau 'linear' lazim digunakan.
# Pastikan data fitur sudah terstandarisasi/ternormalisasi (seperti dari stage-2/4-data-transformation).

# Simpan hasil prediksi dan metrik:
models  = {}
y_preds = {}
metrics = []

# Contoh alur jika melatih 3 varian (Original, SMOTE, RUS):
X_test = df_test.drop(columns=[TARGET_COL]).values
y_test = df_test[TARGET_COL].values

for name, df_train in datasets.items():
    X_train = df_train.drop(columns=[TARGET_COL]).values
    y_train = df_train[TARGET_COL].values

    clf = CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), ensemble=False)
    clf.fit(X_train, y_train)
    models[name] = clf

    y_pred = clf.predict(X_test)
    y_preds[name] = y_pred

    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, zero_division=0)
    recall  = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"[{name}]: Precision: {presisi:.4f} | Recall: {recall:.4f}")

    # TODO: Tampilkan classification report
    print(f"Classification Report [{name}]:") 
    print(classification_report(y_test, y_pred))

    # TODO: Hitung Confusion Matrix & Elemen TP, TN, FP, FN
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0][0] * 1.0
    FP = cm[0][1] * 1.0
    FN = cm[1][0] * 1.0
    TP = cm[1][1] * 1.0

    # TODO: Hitung Sensitivity dan Specificity (Formula Modul Hal. 3):
    # Sensitivity (Recall Positif) = TP / (TP + FN) * 100
    # Specificity (Recall Negatif) = TN / (TN + FP) * 100
    sensivity = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    specifiy = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0

    metrics.append({
        "Model": name,
        "Akurasi": round(akurasi, 4),
        "Presisi": round(presisi, 4),
        "Recall": round(recall, 4),
        "Sensitivitas": round(sensivity, 4),
        "Specivisitas":  round(specifiy, 4),
        "F1-Score": round(f1, 4),
        "Train Size": len(df_train),
    })
# ─────────────────────────────────────────────────────────────────────────────
# 4. UJI COBA INPUT INTERAKTIF DATA PASIEN
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [col for col in df_test.columns if col != TARGET_COL]
CLASS_LABELS = {0: "Tidak Diabetes", 1: "Diabetes"}

# Hitung statistik dari data raw agar input pasien baru dapat di-transform
# dengan skala yang sama seperti saat training model.
raw_df = pd.read_csv(raw_data_path)
feature_mean = raw_df[FEATURE_COLUMNS].mean()
feature_std = raw_df[FEATURE_COLUMNS].std().replace(0, 1.0)


def predict_single_patient(model_name: str, patient_values: dict) -> tuple[int, float, float]:
    """Prediksi satu pasien baru dengan data raw dan transformasi yang konsisten."""
    df_patient = pd.DataFrame([patient_values], columns=FEATURE_COLUMNS)
    df_patient_scaled = (df_patient[FEATURE_COLUMNS] - feature_mean) / feature_std

    model = models[model_name]
    pred = int(model.predict(df_patient_scaled.values)[0])
    proba = model.predict_proba(df_patient_scaled.values)[0]
    return pred, float(proba[0]), float(proba[1])


# if __name__ == "__main__":
#     pilihan = input("Apakah ingin mencoba uji input data pasien baru? (y/n): ")
#     if pilihan.lower() in {"y", "ya", "yes"}:
#         print("\nMasukkan data pasien baru (nilai asli, bukan nilai z-score):")
#         patient_values = {}

#         for col in FEATURE_COLUMNS:
#             while True:
#                 try:
#                     value = float(input(f"{col}: "))
#                     patient_values[col] = value
#                     break
#                 except ValueError:
#                     print("Input tidak valid. Masukkan angka yang benar.")

#         # Pilih model terbaik berdasarkan akurasi untuk prediksi pasien baru.
#         best_model_name = max(metrics, key=lambda item: item["Akurasi"])["Model"]
#         pred_label, prob_no_diabetes, prob_diabetes = predict_single_patient(best_model_name, patient_values)

#         print("\n" + "=" * 60)
#         print(f"Model yang digunakan: {best_model_name}")
#         print(f"Probabilitas Tidak Diabetes: {prob_no_diabetes:.2%}")
#         print(f"Probabilitas Diabetes:      {prob_diabetes:.2%}")
#         print(f"Hasil Prediksi: {CLASS_LABELS[pred_label]}")
#         print("=" * 60)

#         if pred_label == 1:
#             print("Pasien Terprediksi Diabetes")
#         else:
#             print("Pasien Terprediksi Tidak Diabetes")
#     else:
#         print("Uji input data pasien baru dibatalkan.")
