"""
Aplikasi Prediksi Risiko Diabetes - FrontEnd Streamlit
Model: Support Vector Machine (SVM) - RBF Kernel dengan Random Under-Sampling (RUS)
Mata Kuliah: Praktikum Pembelajaran Mesin (UTS)
Program Studi: D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Diabetes Risk Screener - SVM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan premium dan bersih
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .result-box-healthy {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 22px;
        color: #14532d;
    }
    .result-box-diabetes {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px;
        color: #7f1d1d;
    }
    .result-box-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 22px;
        color: #78350f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. LOAD MODEL & STATISTIK SKALER
# ==============================================================================
@st.cache_resource
def load_model_bundle():
    current_dir = Path(__file__).resolve().parent
    bundle_path = current_dir / "svm_bundle.pkl"

    if not bundle_path.exists():
        # Fallback ke output stage-2 jika file di app belum tersedia
        alt_path = current_dir.parent / "stage-2" / "6-3-svm-and-testing" / "output" / "svm_bundle.pkl"
        if alt_path.exists():
            bundle_path = alt_path
        else:
            raise FileNotFoundError(f"File model tidak ditemukan di {bundle_path} maupun {alt_path}")

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


try:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    feature_mean = bundle["feature_mean"]
    feature_std = bundle["feature_std"]
    class_labels = bundle.get("class_labels", {0: "Tidak Diabetes", 1: "Diabetes"})
except Exception as e:
    st.error(f"Gagal memuat model Machine Learning: {e}")
    st.stop()


# ==============================================================================
# 3. SIDEBAR: METRIK MODEL & PRESET DATA
# ==============================================================================
with st.sidebar:
    st.markdown("### 🩺 Sistem Skrining Diabetes")
    st.markdown(
        """
        <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 10px; border-radius: 4px; font-size: 0.88rem;">
            <b>Model:</b> Support Vector Machine (SVM)<br>
            <b>Kernel:</b> Radial Basis Function (RBF)<br>
            <b>Strategi:</b> Random Under-Sampling (RUS)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Performa Pada Data Uji")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Akurasi", "75.84%")
        st.metric("Recall (Klinis)", "76.47%")
    with col_m2:
        st.metric("F1-Score", "68.42%")
        st.metric("Presisi", "61.90%")

    st.caption("Recall 76.47% meminimalkan *False Negative* (pasien diabetes yang luput terdeteksi).")

    st.divider()

    st.markdown("#### ⚡ Preset Uji Cepat")
    preset_choice = st.selectbox(
        "Pilih Contoh Profil Pasien:",
        [
            "Isi Manual (Kustom)",
            "Pasien A (Risiko Rendah / Sehat)",
            "Pasien B (Risiko Tinggi / Diabetes)",
            "Pasien C (Kondisi Pre-Diabetes / Borderline)",
        ],
    )

    # Nilai default berdasarkan preset
    presets = {
        "Isi Manual (Kustom)": {
            "Pregnancies": 1,
            "Glucose": 105,
            "BloodPressure": 72,
            "SkinThickness": 24,
            "Insulin": 90,
            "BMI": 24.5,
            "DiabetesPedigreeFunction": 0.35,
            "Age": 28,
        },
        "Pasien A (Risiko Rendah / Sehat)": {
            "Pregnancies": 1,
            "Glucose": 88,
            "BloodPressure": 68,
            "SkinThickness": 20,
            "Insulin": 75,
            "BMI": 22.1,
            "DiabetesPedigreeFunction": 0.22,
            "Age": 24,
        },
        "Pasien B (Risiko Tinggi / Diabetes)": {
            "Pregnancies": 6,
            "Glucose": 168,
            "BloodPressure": 86,
            "SkinThickness": 36,
            "Insulin": 195,
            "BMI": 38.5,
            "DiabetesPedigreeFunction": 0.88,
            "Age": 54,
        },
        "Pasien C (Kondisi Pre-Diabetes / Borderline)": {
            "Pregnancies": 3,
            "Glucose": 138,
            "BloodPressure": 78,
            "SkinThickness": 28,
            "Insulin": 135,
            "BMI": 29.4,
            "DiabetesPedigreeFunction": 0.52,
            "Age": 42,
        },
    }
    active_values = presets[preset_choice]

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <b>Universitas Airlangga</b><br>
            Fakultas Vokasi — D4 Teknik Informatika<br>
            Praktikum Pembelajaran Mesin
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 4. AREA UTAMA & TAB APLIKASI
# ==============================================================================
st.markdown('<div class="main-header">Sistem Deteksi Dini & Skrining Risiko Diabetes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Aplikasi inferensi klinis berbasis Machine Learning (SVM-RUS) dengan normalisasi Z-score real-time.</div>',
    unsafe_allow_html=True,
)

tab_prediksi, tab_evaluasi, tab_panduan = st.tabs([
    "🔍 Formulir & Prediksi Pasien",
    "📈 Evaluasi Model SVM",
    "📖 Panduan Parameter Medis"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: FORMULIR INPUT & HASIL INFERENSI
# ──────────────────────────────────────────────────────────────────────────────
with tab_prediksi:
    st.markdown("##### Masukkan Parameter Data Medis Pasien")
    st.write("Semua nilai diisi dalam satuan pengukuran klinis standar (angka asli):")

    with st.form("form_pasien"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###### 👤 Profil Demografi & Fisik")
            age = st.number_input(
                "Usia Pasien (Tahun)",
                min_value=18,
                max_value=100,
                value=int(active_values["Age"]),
                help="Usia pasien dalam hitungan tahun (dewasa).",
            )
            pregnancies = st.number_input(
                "Jumlah Kehamilan (Pregnancies)",
                min_value=0,
                max_value=20,
                value=int(active_values["Pregnancies"]),
                help="Jumlah riwayat kehamilan (isi 0 jika laki-laki atau belum pernah hamil).",
            )
            bmi = st.number_input(
                "Indeks Massa Tubuh / BMI (kg/m²)",
                min_value=10.0,
                max_value=70.0,
                value=float(active_values["BMI"]),
                step=0.1,
                format="%.1f",
                help="Berat badan (kg) / [Tinggi badan (m)]^2. Normal: 18.5 - 24.9.",
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function (DPF)",
                min_value=0.01,
                max_value=3.00,
                value=float(active_values["DiabetesPedigreeFunction"]),
                step=0.01,
                format="%.2f",
                help="Skor riwayat diabetes pada silsilah keluarga genetik pasien.",
            )

        with col2:
            st.markdown("###### 🔬 Pemeriksaan Klinis & Laboratorium")
            glucose = st.number_input(
                "Kadar Glukosa Plasma 2 Jam (mg/dL)",
                min_value=40,
                max_value=350,
                value=int(active_values["Glucose"]),
                help="Konsentrasi glukosa plasma darah 2 jam setelah tes toleransi glukosa oral (OGTT).",
            )
            blood_pressure = st.number_input(
                "Tekanan Darah Diastolik (mmHg)",
                min_value=30,
                max_value=160,
                value=int(active_values["BloodPressure"]),
                help="Tekanan darah saat jantung istirahat (angka bawah dari tensi).",
            )
            skin_thickness = st.number_input(
                "Ketebalan Lipatan Kulit Trisep (mm)",
                min_value=5,
                max_value=110,
                value=int(active_values["SkinThickness"]),
                help="Pengukuran lipatan lemak kulit lengan belakang trisep.",
            )
            insulin = st.number_input(
                "Kadar Insulin Serum 2 Jam (μU/mL)",
                min_value=10,
                max_value=900,
                value=int(active_values["Insulin"]),
                help="Kadar hormon insulin dalam darah 2 jam pasca-makan / beban glukosa.",
            )

        submit_btn = st.form_submit_button(
            "🔬 Lakukan Skrining & Prediksi Risiko",
            use_container_width=True,
            type="primary"
        )

    # PROSES INFERENSI KETIKA TOMBOL DITEKAN
    if submit_btn:
        patient_raw = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        # Transformasi Z-Score yang konsisten dengan data training
        df_input = pd.DataFrame([patient_raw], columns=feature_names)
        df_scaled = (df_input - pd.Series(feature_mean)) / pd.Series(feature_std)

        # Prediksi kelas dan probabilitas
        pred_class = int(model.predict(df_scaled.values)[0])
        probabilities = model.predict_proba(df_scaled.values)[0]
        prob_healthy = float(probabilities[0])
        prob_diabetes = float(probabilities[1])
        confidence = max(prob_healthy, prob_diabetes)
        LOW_CONFIDENCE_THRESHOLD = 0.51  # Ambang batas keyakinan rendah (< 51%)

        st.markdown("---")
        st.markdown("### 📋 Hasil Analisis Risiko Pasien")

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                tendency = "Risiko Tinggi (Diabetes)" if pred_class == 1 else "Risiko Rendah (Tidak Diabetes)"
                st.markdown(
                    f"""
                    <div class="result-box-warning">
                        <h3 style="margin-top:0; color:#b45309;">🟡 HASIL DENGAN KEYAKINAN RENDAH (BORDERLINE)</h3>
                        <p style="font-size: 1.02rem; margin-bottom: 8px;">
                            Berdasarkan analisis model SVM, parameter pasien berada sangat dekat dengan batas keputusan (<i>decision boundary</i>).
                        </p>
                        <div style="background-color: #fef9c3; border-left: 4px solid #ca8a04; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-weight: 600; color: #854d0e; font-size: 0.95rem;">
                            ⚠️ <b>PERINGATAN:</b> Tingkat keyakinan model rendah ({confidence:.1%}). <b>Bisa jadi jawaban hasil skrining ini salah ataupun sangat salah.</b>
                        </div>
                        <div style="background-color: #fffbeb; border: 1px dashed #f59e0b; padding: 10px 14px; margin: 10px 0; border-radius: 6px; font-size: 0.9rem; color: #78350f;">
                            👉 <b>Sangat Disarankan:</b> Perhatikan dan baca panel <b>"Deteksi Faktor Risiko Klinis"</b> di sebelah kanan. Nilai pemeriksaan pasien bisa jadi terlalu ekstrem dari data latih sehingga model berkeyakinan rendah. Panel faktor klinis memberikan indikator konsisten berdasarkan ambang batas medis objektif, bukan bergantung pada hasil pelatihan AI.
                        </div>
                        <p style="font-size: 0.92rem; margin-top: 8px; margin-bottom: 6px;">
                            <b>Kecenderungan Lemah Model:</b> Lebih condong ke arah <i>{tendency}</i>.
                        </p>
                        <hr style="border-color: #fde68a;">
                        <b>Tingkat Keyakinan Model:</b> <span style="font-size: 1.3rem; font-weight: bold;">{confidence:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif pred_class == 1:
                st.markdown(
                    f"""
                    <div class="result-box-diabetes">
                        <h3 style="margin-top:0; color:#991b1b;">⚠️ TERINDIKASI RISIKO TINGGI (DIABETES)</h3>
                        <p style="font-size: 1.05rem; margin-bottom: 8px;">
                            Berdasarkan analisis model SVM, pasien memiliki indikasi kuat menderita atau berisiko tinggi terkena diabetes melitus.
                        </p>
                        <hr style="border-color: #fca5a5;">
                        <b>Tingkat Keyakinan Risiko Diabetes:</b> <span style="font-size: 1.3rem; font-weight: bold;">{prob_diabetes:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-healthy">
                        <h3 style="margin-top:0; color:#166534;">✅ TERINDIKASI RISIKO RENDAH (TIDAK DIABETES)</h3>
                        <p style="font-size: 1.05rem; margin-bottom: 8px;">
                            Berdasarkan analisis model SVM, profil klinis pasien saat ini berada dalam ambang batas normal atau risiko rendah terhadap diabetes.
                        </p>
                        <hr style="border-color: #86efac;">
                        <b>Tingkat Keyakinan Tidak Diabetes:</b> <span style="font-size: 1.3rem; font-weight: bold;">{prob_healthy:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("##### Distribusi Probabilitas Prediksi:")
            st.progress(prob_diabetes, text=f"Kemungkinan Diabetes: {prob_diabetes:.1%}")

        with res_col2:
            st.markdown("##### 🔍 Deteksi Faktor Risiko Klinis:")
            warnings = []

            if glucose >= 140:
                warnings.append(f"🔴 **Glukosa Tinggi ({glucose} mg/dL)**: Ambang normal post-prandial adalah < 140 mg/dL.")
            elif glucose >= 120:
                warnings.append(f"🟠 **Glukosa Borderline ({glucose} mg/dL)**: Perlu pengawasan kadar gula darah.")

            if bmi >= 30:
                warnings.append(f"🔴 **Obesitas (BMI {bmi:.1f})**: Faktor risiko utama resistensi insulin.")
            elif bmi >= 25:
                warnings.append(f"🟠 **Kelebihan Berat Badan (BMI {bmi:.1f})**: Indeks massa tubuh di atas normal.")

            if blood_pressure >= 85:
                warnings.append(f"🔴 **Tekanan Darah Tinggi ({blood_pressure} mmHg)**: Waspada komorbiditas kardiovaskular.")

            if age >= 45:
                warnings.append(f"🟠 **Usia Rentan ({age} tahun)**: Risiko diabetes tipe-2 meningkat pada usia ≥ 45 tahun.")

            if dpf >= 0.6:
                warnings.append(f"🟠 **Faktor Riwayat Genetik Kuat (DPF {dpf:.2f})**: Kecenderungan diabetes dalam silsilah keluarga.")

            if warnings:
                for w in warnings:
                    st.markdown(w)
            else:
                st.success("✨ Seluruh indikator utama pasien berada dalam rentang aman/fisiologis normal.")

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    "🩺 **Pemeriksaan Lanjutan Diperlukan:** Karena tingkat keyakinan model tergolong rendah (< 60%), pasien sangat dianjurkan untuk menjalani pemeriksaan laboratorium lanjutan (seperti tes HbA1c atau tes toleransi glukosa oral ulang) untuk memastikan status diabetes secara definitif."
                )

            st.info(
                "💡 **Rekomendasi Medis:** Hasil skrining ini adalah alat bantu deteksi awal berbasis kecerdasan buatan dan bukan merupakan diagnosis dokter mutlak. Tetap konfirmasikan ke dokter spesialis atau fasilitas kesehatan terdekat."
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: EVALUASI MODEL SVM & PERBANDINGAN SKENARIO
# ──────────────────────────────────────────────────────────────────────────────
with tab_evaluasi:
    st.markdown("#### Mengapa Model SVM-RUS Dipilih?")
    st.markdown(
        """
        Dataset diabetes memiliki ketidakseimbangan kelas (*imbalanced data*), di mana jumlah pasien non-diabetes jauh lebih banyak dibanding pasien diabetes.
        Jika hanya mengandalkan akurasi murni, model baseline gagal mendeteksi lebih dari separuh pasien penderita diabetes (**Recall hanya 45.10%**).
        
        Melalui teknik **Random Under-Sampling (RUS)**, model SVM mampu mengenali pola pasien diabetes secara jauh lebih sensitif, menghasilkan **Recall 76.47%** dan **F1-Score 68.42%**.
        """
    )

    st.markdown("##### Tabel Perbandingan Kinerja SVM Antarskenario Latih:")
    tabel_metrik = pd.DataFrame([
        {"Skenario": "Baseline (Data Asli)", "Akurasi": "73.83%", "Presisi": "67.65%", "Recall (Sensitivitas)": "45.10%", "F1-Score": "54.12%", "Evaluasi": "Recall sangat rendah (>50% pasien luput)"},
        {"Skenario": "Oversampling (SMOTE)", "Akurasi": "75.17%", "Presisi": "62.50%", "Recall (Sensitivitas)": "68.63%", "F1-Score": "65.42%", "Evaluasi": "Kinerja seimbang"},
        {"Skenario": "Undersampling (RUS) [AKTIF]", "Akurasi": "75.84%", "Presisi": "61.90%", "Recall (Sensitivitas)": "76.47%", "F1-Score": "68.42%", "Evaluasi": "Terbaik: Akurasi & F1-Score tertinggi, Recall optimal"},
    ])
    st.dataframe(tabel_metrik, use_container_width=True, hide_index=True)

    st.markdown("##### Matriks Evaluasi Klinis pada Data Uji Independen (N=149 Pasien):")
    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    with ecol1:
        st.metric("Benar Sehat (TN)", "74 Pasien", help="Aktual sehat dan berhasil diprediksi sehat.")
    with ecol2:
        st.metric("Alarm Palsu (FP)", "24 Pasien", help="Aktual sehat namun diprediksi waspada diabetes.")
    with ecol3:
        st.metric("Luput / Pasien Tak Terdeteksi (FN)", "12 Pasien", help="Pasien diabetes yang tidak terdeteksi (sangat diminimalkan).")
    with ecol4:
        st.metric("Benar Diabetes (TP)", "39 Pasien", help="Aktual diabetes dan tepat terdeteksi.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: PANDUAN PARAMETER KLINIS
# ──────────────────────────────────────────────────────────────────────────────
with tab_panduan:
    st.markdown("#### Nilai Rujukan Parameter Laboratorium & Medis")
    st.write("Berdasarkan pedoman klinis umum dan standar American Diabetes Association (ADA):")

    df_panduan = pd.DataFrame([
        {
            "Parameter": "Glucose (Glukosa 2 Jam OGTT)",
            "Rentang Normal": "< 140 mg/dL",
            "Pre-Diabetes / Waspada": "140 - 199 mg/dL",
            "Indikasi Diabetes": "≥ 200 mg/dL",
        },
        {
            "Parameter": "BMI (Indeks Massa Tubuh)",
            "Rentang Normal": "18.5 - 24.9 kg/m²",
            "Pre-Diabetes / Waspada": "25.0 - 29.9 (Overweight)",
            "Indikasi Diabetes": "≥ 30.0 (Obesitas)",
        },
        {
            "Parameter": "Blood Pressure (Diastolik)",
            "Rentang Normal": "< 80 mmHg",
            "Pre-Diabetes / Waspada": "80 - 89 mmHg (Pre-hipertensi)",
            "Indikasi Diabetes": "≥ 90 mmHg (Hipertensi Tahap 1/2)",
        },
        {
            "Parameter": "Insulin (Serum 2 Jam)",
            "Rentang Normal": "16 - 166 μU/mL",
            "Pre-Diabetes / Waspada": "> 166 μU/mL (Hiperinsulinemia)",
            "Indikasi Diabetes": "Tinggi (Resistensi) atau Sangat Rendah (Defisiensi)",
        },
        {
            "Parameter": "Diabetes Pedigree Function",
            "Rentang Normal": "< 0.40",
            "Pre-Diabetes / Waspada": "0.40 - 0.70",
            "Indikasi Diabetes": "> 0.70 (Risiko herediter tinggi)",
        },
    ])
    st.table(df_panduan)
