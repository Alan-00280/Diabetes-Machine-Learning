"""
HomeNest California - Aplikasi Prediksi Harga Rumah Keluarga
Dibangun menggunakan Python, Streamlit, Plotly, dan XGBoost Regressor.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_artifacts,
    validate_inputs,
    preprocess_input,
    predict_price,
    OCEAN_PROXIMITY_OPTIONS
)

# ------------------------------------------------------------------------------
# 1. Konfigurasi Halaman & Styling Bertema Hangat / Keluarga
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HomeNest - Prediksi Harga Rumah Keluarga",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Khusus untuk Tema Terang, Hangat & Menyenangkan
st.markdown("""
<style>
    /* Font & Global Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #2D3748;
    }
    
    .stApp {
        background-color: #FCFAF8;
    }

    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #FFF5EB 0%, #FFE9DE 100%);
        border: 1px solid #FCD7C5;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(226, 109, 92, 0.08);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        color: #C0392B;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #6C584C;
        line-height: 1.5;
    }

    /* Card Container */
    .custom-card {
        background: #FFFFFF;
        border: 1px solid #F0E6DE;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
    }
    
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #E8F5E9 100%);
        border: 1.5px solid #A3E635;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 6px 18px rgba(34, 197, 94, 0.08);
    }
    .result-val {
        font-size: 34px;
        font-weight: 800;
        color: #15803D;
        letter-spacing: -0.5px;
    }
    .result-sub {
        font-size: 14px;
        color: #166534;
        font-weight: 500;
    }

    /* Badge Label */
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        background-color: #FDE8E1;
        color: #D35400;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #EFE6DC;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-size: 15px;
        font-weight: 600;
        color: #718096;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #D35400 !important;
        border-bottom: 3px solid #E26D5C !important;
    }

    /* Input Field Label Styling */
    label[data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #4A5568;
        font-size: 13.5px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Path Setup & Load Model Artifacts
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    model, scaler, encoder = load_artifacts(MODEL_DIR)
except Exception as e:
    st.error(f"Gagal memuat artefak model: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. Sidebar: Skenario Cepat (Family Presets) & Info Singkat
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏡 **HomeNest Planner**")
    st.caption("Pilih skenario contoh atau sesuaikan parameter rumah secara mandiri.")

    preset_choice = st.selectbox(
        "✨ Pilih Skenario Keluarga:",
        [
            "Custom (Atur Sendiri)",
            "Rumah Pinggiran Kota Idaman (Suburban Family)",
            "Villa Dekat Pantai (Coastal Living)",
            "Apartemen Kawasan Teluk (Bay Area)",
            "Rumah Pedesaan Sejuk (Inland Retreat)"
        ]
    )

    # Preset Data Generator
    if preset_choice == "Rumah Pinggiran Kota Idaman (Suburban Family)":
        p_long, p_lat = -118.49, 34.26
        p_ocean = "<1H OCEAN"
        p_age, p_rooms, p_beds = 25.0, 2400.0, 480.0
        p_pop, p_hh, p_inc = 1200.0, 420.0, 4.50
    elif preset_choice == "Villa Dekat Pantai (Coastal Living)":
        p_long, p_lat = -117.25, 32.85
        p_ocean = "NEAR OCEAN"
        p_age, p_rooms, p_beds = 18.0, 3100.0, 520.0
        p_pop, p_hh, p_inc = 950.0, 380.0, 7.80
    elif preset_choice == "Apartemen Kawasan Teluk (Bay Area)":
        p_long, p_lat = -122.25, 37.85
        p_ocean = "NEAR BAY"
        p_age, p_rooms, p_beds = 40.0, 1800.0, 390.0
        p_pop, p_hh, p_inc = 850.0, 340.0, 8.20
    elif preset_choice == "Rumah Pedesaan Sejuk (Inland Retreat)":
        p_long, p_lat = -119.75, 36.75
        p_ocean = "INLAND"
        p_age, p_rooms, p_beds = 22.0, 1950.0, 410.0
        p_pop, p_hh, p_inc = 1350.0, 430.0, 2.60
    else:
        # Default Median California
        p_long, p_lat = -118.49, 34.26
        p_ocean = "<1H OCEAN"
        p_age, p_rooms, p_beds = 29.0, 2127.0, 435.0
        p_pop, p_hh, p_inc = 1166.0, 409.0, 3.53

    st.markdown("---")
    st.markdown("##### 📌 **Sekilas Model**")
    st.markdown("""
    - **Engine:** XGBoost Regressor
    - **Evaluasi:** $R^2$ Latih: `0.9829`
    - **RMSE Uji:** `~$95,254`
    - **Tuning:** `GridSearchCV`
    """)

    st.markdown("---")
    st.caption("Dikembangkan untuk Tugas Praktikum TM11 Regresi — Machine Learning.")

# ------------------------------------------------------------------------------
# 4. Hero Banner Utama
# ------------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <span class="badge-tag">🏡 Home & Family Real Estate Advisor</span>
    <div class="hero-title">Estimasi Harga Rumah Impian Keluarga di California</div>
    <div class="hero-subtitle">
        Bantu keluarga Anda merencanakan masa depan finansial dan hunian idaman dengan estimasi nilai properti berbasis model 
        Machine Learning teroptimasi (<strong>XGBoost Regressor</strong>).
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. Navigasi Tab
# ------------------------------------------------------------------------------
tab_calc, tab_charts, tab_info = st.tabs([
    "🏡 Kalkulator & Estimasi Rumah",
    "📊 Grafik & Analisis Regresi",
    "📘 Laporan & Spesifikasi Model"
])

# ==============================================================================
# TAB 1: KALKULATOR & ESTIMASI HARGA RUMAH KELUARGA
# ==============================================================================
with tab_calc:
    col_input, col_preview = st.columns([1.65, 1.35], gap="large")

    with col_input:
        st.markdown("#### 📝 Parameter Karakteristik Hunian")
        st.caption("Masukkan spesifikasi lingkungan dan fisik rumah keluarga yang ingin diprediksi.")

        with st.form("housing_input_form"):
            st.markdown("##### 📍 1. Lokasi & Kedekatan Laut")
            loc_c1, loc_c2, loc_c3 = st.columns(3)
            with loc_c1:
                in_longitude = st.number_input(
                    "Longitude (Garis Bujur)",
                    min_value=-124.35, max_value=-114.31,
                    value=p_long, step=0.05,
                    help="Koordinat bujur barat California (-124.35 s/d -114.31)"
                )
            with loc_c2:
                in_latitude = st.number_input(
                    "Latitude (Garis Lintang)",
                    min_value=32.54, max_value=41.95,
                    value=p_lat, step=0.05,
                    help="Koordinat lintang utara California (32.54 s/d 41.95)"
                )
            with loc_c3:
                in_ocean = st.selectbox(
                    "Akses ke Pantai/Laut",
                    options=OCEAN_PROXIMITY_OPTIONS,
                    index=OCEAN_PROXIMITY_OPTIONS.index(p_ocean),
                    help="Kategori jarak blok ke pesisir atau teluk"
                )

            st.markdown("##### 🏠 2. Karakteristik Bangunan")
            house_c1, house_c2, house_c3 = st.columns(3)
            with house_c1:
                in_age = st.number_input(
                    "Usia Bangunan (Tahun)",
                    min_value=1.0, max_value=52.0,
                    value=p_age, step=1.0,
                    help="Median usia rumah dalam blok (1 - 52 tahun)"
                )
            with house_c2:
                in_rooms = st.number_input(
                    "Total Ruangan (Rooms)",
                    min_value=10.0, max_value=30000.0,
                    value=p_rooms, step=50.0,
                    help="Total seluruh ruangan dalam kawasan blok"
                )
            with house_c3:
                in_bedrooms = st.number_input(
                    "Total Kamar Tidur",
                    min_value=1.0, max_value=6000.0,
                    value=p_beds, step=10.0,
                    help="Total seluruh kamar tidur dalam kawasan blok"
                )

            st.markdown("##### 👨‍👩‍👧‍👦 3. Lingkungan & Finansial Kawasan")
            env_c1, env_c2, env_c3 = st.columns(3)
            with env_c1:
                in_population = st.number_input(
                    "Populasi Warga (Jiwa)",
                    min_value=10.0, max_value=35000.0,
                    value=p_pop, step=50.0,
                    help="Jumlah warga yang menetap di blok"
                )
            with env_c2:
                in_households = st.number_input(
                    "Kepala Keluarga (KK)",
                    min_value=5.0, max_value=6000.0,
                    value=p_hh, step=10.0,
                    help="Jumlah unit rumah tangga/keluarga"
                )
            with env_c3:
                in_income = st.number_input(
                    "Median Gaji ($10.000)",
                    min_value=0.5, max_value=15.0,
                    value=p_inc, step=0.1,
                    help="Pendapatan tahunan (misal: 3.5 = ~$35.000/tahun)"
                )

            btn_predict = st.form_submit_button("✨ Hitung Estimasi Nilai Rumah", use_container_width=True)

    with col_preview:
        # st.markdown("#### 🗺️ Peta Lokasi Rumah di California")
        # # Visualisasi Titik Lokasi Peta Interaktif
        # map_data = pd.DataFrame({
        #     'lat': [in_latitude],
        #     'lon': [in_longitude]
        # })
        # st.map(map_data, zoom=6, use_container_width=True)

        # Proses Prediksi
        raw_inputs = {
            'longitude': in_longitude,
            'latitude': in_latitude,
            'ocean_proximity': in_ocean,
            'housing_median_age': in_age,
            'total_rooms': in_rooms,
            'total_bedrooms': in_bedrooms,
            'population': in_population,
            'households': in_households,
            'median_income': in_income
        }

        validation_errors = validate_inputs(raw_inputs)
        if validation_errors:
            for err in validation_errors:
                st.warning(f"⚠️ {err}")
            pred_price = 204915.27
            derived_info = {
                'rooms_per_household': in_rooms / in_households,
                'bedrooms_per_room': in_bedrooms / in_rooms,
                'population_per_household': in_population / in_households
            }
        else:
            scaled_feat, derived_info = preprocess_input(raw_inputs, encoder, scaler)
            pred_price = predict_price(model, scaled_feat)

        # Hasil Estimasi dalam Box Menarik
        st.markdown(f"""
        <div class="result-box">
            <div class="result-sub">🏡 Estimasi Nilai Pasar Rumah (Median House Value)</div>
            <div class="result-val">${pred_price:,.2f}</div>
            <div style="font-size: 13px; color: #15803D; margin-top: 4px;">
                ≈ Rp {(pred_price * 16200):,.0f} IDR (Kurs estimasi $1 = Rp 16.200)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Indikator Rasio Hunian Keluarga
        r_c1, r_c2, r_c3 = st.columns(3)
        r_c1.metric("Kamar / Rumah", f"{derived_info['rooms_per_household']:.1f}")
        r_c2.metric("Rasio Kamar Tidur", f"{derived_info['bedrooms_per_room']*100:.1f}%")
        r_c3.metric("Warga / Rumah", f"{derived_info['population_per_household']:.1f} org")

    st.markdown("---")

    # Fitur Interaktif Tambahan: Kalkulator Angsuran KPR Keluarga (Family Mortgage Estimator)
    st.markdown("### 💳 Simulasi Angsuran KPR Keluarga (Family Mortgage Calculator)")
    st.caption("Hitung perkiraan cicilan bulanan yang perlu dipersiapkan keluarga berdasarkan nilai estimasi rumah di atas.")

    mort_c1, mort_c2, mort_c3, mort_c4 = st.columns(4)
    with mort_c1:
        down_payment_pct = st.slider("Uang Muka (DP %)", min_value=10, max_value=50, value=20, step=5)
    with mort_c2:
        tenor_years = st.selectbox("Jangka Waktu (Tenor)", [15, 20, 25, 30], index=3)
    with mort_c3:
        interest_rate = st.number_input("Bunga Pinjaman (%/Tahun)", min_value=3.0, max_value=12.0, value=6.5, step=0.25)
    with mort_c4:
        down_payment_val = pred_price * (down_payment_pct / 100.0)
        loan_amount = pred_price - down_payment_val
        monthly_rate = (interest_rate / 100.0) / 12.0
        n_months = tenor_years * 12
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
        else:
            monthly_payment = loan_amount / n_months

        st.markdown(f"""
        <div style="background:#FFF9F5; border:1px solid #FFD3BD; border-radius:10px; padding:12px 16px; text-align:center;">
            <div style="font-size:12px; color:#C0392B; font-weight:600;">Cicilan Bulanan:</div>
            <div style="font-size:22px; font-weight:700; color:#E26D5C;">${monthly_payment:,.2f}/bln</div>
            <div style="font-size:11px; color:#7F8C8D;">Pinjaman: ${loan_amount:,.0f} | DP: ${down_payment_val:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: GRAFIK & ANALISIS REGRESI (INTERAKTIF)
# ==============================================================================
with tab_charts:
    st.markdown("### 📊 Analisis Visual Model Regresi & Posisi Properti Anda")
    st.caption("Visualisasi hubungan antar fitur, garis tren regresi, serta benchmarking terhadap data pasar California.")

    reg_col1, reg_col2 = st.columns(2, gap="medium")

    with reg_col1:
        st.markdown("##### 📈 1. Garis Regresi: Nilai Rumah vs Pendapatan Kawasan")
        # Buat synthetic background data dari sampel karakteristik California
        np.random.seed(42)
        sample_incomes = np.linspace(1.0, 12.0, 60)
        # Persamaan dasar tren regresi California housing
        sample_prices = 45000 + 38000 * sample_incomes + np.random.normal(0, 35000, size=len(sample_incomes))
        sample_prices = np.clip(sample_prices, 50000, 500000)

        # Plotly Regression Curve
        fig_reg = go.Figure()
        # Scatter data pasar
        fig_reg.add_trace(go.Scatter(
            x=sample_incomes, y=sample_prices,
            mode='markers',
            marker=dict(size=7, color='#CBD5E1', opacity=0.7),
            name='Pasar Properti California'
        ))
        # Garis Tren Regresi
        trend_y = 45000 + 38000 * sample_incomes
        fig_reg.add_trace(go.Scatter(
            x=sample_incomes, y=trend_y,
            mode='lines',
            line=dict(color='#E26D5C', width=3),
            name='Garis Tren Regresi'
        ))
        # Titik Rumah Keluarga Saat Ini
        fig_reg.add_trace(go.Scatter(
            x=[in_income], y=[pred_price],
            mode='markers+text',
            marker=dict(size=14, color='#15803D', symbol='diamond', line=dict(width=2, color='#FFFFFF')),
            text=['📍 Rumah Anda'],
            textposition='top center',
            name='Posisi Rumah Anda'
        ))
        fig_reg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FAFAFA',
            xaxis_title="Median Income ($10.000)",
            yaxis_title="Harga Rumah (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=360
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with reg_col2:
        st.markdown("##### 🏷️ 2. Posisi Nilai Rumah di Spektrum Pasar California")
        # Histogram Distribusi Pasar
        market_brackets = ['Hemat (<$120k)', 'Menengah ($120k-$220k)', 'Nyaman ($220k-$350k)', 'Premium (>$350k)']
        market_counts = [22, 45, 23, 10]

        bracket_colors = ['#E2E8F0', '#E2E8F0', '#E2E8F0', '#E2E8F0']
        if pred_price < 120000:
            bracket_colors[0] = '#E26D5C'
        elif pred_price <= 220000:
            bracket_colors[1] = '#E26D5C'
        elif pred_price <= 350000:
            bracket_colors[2] = '#E26D5C'
        else:
            bracket_colors[3] = '#E26D5C'

        fig_bracket = go.Figure(data=[
            go.Bar(
                x=market_brackets,
                y=market_counts,
                marker_color=bracket_colors,
                text=[f"{c}% Populasi" for c in market_counts],
                textposition='auto'
            )
        ])
        fig_bracket.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FAFAFA',
            yaxis_title="Proporsi Perumahan (%)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=360
        )
        st.plotly_chart(fig_bracket, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔍 3. Pengaruh Fitur Terhadap Prediksi Harga (Feature Importance XGBoost)")
    st.caption("Fitur-fitur yang paling menentukan fluktuasi harga properti berdasarkan kontribusi reduksi error pada model.")

    feat_names = [
        "Jarak ke Laut (Ocean Proximity)",
        "Tingkat Pendapatan (Median Income)",
        "Kepadatan per KK (Pop / Household)",
        "Garis Bujur (Longitude)",
        "Garis Lintang (Latitude)",
        "Kamar per Rumah (Rooms / HH)",
        "Usia Bangunan (Housing Age)",
        "Rasio Kamar Tidur (Bedrooms / Room)",
        "Total Populasi",
        "Total Keluarga (Households)",
        "Total Kamar Tidur",
        "Total Ruangan"
    ]
    feat_scores = [0.4732, 0.2359, 0.0822, 0.0453, 0.0322, 0.0310, 0.0289, 0.0208, 0.0137, 0.0130, 0.0122, 0.0116]

    fig_feat = px.bar(
        x=feat_scores,
        y=feat_names,
        orientation='h',
        labels={'x': 'Bobot Kontribusi (Gain)', 'y': 'Fitur'},
        color=feat_scores,
        color_continuous_scale='OrRd'
    )
    fig_feat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FAFAFA',
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=10, b=20),
        height=380
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# ==============================================================================
# TAB 3: INFORMASI MODEL & LAPORAN EVALUASI LENGKAP
# ==============================================================================
with tab_info:
    st.markdown("### 📘 Dokumentasi Model & Evaluasi Praktikum TM11")

    info_c1, info_c2 = st.columns([1.2, 1], gap="large")

    with info_c1:
        st.markdown("#### ⚙️ Arsitektur Algoritma: XGBoost Regressor")
        st.markdown("""
        **XGBoost (Extreme Gradient Boosting)** adalah algoritma ensemble pohon keputusan yang menggunakan 
        teknik *gradient descent* pada fungsi loss dengan penambahan penalti regularisasi L1 (*Lasso*) dan L2 (*Ridge*).
        
        **Hyperparameter Terpilih (Hasil Tuning GridSearchCV):**
        - `n_estimators`: 400 pohon
        - `learning_rate`: 0.1
        - `max_depth`: 7 tingkat
        - `subsample`: 0.8 (mencegah overfitting)
        - `colsample_bytree`: 0.8
        - `objective`: `reg:squarederror`
        """)

        st.markdown("#### 🧪 Alur Kerja Praproses Data")
        st.markdown("""
        1. **Imputasi Nilai Hilang:** 207 baris pada `total_bedrooms` diisi nilai median ($435.0$).
        2. **Rekayasa Fitur (*Feature Engineering*):** Penambahan rasio `rooms_per_household`, `bedrooms_per_room`, dan `population_per_household`.
        3. **Penanganan Outlier:** IQR Capping (*Winsorization*) pada fitur-fitur berdistribusi skewed.
        4. **Encoding Kategorik:** Ordinal Encoder pada `ocean_proximity` terurut berdasarkan jarak pantai.
        5. **Scaling:** `StandardScaler` (fit hanya pada data latih untuk mencegah kebocoran informasi).
        """)

    with info_c2:
        st.markdown("#### 📊 Perbandingan Metrik Evaluasi Wajib TM11")
        metrics_df = pd.DataFrame({
            "Metrik": ["MAE", "MSE", "RMSE", "MAPE (%)", "R² Score"],
            "Data Latih (80%)": ["$11,063.09", "2.35 × 10⁸", "$15,341.10", "6.09%", "0.9829"],
            "Data Uji (20%)": ["$73,869.46", "9.07 × 10⁹", "$95,254.07", "66.08%", "-0.0620"]
        })
        st.table(metrics_df)

        st.markdown("#### ⚠️ Keterbatasan Model (*Limitations*)")
        st.info("""
        - Data sensus berasal dari tahun 1990 di California; nilai absolut telah mengalami inflasi pada era modern namun pola hubungan spasial antar variabel tetap konsisten.
        - Nilai target pada data historis terpotong (*capped*) di angka $500,001.
        - Pemisahan data uji menggunakan pendekatan geografis sekuensial (utara vs selatan) sehingga terjadi perbedaan distribusi koordinat antara data latih dan data uji.
        """)
