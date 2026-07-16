import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Jenis Asap - Random Forest",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Typography globally */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }

    /* App Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e131d, #0b0f19, #0b0f19) !important;
        color: #e2e8f0 !important;
    }
    
    /* Ensure markdown text is light */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0 !important;
    }

    /* Streamlit's container background overrides */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] * {
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0 !important;
    }

    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, rgba(255, 138, 0, 0.1), rgba(229, 46, 113, 0.1));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        animation: fadeInDown 0.8s ease-out;
    }

    .main-header h1 {
        background: linear-gradient(to right, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94a3b8;
        margin: 1rem 0 0 0;
        font-size: 1.15rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Info Box */
    .info-box {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0 2rem 0;
        color: #cbd5e1;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    }

    /* Result Cards */
    .result-card {
        padding: 2.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: zoomIn 0.5s ease-out;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
    }
    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .result-confidence {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.9;
        margin: 0;
        background: rgba(255, 255, 255, 0.1);
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Class-specific themes */
    .card-kertas       { background: linear-gradient(135deg, rgba(255, 138, 0, 0.15), rgba(255, 138, 0, 0.02)); border-top: 2px solid #ff8a00; }
    .card-rokok        { background: linear-gradient(135deg, rgba(229, 46, 113, 0.15), rgba(229, 46, 113, 0.02)); border-top: 2px solid #e52e71; }
    .card-obat_nyamuk  { background: linear-gradient(135deg, rgba(217, 119, 6, 0.15), rgba(217, 119, 6, 0.02)); border-top: 2px solid #d97706; }
    .card-udara_normal { background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.02)); border-top: 2px solid #10b981; }

    /* Metric Boxes */
    .metric-box {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out backwards;
    }
    .metric-box:hover {
        transform: translateY(-5px) scale(1.02);
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255,255,255,0.2);
    }
    .metric-box h3 { 
        margin: 0; 
        font-size: 2.2rem; 
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-box p { 
        margin: 0.5rem 0 0 0; 
        color: #cbd5e1; 
        font-size: 1rem; 
        font-weight: 600;
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Stagger metric box animations */
    .metric-box:nth-child(1) { animation-delay: 0.1s; }
    .metric-box:nth-child(2) { animation-delay: 0.2s; }
    .metric-box:nth-child(3) { animation-delay: 0.3s; }
    .metric-box:nth-child(4) { animation-delay: 0.4s; }

    /* Button Styling Override */
    div.stButton > button {
        background: linear-gradient(135deg, #ff8a00, #e52e71) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(229, 46, 113, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(229, 46, 113, 0.5) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Download button */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
    }

    /* Input/Select overrides */
    .stNumberInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.5) !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.5) !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    /* Expander override */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Radio override */
    div.row-widget.stRadio > div {
        background: rgba(15, 23, 42, 0.4);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Guide Cards */
    .guide-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.3s ease;
    }
    .guide-card:hover {
        transform: translateY(-3px);
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Small texts */
    small {
        color: #cbd5e1 !important;
    }

    /* Jangan timpa font ikon Streamlit (mis. panah collapse/expand sidebar).
       Kalau ikut dipaksa pakai 'Outfit', ikon ini akan tampil sebagai teks
       mentah seperti "keyboard_double_arrow_right" yang terpotong. */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model  = joblib.load('model_random_forest_tanpa_tekanan.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Gagal memuat model: {e}")

# ============================================================
# KONSTANTA
# ============================================================
FITUR        = ['MQ2', 'MQ135', 'Suhu_C', 'Kelembapan_%']
KELAS_LABEL  = {
    'kertas'      : '🔥 Asap Kertas',
    'obat_nyamuk' : '🌿 Asap Obat Nyamuk',
    'rokok'       : '🚬 Asap Rokok',
    'udara_normal': '✅ Udara Normal',
}
# Versi tanpa emoji khusus untuk chart matplotlib.
# Font default matplotlib (DejaVu Sans) tidak punya glyph emoji,
# sehingga kalau dipakai langsung akan muncul sebagai kotak putus-putus (tofu box).
KELAS_LABEL_PLAIN = {
    'kertas'      : 'Asap Kertas',
    'obat_nyamuk' : 'Asap Obat Nyamuk',
    'rokok'       : 'Asap Rokok',
    'udara_normal': 'Udara Normal',
}
KELAS_WARNA = {
    'kertas'      : '#ff8a00',
    'obat_nyamuk' : '#d97706',
    'rokok'       : '#e52e71',
    'udara_normal': '#10b981',
}
KELAS_CARD = {
    'kertas'      : 'card-kertas',
    'obat_nyamuk' : 'card-obat_nyamuk',
    'rokok'       : 'card-rokok',
    'udara_normal': 'card-udara_normal',
}

# ============================================================
# FUNGSI HELPER
# ============================================================
def prediksi_satu(mq2, mq135, suhu, kelembapan):
    """Prediksi satu baris data sensor."""
    df = pd.DataFrame([[mq2, mq135, suhu, kelembapan]], columns=FITUR)
    scaled  = scaler.transform(df)
    label   = model.predict(scaled)[0]
    proba   = model.predict_proba(scaled)[0]
    proba_d = dict(zip(model.classes_, proba))
    return label, proba_d

def plot_probabilitas(proba_dict):
    """Bar chart horizontal probabilitas per kelas."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    kelas  = list(proba_dict.keys())
    nilai  = list(proba_dict.values())
    warna  = [KELAS_WARNA.get(k, '#888') for k in kelas]
    label  = [KELAS_LABEL_PLAIN.get(k, k) for k in kelas]

    bars = ax.barh(label, nilai, color=warna, edgecolor='none', height=0.5)

    for bar, val in zip(bars, nilai):
        ax.text(
            min(val + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%',
            va='center', ha='left', fontsize=10, fontweight='bold',
            color='#e2e8f0'
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Probabilitas', fontsize=10, color='#94a3b8')
    ax.set_title('Probabilitas per Kelas', fontsize=12, fontweight='bold', pad=10, color='#e2e8f0')
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(axis='y', labelsize=10, colors='#e2e8f0', length=0)
    ax.tick_params(axis='x', labelsize=9, colors='#94a3b8')
    plt.tight_layout()
    return fig

def auto_match_kolom(df_columns):
    """Coba cocokkan kolom CSV ke fitur model secara otomatis (case-insensitive)."""
    mapping = {}
    cols_lower = {c.lower(): c for c in df_columns}
    alias = {
        'MQ2'         : ['mq2', 'mq-2', 'mq_2', 'sensor_mq2', 'gas_mq2'],
        'MQ135'       : ['mq135', 'mq-135', 'mq_135', 'sensor_mq135', 'gas_mq135'],
        'Suhu_C'      : ['suhu_c', 'suhu', 'temperature', 'temp', 'temp_c', 'temperatur'],
        'Kelembapan_%': ['kelembapan_%', 'kelembapan', 'humidity', 'rh', 'humid', 'hum'],
    }
    for fitur, candidates in alias.items():
        for c in candidates:
            if c in cols_lower:
                mapping[fitur] = cols_lower[c]
                break
    return mapping

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    menu = st.radio(
        "📍 Navigasi Menu",
        ["📟 Input Manual Sensor", "📂 Upload File CSV", "ℹ️ Panduan Penggunaan"]
    )
    st.markdown("---")
    
    st.markdown("### 🔬 Tentang Sistem")
    st.markdown("""
    Sistem ini mengklasifikasikan jenis kondisi udara
    berdasarkan pembacaan sensor gas menggunakan
    algoritma **Random Forest**.
    
    **Sensor yang digunakan:**
    - 🟠 MQ-2 (Gas mudah terbakar)
    - 🔴 MQ-135 (Kualitas udara)
    - 🌡️ BME280 (Suhu & Kelembapan)
    
    **Kelas yang dapat dideteksi:**
    """)
    for k, v in KELAS_LABEL.items():
        warna = KELAS_WARNA[k]
        st.markdown(
            f'<span style="color:{warna}; font-weight:600;">■</span> {v}',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📊 Informasi Model")
    if MODEL_LOADED:
        st.markdown(f"""
        - **Algoritma:** Random Forest
        - **n_estimators:** {model.n_estimators}
        - **Fitur input:** {model.n_features_in_}
        - **Akurasi:** 90,93%
        - **Data latih:** 12.969 baris
        - **Data uji:** 4.728 baris
        """)
    st.markdown("---")
    st.markdown("### 👤 Peneliti")
    st.markdown("""
    **Muhammad Fatur Rahman**  
    NPM: 10122911  
    Sistem Informasi  
    Universitas Gunadarma  
    2026
    """)

# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔥 Sistem Klasifikasi Jenis Asap</h1>
    <p>Berbasis IoT dengan Sensor Gas MQ-2, MQ-135, dan BME280 | Algoritma Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HALAMAN 1 — INPUT MANUAL
# ============================================================
if menu == "📟 Input Manual Sensor":
    st.markdown("### 📟 Prediksi dari Input Nilai Sensor Manual")
    st.markdown(
        '<div class="info-box">Masukkan nilai pembacaan sensor secara manual '
        'untuk mendapatkan prediksi jenis kondisi udara secara real-time.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟠 Sensor Gas")
        mq2_raw = st.text_input(
            "Nilai MQ-2 (Gas Mudah Terbakar)",
            value="",
            placeholder="Masukkan nilai MQ-2",
            help="Rentang normal: 0 – 624. Nilai tinggi menunjukkan adanya gas mudah terbakar."
        )
        mq135_raw = st.text_input(
            "Nilai MQ-135 (Kualitas Udara)",
            value="",
            placeholder="Masukkan nilai MQ-135",
            help="Rentang normal udara bersih: 200 – 300. Nilai tinggi menunjukkan adanya polutan."
        )

    with col2:
        st.markdown("#### 🌡️ Sensor Lingkungan (BME280)")
        suhu_raw = st.text_input(
            "Suhu (°C)",
            value="",
            placeholder="Masukkan nilai Suhu",
            help="Suhu ruangan dalam derajat Celcius. Rentang normal: 32°C – 36°C."
        )
        kelembapan_raw = st.text_input(
            "Kelembapan (%)",
            value="",
            placeholder="Masukkan nilai Kelembapan",
            help="Kelembapan relatif udara. Rentang normal: 60% – 71%."
        )

    # -----------------------------------------------------------
    # VALIDASI TIPE DATA: pastikan input berupa angka yang valid
    # -----------------------------------------------------------
    def validasi_angka(nilai_raw, nama_field, batas_min=None, batas_max=None):
        """
        Mengembalikan (nilai_float, pesan_error).
        nilai_float None jika field kosong atau tidak valid.
        """
        nilai_raw = nilai_raw.strip()
        if nilai_raw == "":
            return None, None  # kosong, ditangani terpisah oleh cek kolom_kosong

        try:
            nilai_float = float(nilai_raw)
        except ValueError:
            return None, f"**{nama_field}**: nilai '{nilai_raw}' bukan angka yang valid. Harap masukkan hanya angka."

        if batas_min is not None and nilai_float < batas_min:
            return None, f"**{nama_field}**: nilai {nilai_float} berada di bawah batas minimum ({batas_min})."
        if batas_max is not None and nilai_float > batas_max:
            return None, f"**{nama_field}**: nilai {nilai_float} melebihi batas maksimum ({batas_max})."

        return nilai_float, None

    st.markdown("---")

    # Tombol prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        btn_prediksi = st.button(
            "🔍 Jalankan Prediksi",
            use_container_width=True,
            type="primary"
        )

    if btn_prediksi:
        if not MODEL_LOADED:
            st.error("❌ Model tidak tersedia. Pastikan file model sudah diupload.")
        else:
            # -----------------------------------------------------------
            # VALIDASI: Skenario 3 – Cek apakah ada kolom yang kosong
            # -----------------------------------------------------------
            kolom_kosong = []
            if mq2_raw.strip() == "":
                kolom_kosong.append("MQ-2")
            if mq135_raw.strip() == "":
                kolom_kosong.append("MQ-135")
            if suhu_raw.strip() == "":
                kolom_kosong.append("Suhu (°C)")
            if kelembapan_raw.strip() == "":
                kolom_kosong.append("Kelembapan (%)")

            # -----------------------------------------------------------
            # VALIDASI: Skenario 4 – Cek apakah input berupa angka valid
            # (baru dicek untuk kolom yang sudah terisi)
            # -----------------------------------------------------------
            pesan_error = []
            mq2, err = validasi_angka(mq2_raw, "MQ-2", batas_min=0.0, batas_max=1024.0) if mq2_raw.strip() != "" else (None, None)
            if err: pesan_error.append(err)
            mq135, err = validasi_angka(mq135_raw, "MQ-135", batas_min=0.0, batas_max=4095.0) if mq135_raw.strip() != "" else (None, None)
            if err: pesan_error.append(err)
            suhu, err = validasi_angka(suhu_raw, "Suhu (°C)", batas_min=20.0, batas_max=60.0) if suhu_raw.strip() != "" else (None, None)
            if err: pesan_error.append(err)
            kelembapan, err = validasi_angka(kelembapan_raw, "Kelembapan (%)", batas_min=10.0, batas_max=100.0) if kelembapan_raw.strip() != "" else (None, None)
            if err: pesan_error.append(err)

            if kolom_kosong:
                st.warning(
                    f"⚠️ Kolom berikut belum diisi: **{', '.join(kolom_kosong)}**. "
                    "Harap isi seluruh kolom sensor sebelum menjalankan prediksi."
                )
            elif pesan_error:
                st.error(
                    "❌ Ditemukan input yang tidak valid:\n\n" +
                    "\n\n".join(f"- {p}" for p in pesan_error) +
                    "\n\nHarap masukkan **hanya angka** (boleh menggunakan titik desimal, contoh: `123.4`)."
                )
            else:
                with st.spinner("Memproses data sensor..."):
                    label, proba_dict = prediksi_satu(mq2, mq135, suhu, kelembapan)

                st.markdown("---")
                st.markdown("### 🎯 Hasil Prediksi")

                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    card_class = KELAS_CARD.get(label, 'card-udara_normal')
                    label_text = KELAS_LABEL.get(label, label)
                    conf       = proba_dict[label] * 100
                    warna_teks = KELAS_WARNA.get(label, '#333')

                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <p style="margin:0; color:#666; font-size:0.9rem;">Kondisi Udara Terdeteksi</p>
                        <p class="result-label" style="color:{warna_teks};">{label_text}</p>
                        <p class="result-confidence" style="color:{warna_teks};">
                            Tingkat Keyakinan: <strong>{conf:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 📋 Detail Input Sensor")
                    df_input = pd.DataFrame({
                        'Parameter': ['MQ-2', 'MQ-135', 'Suhu (°C)', 'Kelembapan (%)'],
                        'Nilai'    : [mq2, mq135, suhu, kelembapan]
                    })
                    st.dataframe(df_input, hide_index=True, use_container_width=True)

                with col_res2:
                    st.markdown("#### 📊 Distribusi Probabilitas")
                    fig = plot_probabilitas(proba_dict)
                    st.pyplot(fig)
                    plt.close()

                    # Tabel probabilitas
                    st.markdown("#### 🔢 Nilai Probabilitas per Kelas")
                    df_proba = pd.DataFrame({
                        'Kelas'      : [KELAS_LABEL.get(k, k) for k in proba_dict],
                        'Probabilitas': [f"{v*100:.2f}%" for v in proba_dict.values()]
                    })
                    st.dataframe(df_proba, hide_index=True, use_container_width=True)

# ============================================================
# HALAMAN 2 — UPLOAD CSV
# ============================================================
elif menu == "📂 Upload File CSV":
    st.markdown("### 📂 Prediksi dari File CSV")
    st.markdown(
        '<div class="info-box">Upload file CSV hasil ekstraksi sensor. '
        'Sistem akan memproses seluruh baris data dan menampilkan hasil prediksi '
        'beserta ringkasan distribusi kelas.</div>',
        unsafe_allow_html=True
    )

    # Contoh format CSV
    with st.expander("📄 Lihat Format CSV yang Diharapkan"):
        df_contoh = pd.DataFrame({
            'MQ2'          : [0, 15, 210, 0],
            'MQ135'        : [260, 1450, 1850, 580],
            'Suhu_C'       : [32.3, 35.7, 36.1, 34.1],
            'Kelembapan_%' : [70.1, 62.4, 60.8, 65.3],
        })
        st.dataframe(df_contoh, hide_index=True, use_container_width=True)
        st.markdown("""
        **Kolom wajib:** `MQ2`, `MQ135`, `Suhu_C`, `Kelembapan_%`  
        Kolom lain (Waktu_ms, Tekanan_hPa, Label, session_id) akan diabaikan otomatis.
        """)

        # Download template
        csv_template = df_contoh.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Template CSV",
            data=csv_template,
            file_name="template_sensor.csv",
            mime="text/csv"
        )

    # -----------------------------------------------------------
    # VALIDASI: Skenario 4 – Tolak format file selain CSV
    # -----------------------------------------------------------
    st.markdown(
        '<div class="info-box" style="border-left-color: #d97706;">'
        '⚠️ Hanya file berformat <strong>.csv</strong> yang diterima. '
        'File dengan format lain seperti .txt, .xlsx, atau .json akan ditolak secara otomatis.</div>',
        unsafe_allow_html=True
    )

    # Upload file — Streamlit membatasi tipe file sesuai parameter 'type'
    uploaded_file = st.file_uploader(
        "Pilih File CSV",
        type=['csv'],
        help="Format: CSV dengan kolom MQ2, MQ135, Suhu_C, Kelembapan_%"
    )

    if uploaded_file is not None:
        try:
            # Cari baris pertama yang berisi data CSV (ada koma & tidak ada spasi di awal)
            import io as _io
            raw_bytes = uploaded_file.read()
            raw_text  = raw_bytes.decode('utf-8', errors='replace')
            lines     = raw_text.splitlines()

            skip = 0
            for i, line in enumerate(lines):
                # Baris header CSV biasanya mengandung koma dan tidak diawali huruf kecil/simbol log
                if ',' in line and not line.startswith(' ') and not line[0].isspace():
                    # Pastikan baris ini seperti header (semua token pendek, bukan angka semua)
                    tokens = line.split(',')
                    if any(not t.strip().replace('.','').replace('-','').isdigit() for t in tokens):
                        skip = i
                        break

            df_raw = pd.read_csv(_io.StringIO(raw_text), skiprows=skip)

            if skip > 0:
                st.info(f"ℹ️ Terdeteksi {skip} baris non-CSV di awal file (misalnya log boot ESP32) — otomatis dilewati.")

            st.success(f"✅ File berhasil dimuat: **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom** ditemukan.")

            # Tampilkan preview
            with st.expander("👁️ Preview Data (5 Baris Pertama)"):
                st.dataframe(df_raw.head(), use_container_width=True)

            # ── Column Mapping ──────────────────────────────────────
            st.markdown("#### 🔀 Pemetaan Kolom")
            st.caption(
                "Sistem akan mencoba mencocokkan kolom secara otomatis. "
                "Periksa dan sesuaikan jika ada yang salah."
            )

            auto_map = auto_match_kolom(df_raw.columns.tolist())
            pilihan_kolom = ["— pilih kolom —"] + df_raw.columns.tolist()

            label_fitur = {
                'MQ2'         : '📟 MQ-2 (Sensor Gas/Asap)',
                'MQ135'       : '📟 MQ-135 (Sensor Udara)',
                'Suhu_C'      : '🌡️ Suhu (°C)',
                'Kelembapan_%': '💧 Kelembapan (%)',
            }

            cols_map = st.columns(4)
            kolom_mapping = {}
            for i, fitur in enumerate(FITUR):
                default_col = auto_map.get(fitur, None)
                default_idx = pilihan_kolom.index(default_col) if default_col in pilihan_kolom else 0
                with cols_map[i]:
                    pilihan = st.selectbox(
                        label_fitur[fitur],
                        options=pilihan_kolom,
                        index=default_idx,
                        key=f"map_{fitur}"
                    )
                    kolom_mapping[fitur] = pilihan

            # Cek apakah semua fitur sudah dipetakan
            belum_dipetakan = [f for f, v in kolom_mapping.items() if v == "— pilih kolom —"]
            duplikat = [v for v in kolom_mapping.values() if v != "— pilih kolom —" and
                        list(kolom_mapping.values()).count(v) > 1]

            if belum_dipetakan:
                st.warning(f"⚠️ Fitur berikut belum dipetakan ke kolom: **{', '.join(belum_dipetakan)}**")
            elif duplikat:
                st.error(f"❌ Kolom yang sama dipilih untuk lebih dari satu fitur: **{', '.join(set(duplikat))}**")
            else:
                st.info(
                    "✅ Pemetaan: " +
                    " | ".join([f"`{v}` → **{k}**" for k, v in kolom_mapping.items()])
                )

                if not MODEL_LOADED:
                    st.error("❌ Model tidak tersedia.")
                else:
                    if st.button("🚀 Jalankan Prediksi CSV", type="primary"):
                        with st.spinner(f"Memproses {len(df_raw):,} baris data..."):
                            # Buat DataFrame fitur dengan kolom sesuai mapping
                            X = pd.DataFrame({
                                fitur: pd.to_numeric(df_raw[col], errors='coerce')
                                for fitur, col in kolom_mapping.items()
                            })[FITUR]

                            # ---------------------------------------------------
                            # VALIDASI: Skenario 5 – Deteksi karakter non-numerik
                            # ---------------------------------------------------
                            baris_null = X.isnull().any(axis=1).sum()
                            if baris_null > 0:
                                st.warning(
                                    f"⚠️ Terdeteksi **{baris_null} baris** mengandung nilai non-numerik "
                                    f"(karakter huruf atau simbol tidak valid). "
                                    f"Baris-baris tersebut akan dilewati dari proses prediksi."
                                )
                                X = X.dropna()

                            if len(X) == 0:
                                st.error(
                                    "❌ Tidak ada baris data valid setelah validasi. "
                                    "Pastikan kolom sensor berisi nilai numerik, bukan karakter huruf atau simbol."
                                )
                            else:
                                X_sc  = scaler.transform(X)
                                preds = model.predict(X_sc)
                                proba = model.predict_proba(X_sc)

                                df_hasil = df_raw.loc[X.index].copy()
                                # Rename kolom yang dipetakan agar konsisten di output
                                for fitur, col in kolom_mapping.items():
                                    if col != fitur:
                                        df_hasil[fitur] = df_raw.loc[X.index, col]
                                df_hasil['Prediksi']           = preds
                                df_hasil['Label_Prediksi']     = [KELAS_LABEL_PLAIN.get(p, p) for p in preds]
                                df_hasil['Confidence_(%)']     = [f"{max(p)*100:.2f}" for p in proba]

                                for i, kls in enumerate(model.classes_):
                                    df_hasil[f'Prob_{kls}_(%)'] = [f"{p[i]*100:.2f}" for p in proba]

                                st.markdown("---")
                                st.markdown("### 📊 Ringkasan Hasil Prediksi")

                                # Metrik ringkasan
                                dist = pd.Series(preds).value_counts()
                                cols_metric = st.columns(len(model.classes_))
                                for i, kls in enumerate(model.classes_):
                                    jumlah = dist.get(kls, 0)
                                    persen = jumlah / len(preds) * 100
                                    with cols_metric[i]:
                                        st.markdown(f"""
                                        <div class="metric-box" style="border-top-color:{KELAS_WARNA[kls]};">
                                            <h3 style="color:{KELAS_WARNA[kls]};">{jumlah:,}</h3>
                                            <p>{KELAS_LABEL.get(kls, kls)}</p>
                                            <p style="color:#999; font-size:0.8rem;">{persen:.1f}%</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                st.markdown("<br>", unsafe_allow_html=True)

                                # Pie chart distribusi
                                col_pie, col_tbl = st.columns([1, 1])

                                with col_pie:
                                    st.markdown("#### 🥧 Distribusi Prediksi Kelas")
                                    fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
                                    fig2.patch.set_facecolor('none')
                                    warna_pie = [KELAS_WARNA.get(k, '#888') for k in dist.index]
                                    label_pie = [KELAS_LABEL_PLAIN.get(k, k) for k in dist.index]
                                    ax2.pie(
                                        dist.values,
                                        labels=label_pie,
                                        colors=warna_pie,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        textprops={'fontsize': 10, 'color': '#e2e8f0', 'weight': 'bold'},
                                        wedgeprops={'edgecolor': '#0b0f19', 'linewidth': 2}
                                    )
                                    ax2.set_title('Distribusi Kelas Prediksi', fontweight='bold', color='#e2e8f0')
                                    st.pyplot(fig2)
                                    plt.close()

                                with col_tbl:
                                    st.markdown("#### 📋 Tabel Distribusi")
                                    df_dist = pd.DataFrame({
                                        'Kelas'    : [KELAS_LABEL.get(k, k) for k in dist.index],
                                        'Jumlah'   : dist.values,
                                        'Persentase': [f"{v/len(preds)*100:.2f}%" for v in dist.values]
                                    })
                                    st.dataframe(df_dist, hide_index=True, use_container_width=True)

                                    # Confidence rata-rata per kelas
                                    st.markdown("#### 🎯 Rata-rata Confidence per Kelas")
                                    conf_list = []
                                    for kls in model.classes_:
                                        idx = np.where(preds == kls)[0]
                                        if len(idx) > 0:
                                            avg_conf = np.mean([max(proba[i]) for i in idx]) * 100
                                            conf_list.append({
                                                'Kelas'          : KELAS_LABEL.get(kls, kls),
                                                'Avg Confidence' : f"{avg_conf:.2f}%"
                                            })
                                    st.dataframe(pd.DataFrame(conf_list), hide_index=True, use_container_width=True)

                                # Tabel hasil lengkap
                                st.markdown("---")
                                st.markdown("### 📋 Tabel Hasil Prediksi Lengkap")
                                kolom_tampil = FITUR + ['Label_Prediksi', 'Confidence_(%)']
                                if 'Label' in df_hasil.columns:
                                    kolom_tampil = ['Label'] + kolom_tampil
                                st.dataframe(df_hasil[kolom_tampil], use_container_width=True)

                                # Download hasil
                                csv_hasil = df_hasil.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="⬇️ Download Hasil Prediksi (CSV)",
                                    data=csv_hasil,
                                    file_name="hasil_prediksi.csv",
                                    mime="text/csv",
                                    type="primary"
                                )

        except pd.errors.EmptyDataError:
            st.error("❌ File CSV kosong. Pastikan file memiliki data.")
        except pd.errors.ParserError:
            st.error("❌ Format file tidak valid. Pastikan file dalam format CSV yang benar.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")

# ============================================================
# HALAMAN 3 — PANDUAN
# ============================================================
elif menu == "ℹ️ Panduan Penggunaan":
    st.markdown("### ℹ️ Panduan Penggunaan Sistem")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("""
        #### 📟 Mode Input Manual
        1. Pilih tab **Input Manual Sensor**
        2. Masukkan nilai pembacaan dari masing-masing sensor
        3. Klik tombol **Jalankan Prediksi**
        4. Hasil prediksi dan probabilitas akan ditampilkan

        **Panduan nilai sensor:**
        | Sensor | Udara Normal | Asap Terdeteksi |
        |--------|-------------|-----------------|
        | MQ-2   | 0 – 10      | > 50            |
        | MQ-135 | 200 – 350   | > 500           |
        | Suhu   | 32 – 33°C   | > 34°C          |
        | Kelembapan | 69 – 71% | < 65%          |
        """)

    with col_p2:
        st.markdown("""
        #### 📂 Mode Upload CSV
        1. Pilih tab **Upload File CSV**
        2. Pastikan CSV memiliki kolom:
           - `MQ2` — nilai ADC sensor MQ-2
           - `MQ135` — nilai ADC sensor MQ-135
           - `Suhu_C` — suhu dalam derajat Celcius
           - `Kelembapan_%` — kelembapan dalam persen
        3. Upload file dan tunggu proses
        4. Download hasil prediksi dalam format CSV

        #### ⚠️ Catatan Penting
        - Model dilatih menggunakan data dari wadah uji tertutup
        - Pastikan sensor sudah melewati *warm-up* minimal 5 detik
        - Hasil terbaik pada kondisi ruangan tertutup
        """)

    st.markdown("---")
    st.markdown("#### 🎯 Interpretasi Hasil Prediksi")

    for kls, label in KELAS_LABEL.items():
        warna = KELAS_WARNA[kls]
        st.markdown(f"""
        <div class="guide-card" style="border-left: 4px solid {warna}; margin-bottom: 1rem;">
            <strong style="color:{warna}; font-size: 1.1rem;">{label}</strong><br>
            <small style="color:#cbd5e1; font-size: 0.95rem; display: block; margin-top: 0.5rem;">
            {'Terdeteksi emisi partikel karbon dari pembakaran langsung material kertas. Suhu cenderung tinggi dan kelembapan rendah.' if kls == 'kertas' else
             'Terdeteksi emisi aerosol dari obat nyamuk bakar. Ditandai dengan nilai MQ-135 sedang hingga tinggi.' if kls == 'obat_nyamuk' else
             'Terdeteksi emisi gas dari pembakaran tembakau. Nilai MQ-135 tinggi dan MQ-2 sedikit meningkat.' if kls == 'rokok' else
             'Kondisi udara bersih tanpa kontaminan. Nilai MQ-135 rendah (200-350), kelembapan tinggi dan stabil.'}
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    #### 📚 Referensi Teknis
    - **Model:** Random Forest Classifier (n_estimators=10, random_state=42)
    - **Preprocessing:** StandardScaler (fit pada data latih)
    - **Akurasi model:** 90,93% (session-based split)
    - **Data latih:** 12.969 baris | **Data uji:** 4.728 baris
    - **Penelitian:** Klasifikasi Jenis Asap Menggunakan Sensor Gas Berbasis IoT
      dengan Algoritma Random Forest — Universitas Gunadarma, 2026
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding: 1rem 0;">
    🔥 Sistem Klasifikasi Jenis Asap | Muhammad Fatur Rahman (10122911) |
    Universitas Gunadarma 2026<br>
    Sensor: MQ-2 · MQ-135 · BME280 | Algoritma: Random Forest | Akurasi: 90,93%
</div>
""", unsafe_allow_html=True)