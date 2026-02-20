import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="SPK BANSOS", layout="wide")
st.title("Dashboard SPK Bansos Jakarta Barat")
st.caption(
    "Analisis ketimpangan bantuan sosial dan beban pendamping berbasis clustering."
)

# Kolom numerik yang dipakai untuk scaling dan clustering.
NUMERIC_COLS = ["Total_PBI", "Total_PKH", "Total_BPNT", "Jumlah_Pendamping"]
# Kolom identitas wilayah yang wajib ada di data.
AREA_COLS = ["Kecamatan", "Kelurahan"]


def clean_number(value):
    """
    Membersihkan nilai angka dengan format campuran.
    Contoh yang ditangani: 1,234 | 1.234 | Rp 1.234 | " 2.500 "
    Jika gagal diparse, nilai dikembalikan sebagai 0.0 agar pipeline tidak putus.
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    txt = str(value).strip()
    txt = re.sub(r"[^0-9,.-]", "", txt)

    if "," in txt and "." in txt:
        txt = txt.replace(".", "")
        txt = txt.replace(",", ".")
    elif "," in txt and "." not in txt:
        txt = txt.replace(",", "")
    elif "." in txt and txt.count(".") > 1:
        txt = txt.replace(".", "")

    try:
        return float(txt)
    except ValueError:
        return 0.0


def preprocess_data(df):
    """
    Tahap pra-pemrosesan:
    1) Bersihkan seluruh kolom numerik.
    2) Isi data wilayah kosong agar tidak error saat groupby/visualisasi.
    3) Bentuk fitur turunan untuk analisis beban bantuan.
    """
    df_clean = df.copy()
    for col in NUMERIC_COLS:
        df_clean[col] = df_clean[col].apply(clean_number)

    # Isi kolom area yang kosong supaya analisis tidak error.
    for col in AREA_COLS:
        df_clean[col] = df_clean[col].fillna("Tidak diketahui")

    df_clean["Total_Bantuan"] = (
        df_clean["Total_PBI"] + df_clean["Total_PKH"] + df_clean["Total_BPNT"]
    )
    df_clean["Rasio_Beban_Pendamping"] = (
        df_clean["Total_Bantuan"] / (df_clean["Jumlah_Pendamping"] + 1)
    )
    return df_clean


def add_priority_score(df):
    """
    Hitung skor prioritas (0-1) berbasis Min-Max Scaling.
    Logika:
    - Total bantuan (PBI, PKH, BPNT) makin tinggi => prioritas naik.
    - Jumlah pendamping makin rendah => prioritas naik.
    """
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[NUMERIC_COLS])
    scaled = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in NUMERIC_COLS])

    # Score tinggi = bantuan tinggi + pendamping minim.
    score = (
        scaled["Total_PBI_scaled"]
        + scaled["Total_PKH_scaled"]
        + scaled["Total_BPNT_scaled"]
        + (1 - scaled["Jumlah_Pendamping_scaled"])
    ) / 4
    df_with_score = df.copy()
    df_with_score["Skor_Prioritas"] = score
    return df_with_score


def run_clustering(df, algo_name, kmeans_k, dbscan_eps, dbscan_min_samples):
    """
    Jalankan clustering pada data yang sudah di-scale (Min-Max).
    - K-Means: cocok untuk segmentasi global dengan jumlah cluster tetap.
    - DBSCAN: cocok untuk menemukan wilayah padat/outlier berbasis densitas.
    """
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[NUMERIC_COLS])

    if algo_name == "K-Means":
        model = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
    else:
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(features)

    result = df.copy()
    result["Cluster"] = labels
    return result


def plot_scatter_clusters(df, algorithm_name):
    """Visualisasi peta cluster: total bantuan vs jumlah pendamping."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x="Total_Bantuan",
        y="Jumlah_Pendamping",
        hue="Cluster",
        palette="viridis",
        data=df,
        ax=ax,
    )
    ax.set_title(f"Peta Cluster ({algorithm_name})")
    ax.set_xlabel("Total Bantuan (PBI+PKH+BPNT)")
    ax.set_ylabel("Jumlah Pendamping")
    return fig


# =====================
# Sidebar: kontrol input
# =====================
st.sidebar.header("Kontrol Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type="csv")

st.sidebar.subheader("Mode Clustering")
algorithm = st.sidebar.selectbox("Pilih algoritma", ["K-Means", "DBSCAN"])

kmeans_k = st.sidebar.slider("Jumlah cluster K-Means (K)", 2, 8, 3)
dbscan_eps = st.sidebar.slider("DBSCAN eps", 0.05, 1.0, 0.25, 0.05)
dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

st.sidebar.subheader("Refresh Data")
auto_refresh = st.sidebar.checkbox("Auto refresh dashboard", value=False)
refresh_seconds = st.sidebar.slider("Interval refresh (detik)", 5, 120, 30)
manual_refresh = st.sidebar.button("Refresh sekarang")

if auto_refresh and st_autorefresh is not None:
    # Auto rerun berbasis interval jika package tambahan tersedia.
    st_autorefresh(interval=refresh_seconds * 1000, key="refresh_counter")
elif auto_refresh and st_autorefresh is None:
    st.sidebar.info(
        "Install `streamlit-autorefresh` untuk auto refresh: `pip install streamlit-autorefresh`."
    )
    if manual_refresh:
        time.sleep(0.2)
        st.rerun()
elif manual_refresh:
    st.rerun()

if uploaded_file is None:
    # Stop awal agar user tahu file belum diunggah.
    st.warning("Silakan upload file CSV data bansos dari sidebar.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
missing_cols = [c for c in AREA_COLS + NUMERIC_COLS if c not in raw_df.columns]
if missing_cols:
    # Validasi struktur CSV sebelum analisis.
    st.error(f"Kolom wajib belum ada di CSV: {', '.join(missing_cols)}")
    st.stop()

# Pipeline inti analisis:
# 1) Preprocessing data kotor.
# 2) Hitung skor prioritas.
# 3) Lakukan clustering.
df = preprocess_data(raw_df)
df = add_priority_score(df)
df = run_clustering(df, algorithm, kmeans_k, dbscan_eps, dbscan_min_samples)

st.caption(f"Pembaruan terakhir: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

total_pbi = int(df["Total_PBI"].sum())
total_pkh = int(df["Total_PKH"].sum())
total_bpnt = int(df["Total_BPNT"].sum())
total_pendamping = int(df["Jumlah_Pendamping"].sum())
total_bantuan = int(df["Total_Bantuan"].sum())
rasio_global = total_bantuan / (total_pendamping + 1)

# KPI ringkas untuk monitoring cepat.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PBI", f"{total_pbi:,}")
c2.metric("Total PKH", f"{total_pkh:,}")
c3.metric("Total BPNT", f"{total_bpnt:,}")
c4.metric("Total Pendamping", f"{total_pendamping:,}")
c5.metric("Rasio Beban Global", f"{rasio_global:,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Ringkasan Prioritas",
        "Analisis Cluster",
        "Monitoring Kecamatan",
        "Data Detail",
    ]
)

with tab1:
    # Menampilkan wilayah yang paling perlu intervensi.
    st.subheader("Top 10 Wilayah Prioritas Intervensi")
    st.caption("Skor tinggi berarti beban bantuan tinggi dengan pendamping relatif minim.")
    top10 = df.sort_values("Skor_Prioritas", ascending=False).head(10)
    st.dataframe(
        top10[
            [
                "Kecamatan",
                "Kelurahan",
                "Total_Bantuan",
                "Jumlah_Pendamping",
                "Rasio_Beban_Pendamping",
                "Skor_Prioritas",
            ]
        ],
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=top10,
        x="Kelurahan",
        y="Rasio_Beban_Pendamping",
        hue="Kecamatan",
        ax=ax,
    )
    ax.set_title("10 Kelurahan dengan Rasio Beban Tertinggi")
    ax.set_xlabel("Kelurahan")
    ax.set_ylabel("Rasio Beban per Pendamping")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with tab2:
    # Menampilkan hasil segmentasi wilayah berdasarkan algoritma terpilih.
    st.subheader(f"Hasil Clustering: {algorithm}")
    fig = plot_scatter_clusters(df, algorithm)
    st.pyplot(fig)

    cluster_summary = (
        df.groupby("Cluster", dropna=False)
        .agg(
            Jumlah_Wilayah=("Kelurahan", "count"),
            Total_Bantuan=("Total_Bantuan", "sum"),
            Rata_Rasio_Beban=("Rasio_Beban_Pendamping", "mean"),
        )
        .reset_index()
        .sort_values("Total_Bantuan", ascending=False)
    )
    st.dataframe(cluster_summary, use_container_width=True)
    st.info(
        "Gunakan ringkasan cluster untuk menentukan prioritas distribusi dan penempatan pendamping."
    )

with tab3:
    # Monitoring komparatif antar kecamatan / per kecamatan terpilih.
    st.subheader("Perbandingan Beban Bantuan per Kecamatan")
    kec_options = ["Semua Kecamatan"] + sorted(df["Kecamatan"].unique().tolist())
    selected_kec = st.selectbox("Filter kecamatan", kec_options)

    if selected_kec == "Semua Kecamatan":
        monitor_df = df.copy()
    else:
        monitor_df = df[df["Kecamatan"] == selected_kec].copy()

    kec_summary = (
        monitor_df.groupby("Kecamatan", dropna=False)
        .agg(
            Total_Bantuan=("Total_Bantuan", "sum"),
            Total_Pendamping=("Jumlah_Pendamping", "sum"),
            Rata_Rasio_Beban=("Rasio_Beban_Pendamping", "mean"),
        )
        .reset_index()
    )
    st.dataframe(kec_summary, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    sns.scatterplot(
        data=monitor_df,
        x="Total_Bantuan",
        y="Jumlah_Pendamping",
        hue="Kecamatan",
        size="Skor_Prioritas",
        sizes=(40, 300),
        ax=ax2,
    )
    ax2.set_title("Beban Bantuan vs Jumlah Pendamping")
    ax2.set_xlabel("Total Bantuan")
    ax2.set_ylabel("Jumlah Pendamping")
    st.pyplot(fig2)

with tab4:
    # Data akhir untuk audit, verifikasi, dan ekspor lanjutan.
    st.subheader("Data Lengkap + Hasil Analisis")
    shown_cols = AREA_COLS + NUMERIC_COLS + [
        "Total_Bantuan",
        "Rasio_Beban_Pendamping",
        "Skor_Prioritas",
        "Cluster",
    ]
    st.dataframe(df[shown_cols], use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil analisis (CSV)",
        data=csv,
        file_name="hasil_analisis_bansos.csv",
        mime="text/csv",
    )
