"""
WebGIS Interaktif — Monitoring Kebakaran TNBTS (Agustus 2026)
Dibuat untuk analisis before-after citra Sentinel-2 L2A True Color
serta analitik tren & proyeksi sederhana luas area terdampak.

PENTING: Aplikasi ini BUKAN produk operasional resmi. Estimasi arah dan
proyeksi luas bersifat indikatif, dibangun dari data luasan yang dilaporkan
BPBD/TNBTS di media serta interpretasi visual citra Sentinel-2 true color.
Untuk keputusan operasional, rujuk data resmi BNPB, SiPongi+ (KLHK), dan
hotspot NASA FIRMS/VIIRS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_comparison import image_comparison
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="WebGIS Kebakaran TNBTS 2026",
    page_icon="🔥",
    layout="wide",
)

IMG_DIR = "images"
IMAGES = {
    "2026-08-02": {"file": f"{IMG_DIR}/aug02.jpg", "label": "2 Agustus 2026 (Awal / baseline)"},
    "2026-08-07": {"file": f"{IMG_DIR}/aug07.jpg", "label": "7 Agustus 2026 (Pertengahan)"},
    "2026-08-09": {"file": f"{IMG_DIR}/aug09.jpg", "label": "9 Agustus 2026 (Terkini)"},
}

# ---------------------------------------------------------------------------
# Data luas area terbakar — dikutip dari pemberitaan (BPBD Probolinggo / TNBTS)
# ---------------------------------------------------------------------------
AREA_DATA = pd.DataFrame([
    {"tanggal": "2026-08-03", "luas_ha": 7.9,   "catatan": "Titik api pertama, Blok Bantengan/Sariwani", "pasti": True},
    {"tanggal": "2026-08-04", "luas_ha": 44,    "catatan": "Estimasi urutan laporan (tanggal pasti tidak dipublikasikan)", "pasti": False},
    {"tanggal": "2026-08-05", "luas_ha": 51,    "catatan": "Estimasi urutan laporan (tanggal pasti tidak dipublikasikan)", "pasti": False},
    {"tanggal": "2026-08-06", "luas_ha": 70,    "catatan": "Dilaporkan Poskota per Kamis 6/8", "pasti": True},
    {"tanggal": "2026-08-07", "luas_ha": 176,   "catatan": "Dilaporkan Jumat malam 7/8, penutupan total kawasan", "pasti": True},
    {"tanggal": "2026-08-09", "luas_ha": 520,   "catatan": "Dilaporkan Minggu 9/8, masih dalam pendataan", "pasti": True},
])
AREA_DATA["tanggal"] = pd.to_datetime(AREA_DATA["tanggal"])

# ---------------------------------------------------------------------------
# Georeferensi indikatif (BUKAN hasil rektifikasi presisi / tanpa GCP survei)
# Diturunkan dari skala batang "1 km" pada citra (~57.5 px/km) dan anchor
# kawah Bromo (piksel 410,295) pada koordinat kawah Bromo ~7.9425S 112.9501E
# ---------------------------------------------------------------------------
PX_PER_KM = 57.5
ANCHOR_PX = (410, 295)
ANCHOR_LATLON = (-7.9425, 112.9501)
IMG_W, IMG_H = 1020, 831

dlat_per_px = (1 / PX_PER_KM) / 111.32
dlon_per_px = (1 / PX_PER_KM) / (111.32 * np.cos(np.radians(abs(ANCHOR_LATLON[0]))))

def px_to_latlon(x, y):
    lat = ANCHOR_LATLON[0] - (y - ANCHOR_PX[1]) * dlat_per_px
    lon = ANCHOR_LATLON[1] + (x - ANCHOR_PX[0]) * dlon_per_px
    return lat, lon

north_lat, _ = px_to_latlon(ANCHOR_PX[0], 0)
south_lat, _ = px_to_latlon(ANCHOR_PX[0], IMG_H)
_, west_lon = px_to_latlon(0, ANCHOR_PX[1])
_, east_lon = px_to_latlon(IMG_W, ANCHOR_PX[1])
IMG_BOUNDS = [[south_lat, west_lon], [north_lat, east_lon]]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🔥 Kebakaran TNBTS 2026")
st.sidebar.markdown(
    "Monitoring visual & analitik sederhana perkembangan kebakaran hutan "
    "dan lahan di kawasan Taman Nasional Bromo Tengger Semeru, "
    "menggunakan citra **Sentinel-2 L2A True Color**."
)
st.sidebar.info(
    "⚠️ Bukan produk operasional resmi. Rujuk BNPB / SiPongi+ KLHK / "
    "NASA FIRMS untuk data hotspot real-time."
)
page = st.sidebar.radio(
    "Navigasi",
    ["Before–After Slider", "Timeline 3 Waktu", "Peta Interaktif", "Analitik & Prediksi Sebaran"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber citra: Copernicus Sentinel-2 data 2026, diproses via Copernicus Browser.\n\n"
    "Sumber data luas area: pemberitaan BPBD Kab. Probolinggo, TNBTS, dan media "
    "nasional (Agustus 2026)."
)

# ---------------------------------------------------------------------------
# PAGE 1 — Before After Slider
# ---------------------------------------------------------------------------
if page == "Before–After Slider":
    st.title("Perbandingan Before–After: Citra Sentinel-2")
    st.markdown(
        "Geser slider untuk membandingkan kondisi kawasan kaldera Tengger–Bromo "
        "pada tiga titik waktu berbeda."
    )

    pair = st.selectbox(
        "Pilih pasangan pembanding",
        [
            "2 Agustus → 9 Agustus (Awal vs Terkini)",
            "2 Agustus → 7 Agustus (Awal vs Pertengahan)",
            "7 Agustus → 9 Agustus (Pertengahan vs Terkini)",
        ],
    )

    mapping = {
        "2 Agustus → 9 Agustus (Awal vs Terkini)": ("2026-08-02", "2026-08-09"),
        "2 Agustus → 7 Agustus (Awal vs Pertengahan)": ("2026-08-02", "2026-08-07"),
        "7 Agustus → 9 Agustus (Pertengahan vs Terkini)": ("2026-08-07", "2026-08-09"),
    }
    left_key, right_key = mapping[pair]

    image_comparison(
        img1=IMAGES[left_key]["file"],
        img2=IMAGES[right_key]["file"],
        label1=IMAGES[left_key]["label"],
        label2=IMAGES[right_key]["label"],
        width=900,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
    )

    st.markdown("### Pengamatan visual")
    st.markdown(
        """
- **2 Agustus** — kondisi vegetasi flank kaldera masih utuh, belum ada tanda kebakaran; area kawah (tengah) tampak sebagai hamparan pasir vulkanik alami — bukan hasil terbakar.
- **7 Agustus** — mulai tampak area gelap/gosong pada flank barat daya–selatan kaldera (dekat Ngadas), serta sedikit asap tipis; area terdampak resmi ~176 ha saat ini.
- **9 Agustus** — jejak bakar (burn scar) meluas signifikan pada sisi selatan–timur kaldera, disertai kepulan asap tebal yang terbawa angin ke arah timur/timur laut (ke arah Wonokerso–Ledokombo); area terdampak resmi ~520 ha.
        """
    )

# ---------------------------------------------------------------------------
# PAGE 2 — Timeline
# ---------------------------------------------------------------------------
elif page == "Timeline 3 Waktu":
    st.title("Timeline Perkembangan Kebakaran — 3 Titik Waktu")
    cols = st.columns(3)
    stats = {
        "2026-08-02": "Baseline — belum ada kebakaran",
        "2026-08-07": "±176 ha terdampak (dilaporkan resmi)",
        "2026-08-09": "±520 ha terdampak (dilaporkan resmi)",
    }
    for col, key in zip(cols, IMAGES.keys()):
        with col:
            st.image(IMAGES[key]["file"], use_container_width=True)
            st.markdown(f"**{IMAGES[key]['label']}**")
            st.caption(stats[key])

    st.markdown("---")
    st.markdown(
        "Kronologi singkat (berdasarkan laporan BPBD Kab. Probolinggo & TNBTS): "
        "titik api pertama terdeteksi **Senin, 3 Agustus 2026** di sekitar **Blok "
        "Bantengan, Desa Sariwani, Kec. Sukapura**, lalu merembet ke **Bukit "
        "B-29**, jalur **Lingkar Kaldera Tengger (JLKT)**, hingga mengarah ke "
        "**Gunung Kursi**. Seluruh akses wisata TNBTS ditutup total sejak "
        "**Sabtu, 8 Agustus 2026 pukul 22.00 WIB**."
    )

# ---------------------------------------------------------------------------
# PAGE 3 — Peta interaktif
# ---------------------------------------------------------------------------
elif page == "Peta Interaktif":
    st.title("Peta Interaktif — Overlay Citra Terkini")
    st.warning(
        "Georeferensi bersifat **indikatif** (diturunkan dari skala batang citra "
        "dan posisi kawah Bromo), bukan hasil rektifikasi presisi dengan GCP. "
        "Gunakan hanya untuk konteks spasial umum, bukan pengukuran presisi."
    )

    which = st.radio(
        "Tampilkan overlay citra tanggal:",
        list(IMAGES.keys()),
        index=2,
        horizontal=True,
        format_func=lambda k: IMAGES[k]["label"],
    )

    m = folium.Map(
        location=[ANCHOR_LATLON[0], ANCHOR_LATLON[1]],
        zoom_start=13,
        tiles="Esri.WorldImagery",
    )
    folium.raster_layers.ImageOverlay(
        image=IMAGES[which]["file"],
        bounds=IMG_BOUNDS,
        opacity=0.9,
        name=IMAGES[which]["label"],
    ).add_to(m)

    folium.Marker(
        location=[ANCHOR_LATLON[0], ANCHOR_LATLON[1]],
        tooltip="Kawah Bromo (anchor georeferensi)",
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=[-7.9339, 112.9538],
        tooltip="Cemoro Lawang (gerbang wisata)",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    # Panah arah rambatan api (kualitatif, dari kronologi berita — bukan hasil
    # deteksi otomatis)
    spread_path = [
        [-7.965, 112.905],   # sekitar Blok Bantengan / Sariwani (approx, kualitatif)
        [-7.950, 112.930],   # Bukit B-29 (approx)
        [-7.935, 112.955],   # JLKT (approx, dekat kaldera)
        [-7.920, 112.975],   # arah Gunung Kursi (approx)
    ]
    folium.PolyLine(
        spread_path, color="orange", weight=4, dash_array="8",
        tooltip="Arah rambatan api (kualitatif, berdasarkan kronologi laporan lapangan): "
                "Blok Bantengan → Bukit B-29 → JLKT → G. Kursi",
    ).add_to(m)
    for pt, name in zip(spread_path, ["Blok Bantengan", "Bukit B-29", "JLKT", "→ G. Kursi"]):
        folium.CircleMarker(pt, radius=5, color="orange", fill=True, tooltip=name).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)

    st.caption(
        "Koordinat Blok Bantengan, Bukit B-29, JLKT, dan Gunung Kursi pada peta "
        "bersifat **perkiraan kualitatif** untuk ilustrasi arah rambatan sesuai "
        "narasi berita, bukan koordinat survei presisi."
    )

# ---------------------------------------------------------------------------
# PAGE 4 — Analitik & Prediksi
# ---------------------------------------------------------------------------
else:
    st.title("Analitik Tren & Proyeksi Sebaran Kebakaran")
    st.warning(
        "Proyeksi di bawah adalah **ekstrapolasi statistik sederhana** dari data "
        "luas area yang dilaporkan resmi — **bukan** model perilaku api fisik "
        "(yang idealnya memerlukan data angin, kelembapan bahan bakar, "
        "topografi, mis. FARSITE/Cell2Fire/FlamMap). Gunakan sebagai indikasi "
        "skenario, bukan prediksi definitif."
    )

    st.subheader("1. Data luas area terdampak (dilaporkan resmi)")
    st.dataframe(
        AREA_DATA.assign(tanggal=AREA_DATA["tanggal"].dt.strftime("%d %b %Y"))[
            ["tanggal", "luas_ha", "catatan"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=AREA_DATA["tanggal"], y=AREA_DATA["luas_ha"],
        mode="lines+markers", name="Luas terdampak (ha) — dilaporkan",
        line=dict(color="firebrick", width=3), marker=dict(size=9),
    ))

    # Exponential fit using well-dated anchor points only (3, 7, 9 Aug)
    anchors = AREA_DATA[AREA_DATA["pasti"]].copy()
    t0 = anchors["tanggal"].min()
    anchors["t_days"] = (anchors["tanggal"] - t0).dt.days
    logy = np.log(anchors["luas_ha"])
    b, loga = np.polyfit(anchors["t_days"], logy, 1)
    a = np.exp(loga)

    future_dates = pd.date_range(anchors["tanggal"].max(), periods=5, freq="D")
    t_future = (future_dates - t0).days
    pred_mid = a * np.exp(b * t_future)

    # Bound scenario: slower growth rate = average of full period vs latest 2-day burst
    recent = anchors[anchors["tanggal"] >= "2026-08-07"]
    b_recent = np.polyfit((recent["tanggal"] - recent["tanggal"].min()).dt.days,
                           np.log(recent["luas_ha"]), 1)[0]
    b_low = min(b, b_recent) * 0.5  # skenario optimis: laju melambat separuh (efek pemadaman)
    pred_low = anchors["luas_ha"].iloc[-1] * np.exp(
        b_low * (t_future - t_future[0])
    )
    pred_high = a * np.exp(b_recent * t_future)

    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_mid, mode="lines+markers", name="Proyeksi (tren rata-rata)",
        line=dict(color="orange", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_high, mode="lines", name="Skenario terburuk (laju 2 hari terakhir)",
        line=dict(color="darkred", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_low, mode="lines", name="Skenario optimis (laju melambat, pemadaman efektif)",
        line=dict(color="seagreen", width=1, dash="dot"),
        fill="tonexty",
    ))
    fig.update_layout(
        title="Tren luas area terdampak & proyeksi 4 hari ke depan",
        xaxis_title="Tanggal", yaxis_title="Luas terdampak (ha)",
        legend=dict(orientation="h", y=-0.25),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    doubling_time = np.log(2) / b
    c1, c2, c3 = st.columns(3)
    c1.metric("Laju pertumbuhan rata-rata (3–9 Agu)", f"{b*100:.1f} %/hari")
    c2.metric("Estimasi waktu berlipat ganda", f"{doubling_time:.1f} hari")
    c3.metric("Proyeksi 12 Agustus (skenario tengah)", f"{pred_mid[-1]:,.0f} ha")

    st.markdown("### 2. Estimasi arah sebaran (kualitatif)")
    st.markdown(
        """
Berdasarkan kronologi laporan lapangan dan interpretasi visual citra 9 Agustus
(kepulan asap tebal mengarah ke sisi timur–timur laut kaldera):

- **Titik awal:** Blok Bantengan, Desa Sariwani (sisi barat daya kaldera, dekat Sukapura).
- **Arah rambatan utama:** menyusuri Bukit B‑29 → Jalur Lingkar Kaldera Tengger (JLKT) → mengarah ke Gunung Kursi — pola umum **barat daya → timur laut**, mengikuti punggungan kaldera.
- **Risiko lanjutan:** potensi merembet ke **utara (kawasan Watangan)** apabila arah angin berbalik/menguat, sebagaimana diwaspadai oleh BPBD Kab. Probolinggo.
- Pola ini **konsisten** dengan jejak gosong dan arah kepulan asap yang tampak pada citra 9 Agustus di sisi selatan–timur kaldera.
        """
    )

    st.markdown("### 3. Rekomendasi tindak lanjut analisis (untuk kebutuhan riset/MEWLAFOR)")
    st.markdown(
        """
1. Gunakan band **SWIR/NIR** (Sentinel-2 B12/B8A) untuk menghitung **dNBR** (differenced Normalized Burn Ratio) guna estimasi luas & tingkat keparahan bakar yang jauh lebih akurat dibanding interpretasi RGB.
2. Tarik data hotspot **VIIRS/MODIS (NASA FIRMS)** time series untuk memvalidasi arah dan kecepatan rambatan api per jam.
3. Overlay dengan **data angin** (BMKG/ERA5) untuk mengaitkan arah asap dengan pola angin harian.
4. Untuk proposal MEWLAFOR, hasil dNBR pasca-kebakaran ini bisa langsung menjadi *baseline* kebutuhan rehabilitasi (indikator NDVI-recovery time series ke depan).
        """
    )
