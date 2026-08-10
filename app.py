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

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_comparison import image_comparison
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(
    page_title="WebGIS Kebakaran TNBTS 2026",
    page_icon="🔥",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
IMAGES = {
    "2026-08-02": {"file": os.path.join(IMG_DIR, "aug02.jpg"), "label": "2 Agustus 2026 (Awal / baseline)"},
    "2026-08-07": {"file": os.path.join(IMG_DIR, "aug07.jpg"), "label": "7 Agustus 2026 (Pertengahan)"},
    "2026-08-09": {"file": os.path.join(IMG_DIR, "aug09.jpg"), "label": "9 Agustus 2026 (Terkini)"},
}

_missing = [v["file"] for v in IMAGES.values() if not os.path.isfile(v["file"])]
if _missing:
    st.error(
        "File citra tidak ditemukan di server. Pastikan folder images/ "
        "(berisi aug02.jpg, aug07.jpg, aug09.jpg) sudah di-commit ke repository "
        "GitHub, sejajar dengan app.py -- bukan hanya di dalam file zip.\n\n"
        "File yang hilang: " + ", ".join(_missing) +
        f"\n\nIsi folder {IMG_DIR} saat ini: "
        f"{os.listdir(IMG_DIR) if os.path.isdir(IMG_DIR) else '(folder tidak ada)'}"
    )
    st.stop()

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
# Machine Learning — prediksi luasan dan arah sebaran
# ---------------------------------------------------------------------------
def make_wind_features(df):
    """Transform arah angin (derajat) menjadi komponen sin/cos."""
    out = df.copy()
    rad = np.radians(out["wind_dir_deg"].astype(float) % 360)
    out["wind_sin"] = np.sin(rad)
    out["wind_cos"] = np.cos(rad)
    return out


def train_fire_ml(training_df):
    """
    Melatih dua model Random Forest:
      1) target next_area_ha untuk proyeksi luasan,
      2) target sin/cos arah rambatan api (spread_dir_deg).
    Dataset minimal 8 baris agar model tidak terlalu rapuh.
    """
    required = {
        "area_ha", "wind_speed_ms", "wind_dir_deg",
        "next_area_ha", "spread_dir_deg"
    }
    missing = required - set(training_df.columns)
    if missing:
        raise ValueError("Kolom wajib belum lengkap: " + ", ".join(sorted(missing)))

    df = training_df.copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required)).copy()

    if len(df) < 8:
        raise ValueError("Minimal 8 observasi diperlukan untuk melatih model ML.")

    df = make_wind_features(df)
    features = ["area_ha", "wind_speed_ms", "wind_sin", "wind_cos"]

    X = df[features]
    y_area = df["next_area_ha"]
    theta = np.radians(df["spread_dir_deg"] % 360)
    y_dir = np.column_stack([np.sin(theta), np.cos(theta)])

    area_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        random_state=42
    )
    dir_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=2,
            random_state=42
        )
    )

    area_model.fit(X, y_area)
    dir_model.fit(X, y_dir)

    return area_model, dir_model, features, len(df)


def predict_fire_ml(area_model, dir_model, features, area_ha, wind_speed, wind_dir):
    row = pd.DataFrame([{
        "area_ha": float(area_ha),
        "wind_speed_ms": float(wind_speed),
        "wind_dir_deg": float(wind_dir) % 360,
    }])
    row = make_wind_features(row)
    X = row[features]

    pred_area = float(max(0.1, area_model.predict(X)[0]))
    vec = dir_model.predict(X)[0]
    # spread_dir = arah tujuan rambatan api, konvensi 0=N, 90=E.
    pred_dir = float((np.degrees(np.arctan2(vec[0], vec[1])) + 360) % 360)
    return pred_area, pred_dir


def destination_point(lat, lon, distance_km, bearing_deg):
    """Titik tujuan dari lat/lon, jarak km, dan bearing dari utara."""
    R = 6371.0088
    brng = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    d = distance_km / R

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d)
        + np.cos(lat1) * np.sin(d) * np.cos(brng)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), ((np.degrees(lon2) + 540) % 360) - 180


def fire_footprint(center_lat, center_lon, area_ha, bearing_deg,
                   wind_speed_ms, n=72):
    """
    Membuat footprint elips indikatif.
    Luas elips = pi*a*b; sumbu utama dibuat lebih panjang mengikuti arah angin.
    Ini visualisasi GIS, bukan simulasi fisik api.
    """
    area_km2 = max(area_ha / 100.0, 0.01)
    base_radius = np.sqrt(area_km2 / np.pi)

    # Anisotropi meningkat dengan kecepatan angin, tetapi dibatasi.
    elongation = np.clip(1.5 + 0.10 * float(wind_speed_ms), 1.5, 3.5)
    semi_major = np.sqrt(area_km2 * elongation / np.pi)
    semi_minor = area_km2 / (np.pi * semi_major)

    # Titik pusat footprint digeser ke arah rambatan agar polygon menunjukkan
    # "luasan ke depan", bukan sekadar lingkaran di titik api.
    shift = semi_major * 0.45
    c_lat, c_lon = destination_point(
        center_lat, center_lon, shift, bearing_deg
    )

    lat_scale = 111.32
    lon_scale = 111.32 * np.cos(np.radians(c_lat))

    angles = np.linspace(0, 2*np.pi, n)
    brng = np.radians(bearing_deg)
    # local x = east, local y = north
    x = semi_minor * np.cos(angles)
    y = semi_major * np.sin(angles)

    east = x * np.cos(brng) + y * np.sin(brng)
    north = -x * np.sin(brng) + y * np.cos(brng)

    lats = c_lat + north / lat_scale
    lons = c_lon + east / lon_scale
    return list(zip(lats, lons)), semi_major, semi_minor


def direction_name(deg):
    dirs = ["Utara", "Timur Laut", "Timur", "Tenggara",
            "Selatan", "Barat Daya", "Barat", "Barat Laut"]
    return dirs[int((deg + 22.5) // 45) % 8]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🔥 Kebakaran TNBTS 2026")
st.sidebar.markdown("**App Developer:**")
st.sidebar.markdown("**Dr. Adipandang Yudono**")
st.sidebar.markdown("GIS Analytcis Enthusiast")
st.sidebar.markdown("-------------------------------------------------------")

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

    img1_pil = Image.open(IMAGES[left_key]["file"]).convert("RGB")
    img2_pil = Image.open(IMAGES[right_key]["file"]).convert("RGB")

    image_comparison(
        img1=img1_pil,
        img2=img2_pil,
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
else:
    st.title("Analitik GIS & Machine Learning — Prediksi Sebaran Kebakaran")

    st.warning(
        "Prediksi ML bersifat riset/indikatif. Model membutuhkan data historis "
        "yang memiliki luasan kebakaran, kecepatan angin, arah angin, luasan "
        "berikutnya, dan arah rambatan api. Footprint pada peta adalah visualisasi "
        "GIS berbasis hasil model, bukan simulasi fisik operasional."
    )

    st.subheader("1. Data luasan kebakaran saat ini")
    st.dataframe(
        AREA_DATA.assign(tanggal=AREA_DATA["tanggal"].dt.strftime("%d %b %Y"))[
            ["tanggal", "luas_ha", "catatan"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Trend chart lama tetap dipertahankan.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=AREA_DATA["tanggal"], y=AREA_DATA["luas_ha"],
        mode="lines+markers", name="Luas terdampak (ha) — dilaporkan",
        line=dict(color="firebrick", width=3), marker=dict(size=9),
    ))
    anchors = AREA_DATA[AREA_DATA["pasti"]].copy()
    t0 = anchors["tanggal"].min()
    anchors["t_days"] = (anchors["tanggal"] - t0).dt.days
    logy = np.log(anchors["luas_ha"])
    b, loga = np.polyfit(anchors["t_days"], logy, 1)
    a = np.exp(loga)

    future_dates = pd.date_range(anchors["tanggal"].max(), periods=5, freq="D")
    t_future = (future_dates - t0).days
    pred_mid = a * np.exp(b * t_future)

    recent = anchors[anchors["tanggal"] >= "2026-08-07"]
    b_recent = np.polyfit(
        (recent["tanggal"] - recent["tanggal"].min()).dt.days,
        np.log(recent["luas_ha"]), 1
    )[0]
    b_low = min(b, b_recent) * 0.5
    pred_low = anchors["luas_ha"].iloc[-1] * np.exp(
        b_low * (t_future - t_future[0])
    )
    pred_high = a * np.exp(b_recent * t_future)

    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_mid, mode="lines+markers",
        name="Proyeksi statistik (bukan ML)",
        line=dict(color="orange", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_high, mode="lines",
        name="Skenario tinggi",
        line=dict(color="darkred", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_low, mode="lines",
        name="Skenario rendah",
        line=dict(color="seagreen", width=1, dash="dot"),
        fill="tonexty",
    ))
    fig.update_layout(
        title="Tren luas area terdampak dan proyeksi statistik",
        xaxis_title="Tanggal", yaxis_title="Luas terdampak (ha)",
        legend=dict(orientation="h", y=-0.25), height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Machine Learning: variabel angin → luasan & arah rambatan")

    st.markdown(
        """
Model yang digunakan adalah **Random Forest Regression**. Variabel angin
tidak dimasukkan sebagai teks, tetapi ditransformasi menjadi komponen
`sin(arah angin)` dan `cos(arah angin)` agar sifat data arah yang bersirkular
tetap dapat dipelajari model.

**CSV training minimal** menggunakan kolom:
`area_ha, wind_speed_ms, wind_dir_deg, next_area_ha, spread_dir_deg`

- `area_ha` = luas kebakaran saat observasi (ha)
- `wind_speed_ms` = kecepatan angin (m/s)
- `wind_dir_deg` = arah angin dalam derajat, 0° = Utara, 90° = Timur
- `next_area_ha` = luas kebakaran pada observasi berikutnya (ha)
- `spread_dir_deg` = arah rambatan api aktual/terobservasi, 0° = Utara, 90° = Timur
"""
    )

    template = pd.DataFrame([
        [7.9,  2.5,  70,  44,  55],
        [44,   3.0,  75,  51,  62],
        [51,   4.2,  80,  70,  70],
        [70,   5.1,  82, 176,  65],
        [176,  6.0,  85, 520,  58],
        [520,  4.5,  80, 610,  65],
        [610,  3.8,  78, 690,  68],
        [690,  4.8,  82, 780,  70],
    ], columns=[
        "area_ha", "wind_speed_ms", "wind_dir_deg",
        "next_area_ha", "spread_dir_deg"
    ])

    csv_file = st.file_uploader(
        "Upload CSV data training ML (disarankan ≥ 30 observasi)",
        type=["csv"],
        help="Gunakan data historis yang benar-benar terukur. Template di bawah "
             "hanya untuk memahami struktur kolom dan tidak boleh dianggap "
             "sebagai data observasi lapangan."
    )

    if csv_file is not None:
        training_df = pd.read_csv(csv_file)
        st.caption("Dataset training dari pengguna.")
    else:
        training_df = None
        st.info(
            "Belum ada CSV training. Upload data historis untuk mengaktifkan "
            "Random Forest yang valid. Template berikut hanya contoh struktur."
        )
        st.dataframe(template, use_container_width=True, hide_index=True)

    st.markdown("### 3. Input kondisi angin untuk skenario prediksi")
    c1, c2, c3, c4 = st.columns(4)
    current_area = c1.number_input(
        "Luas saat ini (ha)", min_value=0.1,
        value=float(AREA_DATA["luas_ha"].iloc[-1]), step=10.0
    )
    wind_speed = c2.number_input(
        "Kecepatan angin (m/s)", min_value=0.0,
        value=4.0, step=0.5
    )
    wind_dir = c3.number_input(
        "Arah angin (°)", min_value=0.0, max_value=359.9,
        value=70.0, step=5.0
    )
    horizon = c4.number_input(
        "Horizon prediksi (hari)", min_value=1, max_value=7,
        value=1, step=1
    )

    if training_df is not None:
        try:
            area_model, dir_model, features, n_train = train_fire_ml(training_df)
            pred_area_1, pred_dir = predict_fire_ml(
                area_model, dir_model, features,
                current_area, wind_speed, wind_dir
            )

            # Iterasi sederhana untuk horizon beberapa hari.
            area_iter = current_area
            for _ in range(int(horizon)):
                area_iter, pred_dir = predict_fire_ml(
                    area_model, dir_model, features,
                    area_iter, wind_speed, wind_dir
                )
            pred_area = area_iter

            st.success(
                f"Random Forest aktif — {n_train} observasi training digunakan."
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prediksi luas", f"{pred_area:,.1f} ha")
            m2.metric("Pertambahan", f"{pred_area-current_area:,.1f} ha")
            m3.metric("Arah rambatan ML", f"{pred_dir:.0f}°")
            m4.metric("Sektor", direction_name(pred_dir))

            st.markdown("### 4. Peta GIS footprint prediksi")
            center_lat, center_lon = ANCHOR_LATLON

            footprint, semi_major, semi_minor = fire_footprint(
                center_lat, center_lon, pred_area,
                pred_dir, wind_speed
            )

            # Panjang sumbu utama dan minor dalam km.
            extent_km = semi_major * 1.6

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="Esri.WorldImagery",
            )

            # Citra terkini.
            folium.raster_layers.ImageOverlay(
                image=IMAGES["2026-08-09"]["file"],
                bounds=IMG_BOUNDS,
                opacity=0.65,
                name="Sentinel-2 9 Agustus 2026",
            ).add_to(m)

            # Footprint prediksi.
            folium.Polygon(
                locations=footprint,
                tooltip=(
                    f"Footprint prediksi ML: {pred_area:,.1f} ha | "
                    f"Arah {pred_dir:.0f}° ({direction_name(pred_dir)})"
                ),
                popup=(
                    f"<b>Prediksi ML</b><br>"
                    f"Luas: {pred_area:,.1f} ha<br>"
                    f"Arah: {pred_dir:.0f}° ({direction_name(pred_dir)})<br>"
                    f"Angin: {wind_speed:.1f} m/s dari {wind_dir:.0f}°"
                ),
                color="red",
                weight=3,
                fill=True,
                fill_opacity=0.25,
            ).add_to(m)

            # Titik pusat.
            folium.CircleMarker(
                [center_lat, center_lon],
                radius=7, color="red", fill=True,
                tooltip="Pusat referensi prediksi"
            ).add_to(m)

            # Panah arah rambatan.
            end_lat, end_lon = destination_point(
                center_lat, center_lon, extent_km, pred_dir
            )
            folium.PolyLine(
                [[center_lat, center_lon], [end_lat, end_lon]],
                color="yellow", weight=5,
                tooltip=f"Arah rambatan ML: {pred_dir:.0f}°"
            ).add_to(m)

            folium.Marker(
                [end_lat, end_lon],
                tooltip=f"Arah rambatan: {direction_name(pred_dir)}",
                icon=folium.Icon(color="red", icon="arrow-up", prefix="fa")
            ).add_to(m)

            # Panah angin.
            wind_end_lat, wind_end_lon = destination_point(
                center_lat, center_lon, extent_km * 0.65, wind_dir
            )
            folium.PolyLine(
                [[center_lat, center_lon], [wind_end_lat, wind_end_lon]],
                color="cyan", weight=3, dash_array="6",
                tooltip=f"Arah angin input: {wind_dir:.0f}°"
            ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=1100, height=650)

            st.caption(
                "Footprint adalah polygon elips untuk visualisasi GIS. "
                "Sumbu utama mengikuti arah rambatan hasil ML dan anisotropinya "
                "dipengaruhi kecepatan angin. Jangan digunakan sebagai batas "
                "evakuasi atau keputusan operasional."
            )

            st.markdown("### 5. Ringkasan skenario")
            st.write(
                f"Untuk luas awal **{current_area:,.1f} ha**, angin "
                f"**{wind_speed:.1f} m/s** dengan arah **{wind_dir:.0f}°**, "
                f"model memproyeksikan luas sekitar **{pred_area:,.1f} ha** "
                f"dalam **{int(horizon)} hari** dan arah rambatan "
                f"**{pred_dir:.0f}° ({direction_name(pred_dir)})**."
            )

        except Exception as e:
            st.error(f"Model ML belum dapat dijalankan: {e}")
    else:
        st.warning(
            "Upload CSV training untuk menghasilkan prediksi ML dan footprint "
            "spasial. Tanpa data training, aplikasi sengaja tidak mengklaim "
            "prediksi berbasis machine learning."
        )

    st.markdown("---")
    st.markdown("### 6. Struktur data yang disarankan untuk penelitian")
    st.markdown(
        """
Untuk meningkatkan akurasi, dataset training sebaiknya berasal dari time
series hotspot/dNBR dan data meteorologi pada interval waktu yang sama.
Tambahkan variabel seperti **kelembapan relatif, temperatur, curah hujan,
NDVI/NBR, elevasi, slope, aspect, tutupan lahan, dan jarak ke hotspot
sebelumnya**. Dengan demikian model dapat dikembangkan dari Random Forest
menjadi model **spatio-temporal fire spread** yang lebih kuat.

Arah angin harus dibedakan secara konsisten antara **arah asal angin** dan
**arah tujuan angin**. Pada aplikasi ini input `wind_dir_deg` diperlakukan
sebagai **arah tujuan/pergerakan angin** (0° = Utara, 90° = Timur) agar
mudah digunakan sebagai vektor penggerak footprint.
"""
    )

