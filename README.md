# WebGIS Interaktif — Kebakaran TNBTS Agustus 2026

Aplikasi Streamlit untuk memvisualisasikan perkembangan kebakaran hutan dan
lahan di kawasan Taman Nasional Bromo Tengger Semeru (TNBTS) menggunakan tiga
citra Sentinel-2 L2A True Color (2, 7, dan 9 Agustus 2026), dilengkapi:

- Slider before–after antar tanggal
- Galeri timeline 3 titik waktu
- Peta interaktif dengan overlay citra (georeferensi indikatif)
- Analitik tren luas area terdampak + proyeksi statistik sederhana
- Estimasi arah rambatan api (kualitatif, berbasis kronologi laporan lapangan)

## Struktur folder

```
bromo_fire_webgis/
├── app.py
├── requirements.txt
├── images/
│   ├── aug02.jpg
│   ├── aug07.jpg
│   └── aug09.jpg
└── README.md
```

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke repository GitHub (pastikan folder `images/` ikut ter-push, jangan di-gitignore).
2. Buka https://share.streamlit.io/ → **New app**.
3. Pilih repo, branch, dan `app.py` sebagai main file.
4. Deploy.

## Catatan penting / batasan

- **Georeferensi peta bersifat indikatif**, diturunkan dari skala batang "1 km"
  pada citra dan posisi kawah Bromo, bukan hasil rektifikasi presisi dengan
  Ground Control Points (GCP). Untuk kebutuhan pemetaan presisi, gunakan citra
  asli (GeoTIFF) dari Copernicus/Sentinel Hub, bukan file JPG hasil ekspor.
- **Deteksi burn-scar otomatis dari RGB true color TIDAK digunakan** dalam
  aplikasi ini karena hasil pengujian menunjukkan bias tinggi (bayangan lereng
  dan vegetasi gelap ikut terklasifikasi sebagai area terbakar). Data luas
  area yang ditampilkan bersumber dari laporan resmi BPBD/TNBTS di media.
- **Proyeksi luas area** adalah ekstrapolasi statistik (regresi eksponensial)
  dari data historis yang dilaporkan, bukan model perilaku api fisik. Untuk
  hasil yang lebih andal, integrasikan data NBR/dNBR dari band SWIR/NIR
  (B12/B8A) dan hotspot NASA FIRMS/VIIRS.
- **Arah rambatan api** ditampilkan berdasarkan kronologi laporan lapangan
  (Blok Bantengan → Bukit B-29 → JLKT → Gunung Kursi), bukan hasil deteksi
  vektor otomatis dari citra.

## Rekomendasi pengembangan lanjutan

- Tambahkan band SWIR (B12) dan NIR (B8A) dari Sentinel Hub API untuk hitung
  dNBR (differenced Normalized Burn Ratio) — jauh lebih akurat untuk deteksi
  area terbakar dibanding interpretasi RGB.
- Tarik data hotspot VIIRS/MODIS NASA FIRMS via API untuk overlay titik panas
  real-time dan validasi arah rambatan.
- Tambahkan layer angin (BMKG/ERA5) untuk analisis korelasi arah asap–angin.
