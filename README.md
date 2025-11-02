# Radar Harga Hotel (Makcorps + Streamlit + SQLite)

Bandingkan harga hotel yang Anda pilih dengan kompetitor di sekitarnya.
Cukup ketik nama hotel. Kota bisa dikosongkan; aplikasi akan mencoba beberapa kota (daftar dapat diubah).

## Fitur
- Comp-set otomatis atau manual
- Parity alert (selisih > ambang %)
- Vendor termurah per hotel
- Snapshot ke SQLite untuk tren vendor termurah

## Batasan (Makcorps Free)
- Tanggal acak di masa depan
- Maks 30 hotel per kota
- Tidak ada pax/jenis kamar

## Cara Jalankan
```bash
pip install -r requirements.txt
streamlit run app.py
