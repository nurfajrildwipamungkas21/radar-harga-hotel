# app.py
# =============================================================================
# Any Hotel → Auto Competitor — Rate Parity Monitor (+ Gemini Insights)
# Makcorps Free API (harga) + Geoapify (peta & pencarian hotel terdekat)
# Streamlit single-file app + optional SQLite snapshot + Gemini 2.5 Flash
# =============================================================================

import os
import sqlite3
import difflib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from urllib.parse import quote
from math import cos, radians  # untuk jarak approx

# ----------------------------- Demo API Keys (tanam) -------------------------
# ⚠️ Untuk demo saja. Ganti ke env var untuk produksi.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCi19OsrR1lsoN7qs2EU5U4zP-8j_1eHh4")
MAKCORPS_JWT_DEMO = os.getenv("MAKCORPS_JWT", "69075db1f82e933a8d57fcbd")
GEOAPIFY_KEY_DEMO = os.getenv("GEOAPIFY_KEY", "4398724ca14044ee93d70a31d0147993")

# ----------------------------- Config ---------------------------------------
DEFAULT_CITY = ""  # opsional
DEFAULT_HOTEL = "Satya Graha Hotel"
DB_DEFAULT = "hotel_competitor.sqlite"
VENDOR_RATES_TABLE = "vendor_rates"
DEFAULT_CITY_TRY = (
    "yogyakarta, sleman, bantul, depok, kasihan, jakarta, bandung, surabaya, denpasar"
)
DEFAULT_RADIUS_M = 25_000  # 25 km untuk Geoapify Places

SAMPLE_DATA = [
    [
        {"hotelName": "Satya Graha Hotel", "hotelId": "999001"},
        [
            {"price1": "27", "tax1": "3", "vendor1": "Booking.com"},
            {"price2": "26", "tax2": "3", "vendor2": "Agoda.com"},
            {"price3": "25", "tax3": "5", "vendor3": "Priceline"},
            {"price4": "29", "tax4": "3", "vendor4": "Trip.com"},
        ],
    ],
    [
        {"hotelName": "Hotel Neo Malioboro", "hotelId": "a1"},
        [
            {"price1": "24", "tax1": "3", "vendor1": "Booking.com"},
            {"price2": "26", "tax2": "0", "vendor2": "Priceline"},
            {"price3": "25", "tax3": "2", "vendor3": "Hotels.com"},
        ],
    ],
    [
        {"hotelName": "Ibis Styles Yogyakarta", "hotelId": "b2"},
        [
            {"price1": "28", "tax1": "3", "vendor1": "Booking.com"},
            {"price2": "27", "tax2": "3", "vendor2": "Trip.com"},
        ],
    ],
]

# --------------------------- Helper (Makcorps) -------------------------------

def fuzzy_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def flatten_makcorps(raw: List) -> pd.DataFrame:
    """Ubah respons Makcorps Free menjadi DataFrame rapi."""
    rows = []
    for item in raw or []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        meta, quotes = item
        hotel_name = (meta or {}).get("hotelName")
        hotel_id = (meta or {}).get("hotelId")
        if not hotel_name or not isinstance(quotes, list):
            continue
        for q in quotes:
            for k, v in list(q.items()):
                if k.startswith("vendor") and v:
                    idx = k.replace("vendor", "")
                    vendor = str(v)
                    price = q.get(f"price{idx}")
                    tax = q.get(f"tax{idx}")
                    try:
                        price_f = float(price) if price is not None else None
                        tax_f = float(tax) if tax is not None else 0.0
                    except Exception:
                        price_f, tax_f = None, 0.0
                    if price_f is None:
                        continue
                    total = price_f + (tax_f or 0.0)
                    rows.append(
                        {
                            "hotel_name": hotel_name,
                            "hotel_id": hotel_id,
                            "vendor": vendor,
                            "price": price_f,
                            "tax": tax_f,
                            "total": total,
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["total"] > 0]
    return df


def fetch_makcorps_city(city: str, jwt_token: Optional[str]) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Kembalikan (DataFrame, fetched_at, warn). Tanpa JWT → fallback SAMPLE_DATA."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    url = f"https://api.makcorps.com/free/{city.strip()}"
    headers = {"Authorization": f"JWT {jwt_token}"} if jwt_token else {}
    try:
        if not jwt_token:
            raise RuntimeError("No JWT provided; using SAMPLE_DATA for demo.")
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        raw = r.json()
        df = flatten_makcorps(raw)
        if df.empty:
            raise RuntimeError("API returned no rows; using SAMPLE_DATA.")
        return df, ts, None
    except Exception as e:
        df = flatten_makcorps(SAMPLE_DATA)
        # sengaja tidak menampilkan warn ke UI agar tampak profesional
        return df, ts, f"Using SAMPLE_DATA because: {e}"


def find_hotel_id(df: pd.DataFrame, target: str) -> Optional[str]:
    if df.empty:
        return None
    names = df[["hotel_id", "hotel_name"]].drop_duplicates()
    best = None
    best_score = 0.0
    for hid, hname in names.values:
        sc = fuzzy_score(hname, target)
        if sc > best_score:
            best_score = sc
            best = hid
    return best


def select_competitors(
    df: pd.DataFrame,
    our_id: Optional[str],
    manual_list: List[str],
    k: int = 8,
    price_band: Tuple[float, float] = (0.7, 1.3),
    include_nonhotel: bool = False,
) -> pd.DataFrame:
    """Ambil comp-set otomatis/manuaI, filter band harga relatif ke hotel kita."""
    agg = (
        df.groupby(["hotel_id", "hotel_name"], as_index=False)["total"]
        .min()
        .rename(columns={"total": "min_total"})
    )

    # Jika ada manual_list → match fuzzy
    if manual_list:
        chosen = []
        for target in manual_list:
            agg["score"] = agg["hotel_name"].apply(lambda x: fuzzy_score(x, target))
            pick = agg.sort_values("score", ascending=False).head(1)
            if not pick.empty:
                chosen.append(pick[["hotel_id", "hotel_name", "min_total"]])
        if chosen:
            comp = (
                pd.concat(chosen)
                .drop_duplicates(subset=["hotel_id"])
                .reset_index(drop=True)
            )
            return comp

    # Auto: buang kategori non-hotel (opsional)
    comp = agg.copy()
    if not include_nonhotel:
        comp = comp[comp["hotel_name"].str.contains("hotel", case=False, na=False)]
        for bad in ["hostel", "homestay", "guest house", "capsule", "apartment", "kost", "villa"]:
            comp = comp[~comp["hotel_name"].str.contains(bad, case=False, na=False)]

    if our_id and our_id in set(comp["hotel_id"]):
        our_price = float(comp.loc[comp["hotel_id"] == our_id, "min_total"].min())
        lo, hi = price_band
        comp = comp[(comp["min_total"] >= lo * our_price) & (comp["min_total"] <= hi * our_price)]

    if our_id:
        comp = comp[comp["hotel_id"] != our_id]

    return comp.sort_values("min_total").head(k).reset_index(drop=True)


def compute_metrics(df: pd.DataFrame, our_id: Optional[str], our_manual_total: Optional[float]) -> Tuple[pd.DataFrame, Optional[float]]:
    idx = df.groupby(["hotel_id"])["total"].idxmin()
    best = (
        df.loc[idx, ["hotel_id", "hotel_name", "vendor", "total"]]
        .rename(columns={"vendor": "min_vendor", "total": "min_total"})
    )

    our_total = None
    if our_id and our_id in set(best["hotel_id"]):
        our_total = float(best.loc[best["hotel_id"] == our_id, "min_total"].min())
    elif our_manual_total:
        our_total = float(our_manual_total)

    best["is_us"] = best["hotel_id"].eq(our_id)

    if our_total is not None:
        best["pct_above_us"] = (best["min_total"] - our_total) / max(our_total, 1e-9)
    else:
        best["pct_above_us"] = None

    # Sort: kita dulu, lalu kompetitor termurah → termahal
    return best.sort_values(["is_us", "min_total"], ascending=[False, True]).reset_index(drop=True), our_total


def price_index(our_total: float, comp_min_totals: List[float]) -> Optional[float]:
    if our_total is None or not comp_min_totals:
        return None
    avg_comp = sum(comp_min_totals) / len(comp_min_totals)
    if avg_comp <= 0:
        return None
    return 100.0 * (our_total / avg_comp)

# --------------------------- Geoapify helpers -------------------------------

def geo_geocode(name: str, city: Optional[str], api_key: str) -> Optional[Tuple[float, float]]:
    """Geocode nama hotel (opsional + kota). Return (lon, lat) atau None."""
    if not api_key:
        return None
    q = f"{name}, {city}" if city else name
    try:
        r = requests.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": q, "limit": 1, "apiKey": api_key},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json() or {}
        feats = js.get("features", [])
        if not feats:
            return None
        lon, lat = feats[0]["geometry"]["coordinates"]
        return float(lon), float(lat)
    except Exception:
        return None


def geo_places_nearby(center: Tuple[float, float], api_key: str, radius: int = DEFAULT_RADIUS_M, limit: int = 24) -> List[Dict[str, object]]:
    """Cari hotel terdekat dari titik center (lon, lat) dalam radius meter."""
    if not api_key:
        return []
    lon, lat = center
    try:
        r = requests.get(
            "https://api.geoapify.com/v2/places",
            params={
                "categories": "accommodation.hotel",
                "filter": f"circle:{lon},{lat},{radius}",
                "bias": f"proximity:{lon},{lat}",
                "limit": limit,
                "apiKey": api_key,
            },
            timeout=20,
        )
        r.raise_for_status()
        out: List[Dict[str, object]] = []
        for f in (r.json() or {}).get("features", []):
            lon2, lat2 = f["geometry"]["coordinates"]
            name = (f.get("properties") or {}).get("name") or "Hotel"
            out.append({"name": name, "lon": float(lon2), "lat": float(lat2)})
        return out
    except Exception:
        return []


def _dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Perkiraan jarak meter (equirectangular) — cukup akurat untuk < ~50 km."""
    kx = 111_320 * cos(radians((lat1 + lat2) / 2.0))
    ky = 110_540
    dx = (lon2 - lon1) * kx
    dy = (lat2 - lat1) * ky
    return (dx * dx + dy * dy) ** 0.5


def build_static_map_url(
    center: Tuple[float, float],
    markers: List[Dict[str, object]],
    api_key: str,
    width: int = 900,
    height: int = 520,
    zoom: float = 14.0,
) -> str:
    """Bangun URL Geoapify Static Map dengan markers (mendukung 'size')."""
    lon, lat = center

    def enc(m: Dict[str, object]) -> str:
        s = (
            f"lonlat:{m['lon']},{m['lat']};"
            f"type:{m.get('type','awesome')};"
            f"color:{m.get('color','#4c905a')};"
            f"icon:{m.get('icon','hotel')}"
        )
        if m.get("size"):
            s += f";size:{m['size']}"
        return quote(s, safe="")

    marker_param = "%7C".join(enc(m) for m in markers)
    return (
        "https://maps.geoapify.com/v1/staticmap?"
        f"style=osm-bright-smooth&width={width}&height={height}&center=lonlat:{lon},{lat}"
        f"&zoom={zoom}&marker={marker_param}&apiKey={api_key}"
    )

# --------------------------- SQLite helpers ---------------------------------

def connect_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {VENDOR_RATES_TABLE} (
          city TEXT,
          fetched_at DATETIME,
          hotel_name TEXT,
          hotel_id TEXT,
          vendor TEXT,
          price REAL,
          tax REAL,
          total REAL,
          PRIMARY KEY (city, fetched_at, hotel_id, vendor)
        );
        """
    )
    return con


def save_snapshot(con: sqlite3.Connection, city: str, fetched_at: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    recs = df[["hotel_name", "hotel_id", "vendor", "price", "tax", "total"]].to_dict("records")
    rows = [(city, fetched_at, r["hotel_name"], r["hotel_id"], r["vendor"], r["price"], r["tax"], r["total"]) for r in recs]
    cur = con.cursor()
    cur.executemany(
        f"INSERT OR REPLACE INTO {VENDOR_RATES_TABLE} (city, fetched_at, hotel_name, hotel_id, vendor, price, tax, total) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    return len(rows)


def load_latest_snapshot(con: sqlite3.Connection, city: str) -> Optional[pd.DataFrame]:
    q = f"""
        WITH mx AS (
          SELECT city, MAX(fetched_at) AS fetched_at FROM {VENDOR_RATES_TABLE} WHERE city=?
        )
        SELECT v.city, v.fetched_at, v.hotel_name, v.hotel_id, v.vendor, v.price, v.tax, v.total
        FROM {VENDOR_RATES_TABLE} v JOIN mx USING(city, fetched_at)
    """
    df = pd.read_sql_query(q, con, params=[city])
    return df if not df.empty else None

# ------------------------------ Panduan Singkat ------------------------------
GUIDE_TEXT = """
**Cara pakai (bahasa mudah):**

**Max competitors (auto mode)**
- Geser **kiri** = sedikit pesaing → fokus & cepat, tapi bisa ada yang kelewat.
- Geser **kanan** = banyak pesaing → gambaran lebih lengkap, tapi lebih ramai & agak lambat.
- Rekomendasi harian: **6–9**. Pitch cepat: **3–5**.

**Auto price band (vs our min)**
- Angka kiri–kanan = batas harga kompetitor dibanding **harga termurah kita**.
- Contoh **0.70–1.30** artinya ambil hotel yang harganya **70%–130%** dari harga kita.
- Sempit (mis. **0.9–1.1**) = mirip-mirip harga; Lebar (mis. **0.6–1.5**) = lebih banyak segmen.

**Parity alert threshold (%)**
- Batas selisih harga supaya muncul di **Parity Alerts**.
- Kecil (mis. **3–5%**) = sensitif (banyak alert kecil).
- Besar (mis. **8–12%**) = hanya selisih besar (cocok laporan eksekutif).
"""

# ------------------------------ UI ------------------------------------------
st.set_page_config(page_title="Rate Parity — Any Hotel vs Competitors", layout="wide")
st.title("🏨 Rate Parity Monitor — Any Hotel vs Competitors")
with st.expander("Panduan singkat (1 menit baca)"):
    st.markdown(GUIDE_TEXT)

with st.sidebar:
    st.header("Settings")
    our_name = st.text_input("Hotel name (required)", value=DEFAULT_HOTEL, help="Ketik nama hotel yang mau dibandingkan.")
    city = st.text_input("City (optional)", value=DEFAULT_CITY, help="Boleh kosong. Kalau kosong, app akan mencoba beberapa kota di 'Try cities'.")
    try_cities = st.text_input("Try cities (comma, used when City empty)", value=os.getenv("CITY_TRY", DEFAULT_CITY_TRY))

    # Auth & Map options (pakai demo keys bawaan; tetap bisa diganti via env var)
    st.caption("🔐 Makcorps & Geoapify menggunakan DEMO key yang sudah ditanam.")
    jwt = MAKCORPS_JWT_DEMO
    geo_key = GEOAPIFY_KEY_DEMO

    show_map = st.checkbox("Tampilkan peta statis (Geoapify)", value=True)
    radius_m = st.slider("Radius hotel terdekat (meter)", 5_000, 40_000, DEFAULT_RADIUS_M, 500, help="Seberapa jauh peta mencari hotel sekitar dari hotel kamu.")
    max_markers = st.slider("Maksimal marker di peta", 3, 24, 15, help="Batas jumlah pin kompetitor yang ditampilkan di peta.")
    include_nonhotel = st.checkbox("Sertakan non-hotel (homestay/guest house, dll.)", value=False)

# Manual override (opsional). Kosongkan jika ingin auto-competitor.
PLACEHOLDER_COMP = "Hotel Neo Malioboro\nIbis Styles Yogyakarta"
manual_comp = st.text_area(
    "Daftar kompetitor (opsional) - satu per baris",
    value="",
    placeholder=PLACEHOLDER_COMP,
    help="Jika kosong, sistem akan mencari otomatis kompetitor terdekat via Geoapify."
).strip().splitlines()

comp_k = st.slider(
    "Max competitors (auto mode)",
    min_value=3, max_value=12, value=8,
    help="Kiri = lebih sedikit & fokus. Kanan = lebih banyak & lengkap."
)
band_low, band_high = st.slider(
    "Auto price band (vs our min)",
    0.5, 1.8, (0.7, 1.3), 0.05,
    help="Batas harga kompetitor dibanding harga termurah kita. 0.70–1.30 = ambil hotel dengan harga 70%–130% dari kita."
)
parity_threshold = st.slider(
    "Parity alert threshold (%)",
    min_value=1, max_value=20, value=5,
    help="Batas selisih harga untuk muncul di 'Parity Alerts'."
)

st.divider()
persist = st.checkbox("Save snapshot to SQLite (for trends)", value=True, help="Simpan data agar bisa lihat tren siapa vendor yang paling sering termurah.")
db_path = st.text_input("SQLite file", value=DB_DEFAULT)

col_ai = st.columns([1, 1, 2])
with col_ai[0]:
    want_ai = st.checkbox("Generate AI insights (Gemini)", value=True)
with col_ai[1]:
    auto_ai = st.checkbox("Auto after fetch", value=True)

if st.button("Fetch Now", type="primary"):
    st.session_state["do_fetch"] = True
    st.session_state["do_ai"] = want_ai and auto_ai

# --------------------------- Fetch & Logic ----------------------------------

def fetch_for_hotelname(our_name: str, city: str, try_cities_csv: str, jwt: Optional[str]):
    """Return: (df_raw, chosen_city, fetched_at, warn_msg, our_id)"""
    warn_all: List[str] = []
    if city.strip():
        df_raw, fetched_at, warn = fetch_makcorps_city(city.strip(), jwt)
        if warn:
            warn_all.append(warn)
        our_id = find_hotel_id(df_raw, our_name)
        return df_raw, city.strip(), fetched_at, ("\n".join(warn_all) if warn_all else None), our_id

    # City kosong → coba beberapa kota
    chosen_city: Optional[str] = None
    chosen_df: Optional[pd.DataFrame] = None
    chosen_our_id: Optional[str] = None
    last_ts: Optional[str] = None

    for c in [x.strip() for x in try_cities_csv.split(",") if x.strip()] or ["yogyakarta"]:
        df_raw, fetched_at, warn = fetch_makcorps_city(c, jwt)
        if warn:
            warn_all.append(f"{c}: {warn}")
        our_id = find_hotel_id(df_raw, our_name)
        last_ts = fetched_at
        if our_id:
            chosen_city, chosen_df, chosen_our_id = c, df_raw, our_id
            break
        if chosen_df is None:
            chosen_city, chosen_df = c, df_raw

    return (
        chosen_df,
        chosen_city,
        last_ts or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ("\n".join(warn_all) if warn_all else None),
        chosen_our_id,
    )

# ------------------------------ Gemini helper --------------------------------

def gemini_insights(summary_df: pd.DataFrame,
                    our_total: Optional[float],
                    price_index_val: Optional[float],
                    parity_df: Optional[pd.DataFrame],
                    city: Optional[str],
                    our_name: str,
                    price_band: Tuple[float, float],
                    comp_k: int) -> Optional[str]:
    """Panggil Gemini 2.5 Flash untuk merangkum & memberi rekomendasi singkat (ID)."""
    if not GEMINI_API_KEY:
        return None

    # Siapkan ringkasan data untuk prompt
    top_rows = summary_df.copy()
    cols = ["hotel_name", "min_vendor", "min_total", "pct_above_us"]
    if "pct_above_us" not in top_rows.columns:
        top_rows["pct_above_us"] = None
    top_rows = top_rows[cols].head(20)

    table_txt = "\n".join(
        f"- {r.hotel_name} | {r.min_vendor} | {r.min_total} | pct_vs_us={round(r.pct_above_us*100,2) if pd.notnull(r.pct_above_us) else 'n/a'}%"
        for r in top_rows.itertuples(index=False)
    )

    parity_txt = ""
    if parity_df is not None and not parity_df.empty:
        parity_txt = "\n".join(
            f"  * ALERT: {r.hotel_name} — {r.min_vendor} — total={r.min_total} — gap_vs_us%={round(r['gap_vs_us']*100,2)}"
            for _, r in parity_df.iterrows()
        )

    prompt = f"""
Anda adalah analis revenue management hotel. Jelaskan singkat dalam bahasa Indonesia yang mudah dipahami.

Hotel kita: "{our_name}" di kota "{city or '-'}".
Harga termurah kita saat ini: {our_total if our_total is not None else 'n/a'}.
Price Index (vs rata-rata kompetitor = 100): {round(price_index_val,1) if price_index_val else 'n/a'}.
Auto competitor setting: max={comp_k}, price_band={price_band[0]}–{price_band[1]}.

Tabel ringkas (Hotel | Vendor Termurah | Total | % vs kita):
{table_txt}

Parity Alerts (jika ada):
{parity_txt or '(tidak ada alert di atas ambang yang ditetapkan)'}

Tulis:
1) 4–6 poin insight inti (tren harga, siapa paling agresif, posisi kita vs pasar).
2) 3–5 rekomendasi aksi praktis (mis. sesuaikan harga, bundling, channel khusus).
3) Satu kalimat ringkas 'takeaway' eksekutif.
Jangan gunakan istilah teknis berat; langsung to the point.
"""

    try:
        # Prefer SDK baru
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text
        except Exception:
            pass

        # Fallback ke SDK lama
        import google.generativeai as genai_old
        genai_old.configure(api_key=GEMINI_API_KEY)
        model = genai_old.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text
    except Exception as e:
        st.info(f"(AI insight non-aktif: {e})")
        return None

# ------------------------------ Run -----------------------------------------

if st.session_state.get("do_fetch"):
    if not our_name.strip():
        st.error("Please enter a hotel name.")
        st.stop()

    with st.spinner("Fetching rates & building comp-set…"):
        df_raw, chosen_city, fetched_at, warn, our_id = fetch_for_hotelname(our_name, city, try_cities, jwt)

    # UI tetap bersih (tanpa warn teknis)
    st.caption(f"City={chosen_city or 'unknown'} • Rows={len(df_raw)} • Fetched at {fetched_at}")

    # Simpan snapshot
    if persist and db_path and chosen_city:
        try:
            con = connect_db(db_path)
            n = save_snapshot(con, chosen_city, fetched_at, df_raw)
            st.success(f"Saved {n} rows to {db_path}")
        except Exception as e:
            st.warning(f"Failed to save snapshot: {e}")

    # Geocode pusat hotel (untuk peta & auto-competitor)
    center: Optional[Tuple[float, float]] = None
    auto_comp_names: List[str] = []
    places: List[Dict[str, object]] = []
    if show_map and GEOAPIFY_KEY_DEMO:
        center = geo_geocode(our_name, chosen_city or city, GEOAPIFY_KEY_DEMO)
        if center:
            # Ambil hotel terdekat dalam radius (fallback kuat bila nama tidak match)
            places = geo_places_nearby(center, GEOAPIFY_KEY_DEMO, radius=radius_m, limit=max_markers + 8)
            # Buat daftar nama kompetitor resmi (hindari nama yang sama dengan kita)
            core_tokens = [t for t in our_name.lower().split() if len(t) >= 4]
            for p in places:
                nm = str(p.get("name") or "").strip()
                low = nm.lower()
                too_similar = fuzzy_score(nm, our_name) >= 0.82
                contains_core = core_tokens and all(tok in low for tok in core_tokens[:2])
                if nm and not too_similar and not contains_core:
                    auto_comp_names.append(nm)
            auto_comp_names = list(dict.fromkeys(auto_comp_names))[:max_markers]

    # Jika hotel kita tidak ketemu di data kota → boleh input manual rate
    our_manual_total = None
    if not our_id:
        our_manual_total = st.number_input("Manual 'our hotel' total rate (price+tax)", min_value=0.0, value=26.0, step=0.5)

    # Gunakan manual_comp jika diisi; jika kosong → pakai auto_comp_names
    chosen_list = [x for x in manual_comp if x.strip()] or auto_comp_names

    with st.expander("Daftar kompetitor yang digunakan (auto/manual)"):
        st.write(chosen_list or ["(tidak ada)"])

    # Comp-set untuk tabel
    comp_df = select_competitors(
        df_raw, our_id,
        manual_list=chosen_list,
        k=comp_k, price_band=(band_low, band_high),
        include_nonhotel=include_nonhotel
    )

    work_ids = set(comp_df["hotel_id"]) if not comp_df.empty else set()
    if our_id:
        work_ids.add(our_id)
    subset = df_raw[df_raw["hotel_id"].isin(work_ids)].copy()

    if subset.empty and our_manual_total is None:
        st.error("No data to display. Tambah kompetitor manual atau coba kota lain.")
        st.stop()

    summary, our_total = compute_metrics(subset, our_id, our_manual_total)

    comp_only = summary[~summary["is_us"]]["min_total"].tolist()
    pi = price_index(our_total, comp_only)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Summary — Cheapest vendor per hotel (current snapshot)")
        fmt = summary.copy()
        if our_total is not None:
            fmt["pct_above_us(%)"] = (fmt["pct_above_us"] * 100).round(2)
        fmt = fmt.rename(columns={"hotel_name": "Hotel", "min_vendor": "Cheapest Vendor", "min_total": "Cheapest Total"})
        st.dataframe(
            fmt[["Hotel", "Cheapest Vendor", "Cheapest Total"] + (["pct_above_us(%)"] if our_total is not None else [])],
            use_container_width=True,
        )

    with right:
        st.subheader("KPI")
        if our_total is not None:
            st.metric("Our Cheapest Total", f"{our_total:.2f}")
        if pi is not None:
            st.metric("Price Index (vs avg comp-set=100)", f"{pi:.1f}")
        st.metric("Competitors", str(len(comp_only)))

    # -------- Geoapify Static Map (pusat + kompetitor) --------
    if show_map and GEOAPIFY_KEY_DEMO:
        st.subheader("Peta hotel sekitar (Geoapify)")
        try:
            if center:
                # pin pusat (hotel kita) — magenta, bintang, size besar
                markers: List[Dict[str, object]] = [{
                    "lon": center[0], "lat": center[1],
                    "type": "awesome", "color": "#bb3f73", "icon": "star", "size": "x-large"
                }]

                # kompetitor dari Places: skip yang terlalu dekat (< 80 m) agar tidak menimpa pin pusat
                added = 0
                for p in places:
                    nm = str(p.get("name") or "")
                    if not nm or fuzzy_score(nm, our_name) >= 0.82:
                        continue
                    if _dist_m(center[0], center[1], p["lon"], p["lat"]) < 80:
                        continue
                    markers.append({
                        "lon": p["lon"], "lat": p["lat"],
                        "type": "material", "color": "#4c905a", "icon": "hotel"
                    })
                    added += 1
                    if added >= max_markers:
                        break

                # Jika belum cukup, coba geocode nama kompetitor yang dipakai di tabel
                if added < max_markers and chosen_list:
                    for nm in chosen_list:
                        co = geo_geocode(nm, chosen_city or city, GEOAPIFY_KEY_DEMO)
                        if not co:
                            continue
                        if _dist_m(center[0], center[1], co[0], co[1]) < 80:
                            continue
                        markers.append({
                            "lon": co[0], "lat": co[1],
                            "type": "material", "color": "#4c905a", "icon": "hotel"
                        })
                        added += 1
                        if added >= max_markers:
                            break

                map_url = build_static_map_url(center, markers, GEOAPIFY_KEY_DEMO, width=900, height=520, zoom=14.0)
                st.image(map_url, caption="Legenda: ⭐ magenta = hotel kita • 🏨 hijau = kompetitor")
            else:
                st.info("Koordinat hotel tidak ditemukan untuk peta (geocoder tidak menemukan nama).")
        except Exception as e:
            st.warning(f"Gagal memuat peta Geoapify: {e}")

    # -------- Parity Alerts --------
    thr = parity_threshold / 100.0
    parity_hot = None
    if our_total is not None:
        st.subheader("Parity Alerts")
        alerts = summary[~summary["is_us"]].copy()
        alerts["gap_vs_us"] = (alerts["min_total"] - our_total) / max(our_total, 1e-9)
        parity_hot = alerts[alerts["gap_vs_us"].abs() > thr].copy()
        if parity_hot.empty:
            st.success("No parity alerts above threshold.")
        else:
            parity_hot = parity_hot.sort_values("gap_vs_us")
            parity_hot["gap_vs_us(%)"] = (parity_hot["gap_vs_us"] * 100).round(2)
            st.dataframe(
                parity_hot[["hotel_name", "min_vendor", "min_total", "gap_vs_us(%)"]]
                .rename(columns={"hotel_name": "Hotel", "min_vendor": "Cheapest Vendor", "min_total": "Cheapest Total"}),
                use_container_width=True,
            )

    # -------- Tren sederhana dari snapshot --------
    if persist and db_path and chosen_city:
        try:
            con = connect_db(db_path)
            latest_df = load_latest_snapshot(con, chosen_city)
            if latest_df is not None:
                st.subheader("Cheapest Vendor Share (last snapshots)")
                q = f"""
                    WITH ranked AS (
                      SELECT city, fetched_at, hotel_id, vendor, total,
                             RANK() OVER(PARTITION BY city, fetched_at, hotel_id ORDER BY total ASC) AS rnk
                      FROM {VENDOR_RATES_TABLE}
                      WHERE city=?
                    )
                    SELECT vendor, COUNT(*) AS times_cheapest
                    FROM ranked
                    WHERE rnk=1
                    GROUP BY vendor
                    ORDER BY times_cheapest DESC
                """
                vend = pd.read_sql_query(q, con, params=[chosen_city])
                if not vend.empty:
                    vend = vend.set_index("vendor")
                    st.bar_chart(vend)
                else:
                    st.info("Not enough snapshots yet to build a trend.")
        except Exception as e:
            st.warning(f"Trend query failed: {e}")

    # -------- Gemini Insights --------
    if want_ai and (st.session_state.get("do_ai") or st.button("Generate AI Insights now")):
        with st.spinner("Menulis insight singkat dengan Gemini…"):
            insight = gemini_insights(
                summary_df=summary,
                our_total=our_total,
                price_index_val=pi,
                parity_df=parity_hot,
                city=chosen_city,
                our_name=our_name,
                price_band=(band_low, band_high),
                comp_k=comp_k,
            )
        st.subheader("AI Insights (Gemini 2.5 Flash)")
        if insight:
            st.markdown(insight)
        else:
            st.info("Gemini belum dapat digunakan (library belum terpasang atau API error).")
    st.caption("Note: Free API uses random future date & max 30 hotels/city. City auto-try loops over your list until the hotel is found (fuzzy match).")

else:
    st.info("Isi nama hotel di sidebar (city opsional), lalu klik **Fetch Now**. Peta & data harga memakai DEMO keys yang sudah ditanam (bisa diganti ke env var untuk produksi).")
