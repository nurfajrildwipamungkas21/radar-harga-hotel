# app.py
# =============================================================================
# Any Hotel → Auto Competitor — Rate Parity Monitor (Makcorps Free API)
# Streamlit single-file app + optional SQLite
# =============================================================================
# What this does
# - You type ANY hotel name. City is optional.
# - If City is given → fetch that city once.
# - If City empty → the app tries multiple candidate cities (comma list) until the
#   hotel is found via fuzzy match (Makcorps Free requires a city).
# - Build a comp-set (auto or manual list) and compute parity metrics.
# - Optional: save snapshots into SQLite and show vendor-cheapest trend.
#
# Limitations (Makcorps Free): random future date, max 30 hotels per city, no pax/room type.
# =============================================================================

import os
import json
import time
import sqlite3
import difflib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st

# ----------------------------- Config ---------------------------------------
DEFAULT_CITY = ""  # now optional
DEFAULT_HOTEL = "Satya Graha Hotel"
DB_DEFAULT = "hotel_competitor.sqlite"
VENDOR_RATES_TABLE = "vendor_rates"
DEFAULT_CITY_TRY = "yogyakarta, sleman, bantul, depok, kasihan, jakarta, bandung, surabaya, denpasar"

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
        {"hotelName": "Hotel A Malioboro", "hotelId": "a1"},
        [
            {"price1": "24", "tax1": "3", "vendor1": "Booking.com"},
            {"price2": "26", "tax2": "0", "vendor2": "Priceline"},
            {"price3": "25", "tax3": "2", "vendor3": "Hotels.com"},
        ],
    ],
    [
        {"hotelName": "Hotel B Prawirotaman", "hotelId": "b2"},
        [
            {"price1": "28", "tax1": "3", "vendor1": "Booking.com"},
            {"price2": "27", "tax2": "3", "vendor2": "Trip.com"},
        ],
    ],
    [
        {"hotelName": "C Homestay", "hotelId": "c3"},
        [
            {"price1": "18", "tax1": "2", "vendor1": "Booking.com"}
        ],
    ],
]

# --------------------------- Helper functions -------------------------------

def fuzzy_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def flatten_makcorps(raw: List) -> pd.DataFrame:
    """Turn Makcorps free API response into a tidy DataFrame.
    Columns: hotel_name, hotel_id, vendor, price, tax, total
    """
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
    """Return DataFrame + fetched_at timestamp + error.
    If token missing or fetch fails, fall back to SAMPLE_DATA for demo.
    """
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


def select_competitors(df: pd.DataFrame, our_id: Optional[str], manual_list: List[str], k: int = 8,
                       price_band: Tuple[float, float] = (0.7, 1.3)) -> pd.DataFrame:
    agg = df.groupby(["hotel_id", "hotel_name"], as_index=False)["total"].min().rename(columns={"total": "min_total"})

    if manual_list:
        chosen = []
        for target in manual_list:
            agg["score"] = agg["hotel_name"].apply(lambda x: fuzzy_score(x, target))
            pick = agg.sort_values("score", ascending=False).head(1)
            if not pick.empty:
                chosen.append(pick[["hotel_id", "hotel_name", "min_total"]])
        if chosen:
            comp = pd.concat(chosen).drop_duplicates(subset=["hotel_id"]).reset_index(drop=True)
            return comp

    comp = agg.copy()
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
    idx = df.groupby(["hotel_id"])['total'].idxmin()
    best = df.loc[idx, ["hotel_id", "hotel_name", "vendor", "total"]].rename(columns={"vendor": "min_vendor", "total": "min_total"})

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

    return best.sort_values(["is_us", "min_total"], ascending=[False, True]).reset_index(drop=True), our_total


def price_index(our_total: float, comp_min_totals: List[float]) -> Optional[float]:
    if our_total is None or not comp_min_totals:
        return None
    avg_comp = sum(comp_min_totals) / len(comp_min_totals)
    if avg_comp <= 0:
        return None
    return 100.0 * (our_total / avg_comp)


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


# ------------------------------ UI ------------------------------------------
st.set_page_config(page_title="Rate Parity — Any Hotel vs Competitors", layout="wide")
st.title("🏨 Rate Parity Monitor — Any Hotel vs Competitors")

with st.sidebar:
    st.header("Settings")
    our_name = st.text_input("Hotel name (required)", value=DEFAULT_HOTEL, help="Ketik nama hotel yang mau dibandingkan.")
    city = st.text_input("City (optional)", value=DEFAULT_CITY, help="Kosongkan untuk auto-try multiple cities.")
    try_cities = st.text_input("Try cities (comma, used when City empty)", value=os.getenv("CITY_TRY", DEFAULT_CITY_TRY))

    jwt = st.text_input("Makcorps JWT", type="password", help="Kosongkan untuk demo (pakai sample data).")

    manual_comp = st.text_area(
        "Daftar kompetitor (opsional) - satu per baris",
        value="",
        placeholder="Hotel A Malioboro"
Hotel B Prawirotaman",
    ).strip().splitlines()

    comp_k = st.slider("Max competitors (auto mode)", min_value=3, max_value=12, value=8)
    band_low, band_high = st.slider("Auto price band (vs our min)", 0.5, 1.8, (0.7, 1.3), 0.05)
    parity_threshold = st.slider("Parity alert threshold (%)", min_value=1, max_value=20, value=5)

    st.divider()
    persist = st.checkbox("Save snapshot to SQLite (for trends)", value=True)
    db_path = st.text_input("SQLite file", value=DB_DEFAULT)
    if st.button("Fetch Now", type="primary"):
        st.session_state["do_fetch"] = True

# --------------------------- Fetch & Logic ----------------------------------
def fetch_for_hotelname(our_name: str, city: str, try_cities_csv: str, jwt: Optional[str]):
    """Return tuple: (df_raw, chosen_city, fetched_at, warn_msg, our_id)
    If city provided: fetch once. If empty: iterate try_cities until hotel found.
    """
    warn_all = []
    if city.strip():
        df_raw, fetched_at, warn = fetch_makcorps_city(city.strip(), jwt)
        if warn:
            warn_all.append(warn)
        our_id = find_hotel_id(df_raw, our_name)
        return df_raw, city.strip(), fetched_at, "
".join(warn_all) if warn_all else None, our_id

    # City empty → try multiple
    chosen_city = None
    chosen_df = None
    chosen_our_id = None
    last_ts = None

    for c in [x.strip() for x in try_cities_csv.split(',') if x.strip()] or ["yogyakarta"]:
        df_raw, fetched_at, warn = fetch_makcorps_city(c, jwt)
        if warn:
            warn_all.append(f"{c}: {warn}")
        our_id = find_hotel_id(df_raw, our_name)
        last_ts = fetched_at
        if our_id:
            chosen_city, chosen_df, chosen_our_id = c, df_raw, our_id
            break
        # keep the first df as fallback if none matches
        if chosen_df is None:
            chosen_city, chosen_df = c, df_raw

    return chosen_df, chosen_city, last_ts or datetime.now(timezone.utc).isoformat(timespec='seconds'), "
".join(warn_all) if warn_all else None, chosen_our_id


if st.session_state.get("do_fetch"):
    if not our_name.strip():
        st.error("Please enter a hotel name.")
        st.stop()

    with st.spinner("Fetching Makcorps…"):
        df_raw, chosen_city, fetched_at, warn, our_id = fetch_for_hotelname(our_name, city, try_cities, jwt)

    if warn:
        st.info(warn)

    st.caption(f"City={chosen_city or 'unknown'} • Rows={len(df_raw)} • Fetched at {fetched_at}")

    # Save snapshot
    if persist and db_path and chosen_city:
        try:
            con = connect_db(db_path)
            n = save_snapshot(con, chosen_city, fetched_at, df_raw)
            st.success(f"Saved {n} rows to {db_path}")
        except Exception as e:
            st.warning(f"Failed to save snapshot: {e}")

    # Manual rate input if our hotel not found in chosen city
    our_manual_total = None
    if not our_id:
        st.warning("Hotel name not found in fetched city results. You can still compare using manual comp-set and manual rate.")
        our_manual_total = st.number_input("Manual 'our hotel' total rate (price+tax)", min_value=0.0, value=26.0, step=0.5)

    # Competitors
    comp_df = select_competitors(df_raw, our_id, [x for x in manual_comp if x.strip()], k=comp_k, price_band=(band_low, band_high))

    work_ids = set(comp_df["hotel_id"]) if not comp_df.empty else set()
    if our_id:
        work_ids.add(our_id)
    subset = df_raw[df_raw["hotel_id"].isin(work_ids)].copy()

    if subset.empty and our_manual_total is None:
        st.error("No data to display. Add manual competitors or try another city.")
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
        st.dataframe(fmt[["Hotel", "Cheapest Vendor", "Cheapest Total"] + (["pct_above_us(%)"] if our_total is not None else [])], use_container_width=True)

    with right:
        st.subheader("KPI")
        if our_total is not None:
            st.metric("Our Cheapest Total", f"{our_total:.2f}")
        if pi is not None:
            st.metric("Price Index (vs avg comp-set=100)", f"{pi:.1f}")
        st.metric("Competitors", str(len(comp_only)))

    if our_total is not None:
        st.subheader("Parity Alerts")
        thr = parity_threshold / 100.0
        alerts = summary[~summary["is_us"]].copy()
        alerts["gap_vs_us"] = (alerts["min_total"] - our_total) / max(our_total, 1e-9)
        hot = alerts[alerts["gap_vs_us"].abs() > thr].copy()
        if hot.empty:
            st.success("No parity alerts above threshold.")
        else:
            hot = hot.sort_values("gap_vs_us")
            hot["gap_vs_us(%)"] = (hot["gap_vs_us"] * 100).round(2)
            st.dataframe(hot[["hotel_name", "min_vendor", "min_total", "gap_vs_us(%)"]].rename(columns={"hotel_name": "Hotel", "min_vendor": "Cheapest Vendor", "min_total": "Cheapest Total"}), use_container_width=True)

    # Trend (if snapshots)
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

    st.caption("Note: Free API uses random future date & max 30 hotels/city. City auto-try loops over your list until the hotel is found (fuzzy match).")
else:
    st.info("Isi nama hotel di sidebar (city opsional), lalu klik **Fetch Now** untuk mulai.")
