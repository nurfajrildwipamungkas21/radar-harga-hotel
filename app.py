# --------------------------- Geoapify helpers -------------------------------
def geo_geocode(name: str, city: Optional[str], api_key: str) -> Optional[tuple[float, float]]:
    """Geocode a hotel name (optionally with city). Returns (lon, lat) or None."""
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

def build_static_map_url(center: tuple[float, float], markers: list[dict], api_key: str,
                         width: int = 800, height: int = 480, zoom: float = 14.0) -> str:
    lon, lat = center
    def enc(m: dict) -> str:
        s = (
            f"lonlat:{m['lon']},{m['lat']};"
            f"type:{m.get('type','awesome')};"
            f"color:{m.get('color','#4c905a')};"
            f"icon:{m.get('icon','hotel')}"
        )
        return quote(s, safe='')
    marker_param = "%7C".join(enc(m) for m in markers)
    return (
        "https://maps.geoapify.com/v1/staticmap?"
        f"style=osm-bright-smooth&width={width}&height={height}&center=lonlat:{lon},{lat}"
        f"&zoom={zoom}&marker={marker_param}&apiKey={api_key}"
    )
