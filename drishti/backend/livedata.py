"""Live public data feeds. All free, none requiring an API key or payment.

  USGS FDSN earthquake catalogue  - real seismic events, epicentre + magnitude
  Open-Meteo forecast API         - real rainfall over the district
  Open-Meteo elevation API        - real terrain (used offline via cache)

Every call is cached and wrapped: if the venue wifi dies mid-demo the
dashboard keeps working on the last good snapshot, and says so. A disaster
tool that falls over when the network does would be self-defeating.
"""
import json, os, threading, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(DATA, "livecache.json")
TTL = 900          # seconds; these feeds do not move faster than this
UA = {"User-Agent": "DRISHTI-Avinya2026/1.0 (disaster response research)"}

_lock = threading.Lock()
_mem = {}


def _get(url, timeout=12):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cached(key, fn):
    now = time.time()
    with _lock:
        hit = _mem.get(key)
        if hit and now - hit["at"] < TTL:
            return hit["value"], "live-cache"
    try:
        val = fn()
        with _lock:
            _mem[key] = {"at": now, "value": val}
            _persist()
        return val, "live"
    except Exception as e:
        disk = _load_disk().get(key)
        if disk:
            return disk["value"], "stale-cache (%s)" % type(e).__name__
        return None, "unavailable (%s)" % type(e).__name__


def _load_disk():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _persist():
    try:
        os.makedirs(DATA, exist_ok=True)
        json.dump(_mem, open(CACHE_PATH, "w"), default=str)
    except Exception:
        pass


# ------------------------------------------------------------------ feeds
def usgs_quakes(bbox=(26.0, 72.0, 37.0, 90.0), days=60, minmag=4.0):
    """Real earthquakes near the district. This is exactly the feed that would
    seed the engine's shake prior in a live event."""
    s, w, n, e = bbox
    q = urllib.parse.urlencode({
        "format": "geojson", "starttime": time.strftime(
            "%Y-%m-%d", time.gmtime(time.time() - days * 86400)),
        "minlatitude": s, "maxlatitude": n,
        "minlongitude": w, "maxlongitude": e,
        "minmagnitude": minmag, "orderby": "time",
    })
    j = _get("https://earthquake.usgs.gov/fdsnws/event/1/query?" + q)
    return [{
        "id": f["id"], "mag": f["properties"]["mag"],
        "place": f["properties"]["place"],
        "time": f["properties"]["time"],
        "lon": f["geometry"]["coordinates"][0],
        "lat": f["geometry"]["coordinates"][1],
        "depth_km": f["geometry"]["coordinates"][2],
    } for f in j.get("features", [])]


def rainfall(lat=30.45, lon=79.15):
    """Real rainfall + river-relevant weather over the district centroid."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation,rain",
        "daily": "precipitation_sum",
        "past_days": 3, "forecast_days": 2, "timezone": "Asia/Kolkata",
    })
    j = _get("https://api.open-meteo.com/v1/forecast?" + q)
    hr = j.get("hourly", {})
    prec = [p for p in (hr.get("precipitation") or []) if p is not None]
    daily = j.get("daily", {})
    return {
        "hourly_time": (hr.get("time") or [])[-48:],
        "hourly_precip_mm": prec[-48:],
        "total_72h_mm": round(sum(prec[-72:]), 1) if prec else 0.0,
        "peak_hourly_mm": round(max(prec), 1) if prec else 0.0,
        "daily_time": daily.get("time"),
        "daily_precip_mm": daily.get("precipitation_sum"),
    }


def rain_factor(lat=30.45, lon=79.15):
    """Map observed rainfall onto the engine's [0,1] rain driver.

    50 mm over 72 h is an ordinary monsoon spell here; 300 mm is the kind of
    total that preceded the 2013 Kedarnath disaster.
    """
    r, status = _cached("rain", lambda: rainfall(lat, lon))
    if not r:
        return 1.0, "default (no live feed)"
    mm = r.get("total_72h_mm", 0.0)
    f = max(0.35, min(1.6, 0.35 + 1.25 * (mm / 300.0)))
    return round(f, 3), status


def snapshot():
    quakes, qs = _cached("usgs", usgs_quakes)
    rain, rs = _cached("rain", rainfall)
    f, _ = rain_factor()
    return {
        "usgs": {"status": qs, "count": len(quakes or []),
                 "events": (quakes or [])[:12]},
        "rainfall": {"status": rs, **(rain or {})},
        "derived_rain_factor": f,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "note": ("All feeds are free and key-less. Cached for %ds; the "
                 "dashboard stays usable offline on the last snapshot." % TTL),
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _mem.update(_load_disk())
    s = snapshot()
    print("USGS      :", s["usgs"]["status"], "-", s["usgs"]["count"], "events")
    for e in s["usgs"]["events"][:5]:
        print("   M%.1f  %s" % (e["mag"], e["place"]))
    print("Rainfall  :", s["rainfall"]["status"],
          "- 72h total %.1f mm, peak %.1f mm/h"
          % (s["rainfall"].get("total_72h_mm", 0),
             s["rainfall"].get("peak_hourly_mm", 0)))
    print("rain_factor fed to engine:", s["derived_rain_factor"])
