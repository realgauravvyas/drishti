"""Fetch a real gazetteer (settlements + rivers + roads) from OpenStreetMap via Overpass.
Free, no API key. Run once; output is cached to data/gazetteer.json
"""
import json, urllib.request, urllib.parse, os, sys, time

# Upper Alaknanda / Mandakini basin - Rudraprayag + Chamoli + Tehri, Uttarakhand.
# Real multi-hazard terrain: seismic zone V, landslide-prone, flash-flood history (2013 Kedarnath).
BBOX = (30.05, 78.75, 30.85, 79.65)   # south, west, north, east
OUT = os.path.join(os.path.dirname(__file__), "data", "gazetteer.json")

Q_PLACES = f"""
[out:json][timeout:120];
(
  node["place"~"^(city|town|village|hamlet)$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out body;
"""

Q_WATER = f"""
[out:json][timeout:120];
(
  way["waterway"="river"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;
"""

Q_ROADS = f"""
[out:json][timeout:90];
(
  way["highway"~"^(trunk|primary|secondary)$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom qt;
"""

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

def overpass(q, label):
    """Fetch with on-disk cache so a rate-limit on one query never loses the others."""
    cache = os.path.join(os.path.dirname(__file__), "data", f"_raw_{label}.json")
    if os.path.exists(cache):
        j = json.load(open(cache, encoding="utf-8"))
        print(f"  [{label}] {len(j.get('elements',[]))} elements from cache")
        return j
    last = None
    for ep in ENDPOINTS:
        for attempt in range(2):
            try:
                data = urllib.parse.urlencode({"data": q}).encode()
                req = urllib.request.Request(ep, data=data, headers={
                    "User-Agent": "DRISHTI-Avinya2026/1.0 (hackathon research)"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    j = json.loads(r.read().decode())
                print(f"  [{label}] {len(j.get('elements',[]))} elements from {ep}")
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                json.dump(j, open(cache, "w", encoding="utf-8"))
                return j
            except Exception as e:
                last = e
                print(f"  [{label}] attempt failed on {ep}: {e}")
                time.sleep(20)
    raise SystemExit(f"Overpass failed for {label}: {last}")

def main():
    print("Fetching settlements...")
    places = overpass(Q_PLACES, "places")
    print("Fetching rivers...")
    water = overpass(Q_WATER, "rivers")
    print("Fetching roads (optional)...")
    try:
        roads = overpass(Q_ROADS, "roads")
    except SystemExit as e:
        print("  roads unavailable (%s) - continuing without; "
              "route graph falls back to terrain-weighted kNN" % e)
        roads = {"elements": []}

    settlements = []
    for el in places["elements"]:
        t = el.get("tags", {})
        name = t.get("name:en") or t.get("name")
        if not name:
            continue
        settlements.append({
            "osm_id": el["id"],
            "name": name,
            "name_hi": t.get("name:hi"),
            "place": t.get("place"),
            "lat": el["lat"], "lon": el["lon"],
            "population": int(t["population"]) if str(t.get("population","")).isdigit() else None,
        })

    rivers = []
    for el in water["elements"]:
        g = el.get("geometry") or []
        if len(g) < 2: continue
        rivers.append({"name": el.get("tags",{}).get("name"),
                       "coords": [[p["lat"], p["lon"]] for p in g]})

    rd = []
    for el in roads["elements"]:
        g = el.get("geometry") or []
        if len(g) < 2: continue
        rd.append({"name": el.get("tags",{}).get("name"),
                   "cls": el.get("tags",{}).get("highway"),
                   "bridge": el.get("tags",{}).get("bridge") is not None,
                   "coords": [[p["lat"], p["lon"]] for p in g]})

    out = {"bbox": BBOX, "settlements": settlements, "rivers": rivers, "roads": rd}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nSaved {len(settlements)} settlements, {len(rivers)} river ways, {len(rd)} roads -> {OUT}")

if __name__ == "__main__":
    main()
