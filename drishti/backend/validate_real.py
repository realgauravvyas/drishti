"""Real-world correctness checks.

Everything else in this project is measured against a simulator. This file
checks the parts that must agree with the REAL world:

  1. the gazetteer really is Rudraprayag/Chamoli, with correct coordinates
  2. terrain elevations match published values for known places
  3. hazard exposure is physically sensible (riverside low ground floods,
     high ridges do not)
  4. seismic attenuation reproduces published MMI for a real earthquake
  5. the live USGS and Open-Meteo feeds parse and are in plausible ranges
  6. fuzzy place resolution survives real misspellings of real villages

Run:  python validate_real.py
"""
import json, math, os, sys

from gazetteer import get_gazetteer, haversine_km

PASS, FAIL, WARN = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print("  %s %-52s %s" % ("PASS" if cond else "FAIL", name, detail))


def warn(name, detail=""):
    WARN.append((name, detail))
    print("  WARN %-52s %s" % (name, detail))


# Published coordinates and elevations (Survey of India / OSM / public record).
# Elevation tolerance is generous: OSM nodes mark a village centroid, and
# terrain in these valleys changes fast over a few hundred metres.
KNOWN = [
    # name,        lat,      lon,     elevation_m, tol_m
    ("Kedarnath",  30.7346, 79.0669, 3583, 700),
    ("Rudraprayag", 30.2844, 78.9811, 610, 400),
    ("Gaurikund",  30.6533, 79.0233, 1990, 700),
    ("Karnaprayag", 30.2597, 79.2153, 788, 400),
    ("Gopeshwar",  30.4076, 79.3186, 1450, 500),
    ("Ukhimath",   30.5150, 79.0900, 1311, 500),
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    g = get_gazetteer()

    print("\n[1] GAZETTEER IDENTITY  (is this really the right district?)")
    print("-" * 74)
    check("settlements loaded", len(g.settlements) > 1500,
          "%d settlements" % len(g.settlements))
    check("river geometry loaded", len(g.rivers) >= 20, "%d river ways" % len(g.rivers))
    check("road network loaded", len(g.roads) >= 200,
          "%d ways, %d bridges" % (len(g.roads), sum(1 for r in g.roads if r.get("bridge"))))

    print("\n[2] REAL PLACES AT REAL COORDINATES")
    print("-" * 74)
    byname = {}
    for st in g.settlements.values():
        byname.setdefault(st.name.lower(), []).append(st)
    for name, lat, lon, elev, tol in KNOWN:
        cands = byname.get(name.lower(), [])
        if not cands:
            warn("%s present in gazetteer" % name, "not found in OSM extract")
            continue
        best = min(cands, key=lambda s: haversine_km(lat, lon, s.lat, s.lon))
        d = haversine_km(lat, lon, best.lat, best.lon)
        check("%s within 5 km of published coords" % name, d < 5.0,
              "off by %.2f km" % d)

    print("\n[3] TERRAIN ELEVATION vs PUBLISHED VALUES  (Open-Meteo / SRTM)")
    print("-" * 74)
    for name, lat, lon, elev, tol in KNOWN:
        cands = byname.get(name.lower(), [])
        if not cands:
            continue
        best = min(cands, key=lambda s: haversine_km(lat, lon, s.lat, s.lon))
        if best.elevation_m <= 0:
            warn("%s elevation present" % name, "no elevation cached")
            continue
        check("%s elevation ~%dm" % (name, elev), abs(best.elevation_m - elev) < tol,
              "got %.0fm (published %dm, tol %dm)" % (best.elevation_m, elev, tol))

    print("\n[4] HAZARD EXPOSURE IS PHYSICALLY SENSIBLE")
    print("-" * 74)
    riverside = [s for s in g.settlements.values() if s.river_dist_km < 0.4]
    upland = [s for s in g.settlements.values() if s.river_dist_km > 6.0]
    if riverside and upland:
        fr = sum(s.flood_exposure for s in riverside) / len(riverside)
        fu = sum(s.flood_exposure for s in upland) / len(upland)
        check("riverside floods more than upland", fr > fu * 2.0,
              "riverside %.2f vs upland %.2f" % (fr, fu))
    hi = [s for s in g.settlements.values() if s.elevation_m > 2500]
    lo = [s for s in g.settlements.values() if 0 < s.elevation_m < 900]
    if hi and lo:
        lh = sum(s.landslide_exposure for s in hi) / len(hi)
        ll = sum(s.landslide_exposure for s in lo) / len(lo)
        check("steep high ground slides more than valley floor", lh > ll,
              "high %.2f vs low %.2f" % (lh, ll))
        fh = sum(s.flood_exposure for s in hi) / len(hi)
        fl = sum(s.flood_exposure for s in lo) / len(lo)
        check("valley floor floods more than high ground", fl > fh,
              "low %.2f vs high %.2f" % (fl, fh))
    elevs = [s.elevation_m for s in g.settlements.values() if s.elevation_m > 0]
    check("elevation range matches Garhwal Himalaya", min(elevs) > 300 and max(elevs) > 3000,
          "%.0f - %.0f m" % (min(elevs), max(elevs)))

    print("\n[5] SEISMIC ATTENUATION vs PUBLISHED MMI")
    print("-" * 74)
    # Reference: 2013 M6.9-class Himalayan events produce roughly MMI VII in
    # the epicentral zone, decaying to MMI IV-V by ~100 km. Our engine uses
    # mmi = 1.5M - 3.2 log10(d) - 3.4 (a standard log-distance form).
    from fusion import FusionEngine
    eng = FusionEngine(gaz=g, epicentre=(30.545, 79.055), magnitude=6.8)

    import config as C

    def mmi(d):
        return C.mmi_at(6.8, d)

    for d, lo_, hi_ in ((5.0, 5.5, 8.5), (30.0, 3.5, 7.0), (100.0, 2.0, 5.5)):
        v = mmi(d)
        check("MMI at %.0f km in [%.1f, %.1f]" % (d, lo_, hi_), lo_ <= v <= hi_,
              "MMI %.1f" % v)
    check("MMI decays with distance", mmi(5) > mmi(30) > mmi(100), "monotone")

    print("\n[6] LIVE PUBLIC FEEDS")
    print("-" * 74)
    import livedata
    livedata._mem.update(livedata._load_disk())
    snap = livedata.snapshot()
    u = snap["usgs"]
    if u["status"].startswith("unavailable"):
        warn("USGS feed reachable", u["status"])
    else:
        check("USGS feed parses", u["count"] >= 0, "%d events (%s)" % (u["count"], u["status"]))
        if u["events"]:
            mags = [e["mag"] for e in u["events"] if e.get("mag") is not None]
            check("USGS magnitudes plausible", all(2.0 <= m <= 9.5 for m in mags),
                  "range %.1f - %.1f" % (min(mags), max(mags)))
            lats = [e["lat"] for e in u["events"]]
            check("USGS events inside requested bbox", all(20 <= a <= 40 for a in lats),
                  "lat %.1f - %.1f" % (min(lats), max(lats)))
    r = snap["rainfall"]
    if r.get("status", "").startswith("unavailable"):
        warn("Open-Meteo reachable", r.get("status"))
    else:
        mm = r.get("total_72h_mm", 0.0)
        check("rainfall in physical range", 0 <= mm < 2000, "%.1f mm / 72h" % mm)
        f = snap["derived_rain_factor"]
        check("derived rain factor bounded", 0.3 <= f <= 1.65, "factor %.3f" % f)

    print("\n[7] FUZZY RESOLUTION ON REAL MISSPELLINGS")
    print("-" * 74)
    # How these names actually get mangled over a bad line or by a panicked caller.
    cases = [
        ("Rudraprayag", ["rudarprayag", "rudra prayag", "RUDRAPRYAG"]),
        ("Karnaprayag", ["karan prayag", "karnprayag"]),
        ("Gopeshwar",   ["gopeshwer", "gopeswar"]),
        ("Ukhimath",    ["ukimath", "ukhi math"]),
    ]
    for truth_name, variants in cases:
        for v in variants:
            res = g.resolve(v)
            names = [g.settlements[sid].name.lower() for sid, _ in res]
            ok = truth_name.lower() in names
            check("'%s' resolves to %s" % (v, truth_name), ok,
                  "top: %s" % (", ".join(names[:3]) if names else "none"))

    print("\n[8] AMBIGUITY IS PRESERVED, NOT GUESSED AWAY")
    print("-" * 74)
    dupes = {n: v for n, v in byname.items() if len(v) > 2}
    check("district really does contain duplicate village names", len(dupes) > 5,
          "%d names shared by 3+ settlements" % len(dupes))
    if dupes:
        n = max(dupes, key=lambda k: len(dupes[k]))
        res = g.resolve(n)
        spread = 1.0 - max(p for _, p in res) if res else 0
        check("ambiguous name keeps mass on alternatives", spread > 0.35,
              "'%s' x%d, top candidate only %.0f%%"
              % (n, len(dupes[n]), 100 * max(p for _, p in res)))

    print("\n" + "=" * 74)
    print("REAL-DATA VALIDATION: %d passed, %d failed, %d warnings"
          % (len(PASS), len(FAIL), len(WARN)))
    if FAIL:
        print("\nFAILURES:")
        for n, d in FAIL:
            print("   - %s  %s" % (n, d))
    print("=" * 74)
    json.dump({"passed": len(PASS), "failed": len(FAIL), "warnings": len(WARN),
               "failures": FAIL, "warns": WARN},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "realworld_validation.json"), "w"), indent=1)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
