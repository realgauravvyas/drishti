"""Gazetteer: real OSM settlements + terrain exposure + fuzzy location resolution.

The hardest part of the "information fog" is that ground reports name places
loosely ("the village past the broken bridge near Ukhimath"), and Indian
districts contain many settlements with near-identical names. This module
turns a fuzzy human phrase into a *probability distribution over settlements*
rather than a single guess - uncertainty is preserved, not discarded.
"""
import json, math, os, re
from dataclasses import dataclass, field
from rapidfuzz import fuzz, process

DATA = os.path.join(os.path.dirname(__file__), "data")
GAZ = os.path.join(DATA, "gazetteer.json")
ELEV_CACHE = os.path.join(DATA, "elevation.json")

# Population priors by OSM place class (Census-2011 style rural averages for
# Uttarakhand hill districts). Real population tags override these.
POP_PRIOR = {"city": 90000, "town": 16000, "village": 1150, "hamlet": 220}


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


@dataclass
class Settlement:
    sid: str
    name: str
    lat: float
    lon: float
    place: str
    population: int
    elevation_m: float = 0.0
    river_dist_km: float = 99.0
    flood_exposure: float = 0.0
    landslide_exposure: float = 0.0
    connectivity: float = 0.5     # pre-event connectedness -> drives silence baseline
    aliases: list = field(default_factory=list)

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("sid", "name", "lat", "lon", "place", "population", "elevation_m",
                 "river_dist_km", "flood_exposure", "landslide_exposure", "connectivity")}


class Gazetteer:
    def __init__(self, path=GAZ):
        raw = json.load(open(path, encoding="utf-8"))
        self.bbox = raw["bbox"]
        self.rivers = raw.get("rivers", [])
        self.roads = raw.get("roads", [])
        self.settlements = {}
        for s in raw["settlements"]:
            sid = "S%d" % s["osm_id"]
            pop = s.get("population") or POP_PRIOR.get(s.get("place"), 400)
            # deterministic jitter so villages are not all identical size
            jitter = 0.55 + ((s["osm_id"] * 2654435761) % 1000) / 1000.0 * 0.9
            self.settlements[sid] = Settlement(
                sid=sid, name=s["name"], lat=s["lat"], lon=s["lon"],
                place=s.get("place") or "village",
                population=max(40, int(pop * jitter)),
                aliases=[a for a in [s.get("name_hi")] if a],
            )
        self._name_index = {sid: st.name.lower() for sid, st in self.settlements.items()}
        self._compute_exposure()

    # ------------------------------------------------------------- exposure
    def _river_points(self):
        pts = []
        for r in self.rivers:
            c = r["coords"]
            step = max(1, len(c) // 60)
            pts.extend(c[::step])
        return pts

    def _compute_exposure(self):
        rp = self._river_points()
        elev = {}
        if os.path.exists(ELEV_CACHE):
            elev = json.load(open(ELEV_CACHE, encoding="utf-8"))
        for sid, s in self.settlements.items():
            best = 99.0
            for (rlat, rlon) in rp:
                if abs(rlat - s.lat) > 0.12 or abs(rlon - s.lon) > 0.12:
                    continue
                d = haversine_km(s.lat, s.lon, rlat, rlon)
                if d < best:
                    best = d
            s.river_dist_km = round(best, 3)
            s.elevation_m = float(elev.get(sid, 0.0))

        vals = [s.elevation_m for s in self.settlements.values() if s.elevation_m > 0]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        rng = max(1.0, hi - lo)
        for s in self.settlements.values():
            rel = (s.elevation_m - lo) / rng if s.elevation_m > 0 else 0.5
            # Flood exposure: close to river AND low-lying
            prox = math.exp(-s.river_dist_km / 1.6)
            s.flood_exposure = round(min(1.0, prox * (1.25 - 0.75 * rel)), 3)
            # Landslide exposure: steep upper-slope terrain near valley walls
            s.landslide_exposure = round(min(1.0, 0.35 + 0.65 * rel) *
                                         (0.5 + 0.5 * math.exp(-s.river_dist_km / 4.0)), 3)
            # Connectivity: larger + lower places are better connected
            size = min(1.0, math.log10(max(10, s.population)) / 4.5)
            s.connectivity = round(max(0.08, min(0.98, 0.75 * size + 0.35 * (1 - rel))), 3)

    # --------------------------------------------------- fuzzy geo-resolution
    LANDMARK_HINTS = re.compile(
        r"\b(near|past|beyond|behind|below|above|opposite|towards?|next to|"
        r"upstream of|downstream of|across from|village|gaon|gaun)\b", re.I)

    def resolve(self, phrase, hint_latlon=None, topk=4):
        """Return [(sid, probability)] for a fuzzy place phrase.

        Preserves ambiguity: if three villages match equally well, all three
        carry mass. This is what stops the EOC from confidently assigning a
        boat to the wrong Devalgaon.
        """
        if not phrase:
            return []
        p = phrase.lower().strip()
        p = self.LANDMARK_HINTS.sub(" ", p)
        p = re.sub(r"[^a-z\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            return []

        cands = process.extract(p, self._name_index, scorer=fuzz.WRatio,
                                limit=40, processor=None)
        if not cands:
            return []
        scored = []
        for _name, score, sid in cands:
            s = self.settlements[sid]
            w = (score / 100.0) ** 3.0          # sharpen: weak matches die fast
            if hint_latlon:                      # spatial prior from partial GPS
                d = haversine_km(hint_latlon[0], hint_latlon[1], s.lat, s.lon)
                w *= math.exp(-d / 12.0)
            w *= 0.6 + 0.4 * min(1.0, math.log10(max(10, s.population)) / 4.5)
            if w > 1e-4:
                scored.append((sid, w))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[1])
        scored = scored[:topk]
        tot = sum(w for _, w in scored)
        return [(sid, round(w / tot, 4)) for sid, w in scored]

    # ------------------------------------------------------------- helpers
    def near(self, lat, lon, radius_km=6.0):
        out = []
        for s in self.settlements.values():
            if abs(s.lat - lat) > 0.1 or abs(s.lon - lon) > 0.1:
                continue
            d = haversine_km(lat, lon, s.lat, s.lon)
            if d <= radius_km:
                out.append((s.sid, d))
        out.sort(key=lambda x: x[1])
        return out

    def subset_bbox(self, s, w, n, e):
        return [st for st in self.settlements.values()
                if s <= st.lat <= n and w <= st.lon <= e]


_G = None


def get_gazetteer():
    global _G
    if _G is None:
        _G = Gazetteer()
    return _G


if __name__ == "__main__":
    g = get_gazetteer()
    print("%d settlements, %d rivers, %d roads" %
          (len(g.settlements), len(g.rivers), len(g.roads)))
    for q in ["Ukhimath", "the village past Gaurikund", "Devalgaon", "chandrapuri"]:
        print("\n  '%s' ->" % q)
        for sid, pr in g.resolve(q):
            st = g.settlements[sid]
            print("     %5.2f  %-24s pop=%6d flood=%.2f river=%.1fkm"
                  % (pr, st.name, st.population, st.flood_exposure, st.river_dist_km))
