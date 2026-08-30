"""DRISHTI API - FastAPI service for the District Emergency Operations Centre.

Runs entirely offline once the gazetteer is cached. Live public feeds (USGS,
Open-Meteo) are used when reachable and degrade silently when they are not,
because a disaster demo on venue wifi must never depend on the network.
"""
import json, os, threading, time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config as C
from fusion import FusionEngine
from allocate import RouteNetwork, build_plan, DEPOTS
from gazetteer import get_gazetteer
import livedata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "frontend"))

app = FastAPI(title="DRISHTI", version="1.0",
              description="Post-Disaster Information Fog resolver (Avinya 2026, PS-5)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE = {"ready": False, "error": None, "progress": "starting"}
_lock = threading.Lock()
_cache = {}


def boot():
    """Load data and warm the engine. Runs once, in a background thread, so
    the UI can show a progress state instead of a blank page."""
    try:
        STATE["progress"] = "loading gazetteer"
        gaz = get_gazetteer()

        STATE["progress"] = "loading scenario"
        reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
        truth = {}
        tp = os.path.join(DATA, "ground_truth.json")
        if os.path.exists(tp):
            truth = json.load(open(tp, encoding="utf-8"))

        from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX, DURATION_HRS
        STATE["progress"] = "preparing evidence graph"
        eng = FusionEngine(gaz=gaz, epicentre=EPICENTRE, magnitude=MAGNITUDE,
                           aoi=SCENARIO_BBOX)
        recs = eng.prepare(reports)

        STATE["progress"] = "building route network"
        net = RouteNetwork(gaz)

        STATE.update({
            "gaz": gaz, "reports": reports, "truth": truth, "eng": eng,
            "recs": {r["rid"]: r for r in recs}, "net": net,
            "epicentre": EPICENTRE, "magnitude": MAGNITUDE,
            "duration": DURATION_HRS, "aoi": SCENARIO_BBOX,
            "raw_by_id": {r["rid"]: r for r in reports},
            "ready": True, "progress": "ready",
        })
    except Exception as e:  # surfaced to the UI rather than dying silently
        import traceback
        STATE["error"] = "%s: %s" % (type(e).__name__, e)
        STATE["progress"] = "failed"
        traceback.print_exc()


threading.Thread(target=boot, daemon=True).start()


def need_ready():
    if not STATE.get("ready"):
        raise HTTPException(503, "engine warming up: %s" % STATE.get("progress"))


def belief_at(t, use_cache=True):
    key = round(float(t), 2)
    if use_cache and key in _cache:
        return _cache[key]
    with _lock:
        if use_cache and key in _cache:
            return _cache[key]
        b, _ = STATE["eng"].fuse(key)
        if len(_cache) > 60:
            _cache.clear()
        _cache[key] = b
    return b


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"ready": STATE.get("ready", False), "progress": STATE.get("progress"),
            "error": STATE.get("error")}


@app.get("/api/scenario")
def scenario():
    need_ready()
    return {
        "epicentre": STATE["epicentre"], "magnitude": STATE["magnitude"],
        "duration_hours": STATE["duration"], "aoi": STATE["aoi"],
        "depots": DEPOTS,
        "settlements_in_aoi": len(STATE["eng"]._aoi_sids or []),
        "reports_total": len(STATE["reports"]),
        "rivers": [r["coords"][::3] for r in STATE["gaz"].rivers[:40]],
        "source_channels": list(C.SOURCE_PRIORS.keys()),
    }


@app.get("/api/state")
def state(t: float = Query(24.0, ge=0.0, le=48.0)):
    """Full district belief at time t (hours since onset)."""
    need_ready()
    b = belief_at(t)
    eng = STATE["eng"]
    rows = sorted(b.values(), key=lambda v: -v["priority"])
    dark = [v for v in rows if v["silence"].get("is_dark")]
    contested = [v for v in rows if v["contradiction"] > 0.45]
    n_live = sum(1 for r in STATE["reports"] if r["t_hours"] <= t)
    agg = {"INTACT": 0, "MINOR": 0, "MAJOR": 0, "CATASTROPHIC": 0}
    pop_at_risk = 0
    for v in rows:
        agg[v["state"]] += 1
        pop_at_risk += v["population"] * (v["distribution"]["MAJOR"]
                                          + v["distribution"]["CATASTROPHIC"])
    return {
        "t": t,
        "settlements": rows,
        "summary": {
            "reports_ingested": n_live,
            "settlements": len(rows),
            "state_counts": agg,
            "population_at_risk": int(pop_at_risk),
            "dark_zones": len(dark),
            "contested": len(contested),
            "prior_trust": round(getattr(eng, "prior_trust", 0.0), 3),
            "source_credibility": {k: round(eng.credibility(k), 3)
                                   for k in C.SOURCE_PRIORS},
        },
    }


@app.get("/api/settlement/{sid}")
def settlement(sid: str, t: float = Query(24.0, ge=0.0, le=48.0)):
    """Full audit trail for one settlement - every report that moved belief."""
    need_ready()
    b = belief_at(t)
    if sid not in b:
        raise HTTPException(404, "unknown settlement")
    v = dict(b[sid])
    recs, raw = STATE["recs"], STATE["raw_by_id"]
    ev = []
    for rid in v.get("evidence", []):
        r = recs.get(rid)
        if not r or r["t"] > t:
            continue
        geo = dict(r["geo"])
        ev.append({
            "rid": rid, "t": r["t"], "source": r["source"],
            "source_id": r["source_id"], "text": r["text"],
            "claimed_place": raw.get(rid, {}).get("claimed_place", ""),
            "claim_severity": r["claim_sev"], "sev_conf": r["sev_conf"],
            "independence": r["independence"], "panic": r["panic"],
            "specificity": r["spec"], "hazards": r["hazards"],
            "geo_probability": round(geo.get(sid, 0.0), 3),
            "credibility": round(STATE["eng"].credibility(r["source"]), 3),
            "cluster": r["cluster"],
        })
    ev.sort(key=lambda x: -(x["independence"] * x["geo_probability"]))
    v["evidence_detail"] = ev
    if STATE["truth"]:
        v["_ground_truth"] = STATE["truth"].get(sid)   # demo/eval only
    return v


@app.get("/api/plan")
def plan(t: float = Query(24.0, ge=0.0, le=48.0)):
    need_ready()
    b = belief_at(t)
    net = STATE["net"]
    with _lock:
        net.apply_damage(b)
        p = build_plan(b, net, t_now=t)
    blocked = [{"a": a, "b": bb, "hazard": d.get("hazard", "")}
               for a, bb, d in net.G.edges(data=True)
               if d.get("blocked") and isinstance(a[0], float)]
    p["blocked_edges"] = blocked[:600]
    return p


@app.get("/api/reports")
def reports(t: float = Query(24.0, ge=0.0, le=48.0), limit: int = 60,
            source: str = None):
    """Live incoming feed, newest first, as the EOC would watch it."""
    need_ready()
    recs = STATE["recs"]
    out = []
    for r in STATE["reports"]:
        if r["t_hours"] > t:
            continue
        if source and r["source"] != source:
            continue
        rec = recs.get(r["rid"])
        out.append({
            "rid": r["rid"], "t": r["t_hours"], "source": r["source"],
            "text": r["raw_text"], "claimed_place": r.get("claimed_place"),
            "gps": r.get("gps"),
            "resolved": [{"sid": s, "p": p} for s, p in (rec["geo"][:3] if rec else [])],
            "independence": rec["independence"] if rec else None,
            "panic": rec["panic"] if rec else None,
        })
    out.sort(key=lambda x: -x["t"])
    return {"count": len(out), "reports": out[:limit]}


@app.get("/api/metrics")
def metrics():
    """Benchmark results, precomputed by evaluate.py / metrics.py."""
    res = {}
    for name in ("evaluation", "metrics"):
        p = os.path.join(DATA, "%s.json" % name)
        if os.path.exists(p):
            res[name] = json.load(open(p, encoding="utf-8"))
    if not res:
        raise HTTPException(404, "run `python evaluate.py && python metrics.py` first")
    return res


@app.get("/api/live")
def live():
    """Real public feeds - free, no API key. Cached, and safe when offline."""
    return livedata.snapshot()


@app.get("/api/config")
def get_config():
    return {"source_priors": C.SOURCE_PRIORS, "asset_types": C.ASSET_TYPES,
            "damage_states": C.DAMAGE_STATES,
            "severity_weight": C.SEVERITY_WEIGHT,
            "casualty_rate": C.CASUALTY_RATE}


# --------------------------------------------------------------- frontend
if os.path.isdir(FRONTEND):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
