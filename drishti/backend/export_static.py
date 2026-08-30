"""Export a self-contained static build of the dashboard for GitHub Pages.

The live app needs Python. Judges, teammates and anyone following a README
link do not want to install Python. So we precompute the engine's output at a
set of timesteps and emit a folder that is pure HTML + JSON, servable from
GitHub Pages with no backend at all.

    python export_static.py       ->  ../../docs/demo/

What is exported:
  scenario.json          district geometry, depots, epicentre
  state_<t>.json         full district belief at each timestep
  plan_<t>.json          asset tasking at each timestep
  detail/<sid>.json      evidence drill-down for the settlements that matter
  metrics.json           benchmark + validation results
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "demo"))
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "frontend"))

TIMESTEPS = [1, 3, 6, 9, 12, 15, 18, 21, 24]
DETAIL_TOP = 140          # settlements to ship evidence for


def rnd(v, n=3):
    return round(v, n) if isinstance(v, float) else v


def slim_settlement(v):
    """Trim to what the UI actually reads, and round hard. Full precision on
    1102 settlements x 9 timesteps is ~15 MB; this brings it under 3 MB."""
    s = {
        "sid": v["sid"], "name": v["name"],
        "lat": round(v["lat"], 5), "lon": round(v["lon"], 5),
        "population": v["population"],
        "dsi": rnd(v["dsi"]), "state": v["state"],
        "confidence": rnd(v["confidence"]),
        "evidence_mass": rnd(v["evidence_mass"]),
        "contradiction": rnd(v["contradiction"]),
        "priority": rnd(v["priority"], 2), "voi": rnd(v["voi"], 2),
        "raw_report_count": v["raw_report_count"],
        "hazards": v["hazards"], "needs": v.get("needs", []),
        "distribution": {k: rnd(x) for k, x in v["distribution"].items()},
    }
    sil = v.get("silence") or {}
    if sil:
        s["silence"] = {k: sil.get(k) for k in
                        ("is_dark", "observed", "expected", "z",
                         "blackout_risk", "reason", "tower_down_h")
                        if k in sil}
    else:
        s["silence"] = {}
    return s


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from fusion import FusionEngine
    from allocate import RouteNetwork, build_plan, DEPOTS
    from gazetteer import get_gazetteer
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX, DURATION_HRS
    import config as C

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "data", "detail"), exist_ok=True)

    gaz = get_gazetteer()
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    truth = {}
    tp = os.path.join(DATA, "ground_truth.json")
    if os.path.exists(tp):
        truth = json.load(open(tp, encoding="utf-8"))

    print("preparing engine...")
    eng = FusionEngine(gaz=gaz, epicentre=EPICENTRE, magnitude=MAGNITUDE,
                       aoi=SCENARIO_BBOX)
    recs = eng.prepare(reports)
    rec_by_id = {r["rid"]: r for r in recs}
    raw_by_id = {r["rid"]: r for r in reports}
    net = RouteNetwork(gaz)

    d = os.path.join(OUT, "data")
    json.dump({
        "epicentre": EPICENTRE, "magnitude": MAGNITUDE,
        "duration_hours": DURATION_HRS, "aoi": SCENARIO_BBOX,
        "depots": DEPOTS, "timesteps": TIMESTEPS,
        "settlements_in_aoi": len(eng._aoi_sids or []),
        "reports_total": len(reports),
        "source_channels": list(C.SOURCE_PRIORS.keys()),
        "static": True,
    }, open(os.path.join(d, "scenario.json"), "w"))

    last = None
    for t in TIMESTEPS:
        belief, _ = eng.fuse(float(t))
        rows = sorted(belief.values(), key=lambda v: -v["priority"])
        agg = {k: 0 for k in C.DAMAGE_STATES}
        pop = 0
        for v in rows:
            agg[v["state"]] += 1
            pop += v["population"] * (v["distribution"]["MAJOR"]
                                      + v["distribution"]["CATASTROPHIC"])
        state = {
            "t": t,
            "settlements": [slim_settlement(v) for v in rows],
            "summary": {
                "reports_ingested": sum(1 for r in reports if r["t_hours"] <= t),
                "settlements": len(rows), "state_counts": agg,
                "population_at_risk": int(pop),
                "dark_zones": sum(1 for v in rows if v["silence"].get("is_dark")),
                "contested": sum(1 for v in rows if v["contradiction"] > 0.45),
                "prior_trust": round(eng.prior_trust, 3),
                "source_credibility": {k: round(eng.credibility(k), 3)
                                       for k in C.SOURCE_PRIORS},
            },
        }
        json.dump(state, open(os.path.join(d, "state_%d.json" % t), "w"))

        net.apply_damage(belief)
        plan = build_plan(belief, net, t_now=float(t))
        plan["blocked_edges"] = [
            {"a": [round(a[0], 5), round(a[1], 5)],
             "b": [round(b[0], 5), round(b[1], 5)]}
            for a, b, dd in net.G.edges(data=True)
            if dd.get("blocked") and isinstance(a[0], float)][:500]
        json.dump(plan, open(os.path.join(d, "plan_%d.json" % t), "w"))
        print("  t=%2dh  %d settlements, %d dark, %d taskings"
              % (t, len(rows), state["summary"]["dark_zones"],
                 len(plan["assignments"])))
        last = (belief, rows)

    # evidence drill-down for the settlements a judge will actually click
    belief, rows = last
    want = [v["sid"] for v in rows[:DETAIL_TOP]]
    want += [v["sid"] for v in rows if v["silence"].get("is_dark")][:60]
    for sid in dict.fromkeys(want):
        v = dict(belief[sid])
        ev = []
        for rid in v.get("evidence", []):
            r = rec_by_id.get(rid)
            if not r:
                continue
            ev.append({
                "rid": rid, "t": r["t"], "source": r["source"],
                "text": r["text"],
                "claimed_place": raw_by_id.get(rid, {}).get("claimed_place", ""),
                "claim_severity": r["claim_sev"], "sev_conf": r["sev_conf"],
                "independence": r["independence"], "panic": r["panic"],
                "specificity": r["spec"], "hazards": r["hazards"],
                "geo_probability": round(dict(r["geo"]).get(sid, 0.0), 3),
                "credibility": round(eng.credibility(r["source"]), 3),
            })
        ev.sort(key=lambda x: -(x["independence"] * x["geo_probability"]))
        out = slim_settlement(v)
        out["evidence_detail"] = ev
        if truth.get(sid):
            out["_ground_truth"] = truth[sid]
        json.dump(out, open(os.path.join(d, "detail", "%s.json" % sid), "w"))
    print("  exported %d evidence drill-downs" % len(set(want)))

    for name in ("metrics", "evaluation", "validation", "calibration",
                 "realworld_validation"):
        p = os.path.join(DATA, "%s.json" % name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(d, "%s.json" % name))

    html = open(os.path.join(FRONTEND, "index.html"), encoding="utf-8").read()
    # Declare the static build up front rather than discovering it by letting
    # an API call fail: a 404 in the console makes a working demo look broken.
    html = html.replace(
        "<body>", '<body>\n<script>window.DRISHTI_STATIC=true;</script>', 1)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
    open(os.path.join(OUT, ".nojekyll"), "w").close()

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(OUT) for f in fs)
    print("\nstatic build -> %s  (%.1f MB)" % (OUT, total / 1e6))
    print("GitHub Pages: settings -> Pages -> deploy from branch, /docs")
    print("then open  https://<user>.github.io/<repo>/demo/")


if __name__ == "__main__":
    main()
