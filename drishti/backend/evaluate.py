"""Evaluation harness - does DRISHTI actually beat what an EOC does today?

We score every method on the only question that matters operationally:
given that you can only reach K settlements in the next few hours, how many
of the genuinely catastrophic ones did you choose, and how many people in
real trouble did you reach?

Baselines represent what actually happens now:
  VOLUME      rank by how many reports mention a place  (a live dashboard,
              a WhatsApp war-room, "where is the noise")
  LOUDEST     rank by the most alarming claim received  (panic-driven)
  NAIVE_MEAN  average the claimed severities, treat all sources equally
              (a spreadsheet at the EOC)
  DRISHTI     full engine
  plus one-mechanism-removed ablations, to show what each part earns.
"""
import json, math, os, sys
from collections import defaultdict

import config as C
from fusion import FusionEngine
from extract import extract
from gazetteer import get_gazetteer
from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX

DATA = os.path.join(os.path.dirname(__file__), "data")
CATASTROPHIC = "CATASTROPHIC"


# ------------------------------------------------------------- baselines
def baseline_rankings(recs, truth):
    vol = defaultdict(float)
    loud = defaultdict(float)
    tot = defaultdict(float)
    cnt = defaultdict(float)
    for r in recs:
        for sid, gp in r["geo"]:
            vol[sid] += gp
            if r["claim_sev"] is not None:
                loud[sid] = max(loud[sid], r["claim_sev"])
                tot[sid] += r["claim_sev"] * gp
                cnt[sid] += gp
    mean = {k: tot[k] / cnt[k] for k in cnt if cnt[k] > 0}
    # tie-break by population, as a human would
    return {
        "VOLUME": sorted(vol, key=lambda s: -vol[s]),
        "LOUDEST": sorted(loud, key=lambda s: (-loud[s], -vol.get(s, 0))),
        "NAIVE_MEAN": sorted(mean, key=lambda s: (-mean[s], -vol.get(s, 0))),
    }


def score(ranking, truth, K):
    """Operational scoring at deployment budget K."""
    topk = [s for s in ranking[:K] if s in truth]
    cat_all = [s for s, v in truth.items() if v["state"] == CATASTROPHIC]
    cat_hit = [s for s in topk if truth[s]["state"] == CATASTROPHIC]
    reached = sum(truth[s]["casualties"] for s in topk)
    total_cas = sum(v["casualties"] for v in truth.values())
    return {
        "catastrophic_found": len(cat_hit),
        "catastrophic_total": len(cat_all),
        "recall": round(len(cat_hit) / max(1, len(cat_all)), 4),
        "casualties_reached": reached,
        "casualties_total": total_cas,
        "casualty_coverage": round(reached / max(1, total_cas), 4),
        "wasted_deployments": sum(1 for s in topk
                                  if truth[s]["state"] in ("INTACT", "MINOR")),
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))

    K = 40                      # realistic: ~40 deployable asset-hours in phase 1
    T = float(os.environ.get("EVAL_T", "24"))

    eng = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    recs = eng.prepare(reports)
    recs_t = [r for r in recs if r["t"] <= T]

    rows = []
    base = baseline_rankings(recs_t, truth)
    for name, rk in base.items():
        rows.append((name, score(rk, truth, K)))

    variants = [
        ("DRISHTI (full)",        {"silence": True,  "independence": True,
                                   "credibility": True, "prior": True}),
        ("  - no silence engine", {"silence": False, "independence": True,
                                   "credibility": True, "prior": True}),
        ("  - no independence",   {"silence": True,  "independence": False,
                                   "credibility": True, "prior": True}),
        ("  - no source credib.", {"silence": True,  "independence": True,
                                   "credibility": False, "prior": True}),
        ("  - no terrain prior",  {"silence": True,  "independence": True,
                                   "credibility": True, "prior": False}),
    ]
    detail = {}
    for name, flags in variants:
        eng.flags = flags
        eng.cred = {k: list(v) for k, v in C.SOURCE_PRIORS.items()}
        out, _ = eng.fuse(T)
        rk = sorted(out, key=lambda s: -out[s]["priority"])
        rows.append((name, score(rk, truth, K)))
        if name.startswith("DRISHTI"):
            detail = {"out": out, "rank": rk}

    # ---------------------------------------------------------- report
    print("=" * 78)
    print("DRISHTI EVALUATION   t=%.0fh after onset   deployment budget K=%d" % (T, K))
    print("district: %d settlements, %d reports ingested"
          % (len(truth), len(recs_t)))
    print("=" * 78)
    hdr = "%-24s %10s %10s %12s %10s" % (
        "METHOD", "CATA-HIT", "RECALL", "PEOPLE-RCHD", "WASTED")
    print(hdr)
    print("-" * 78)
    for name, s in rows:
        print("%-24s %5d/%-4d %9.1f%% %12d %10d" % (
            name, s["catastrophic_found"], s["catastrophic_total"],
            100 * s["recall"], s["casualties_reached"], s["wasted_deployments"]))
    print("-" * 78)

    full = dict(rows)["DRISHTI (full)"]
    vol = dict(rows)["VOLUME"]
    if vol["recall"] > 0:
        print("\nDRISHTI vs VOLUME ranking: %.1fx more catastrophic zones found, "
              "%.1fx more people reached"
              % (full["recall"] / max(1e-9, vol["recall"]),
                 full["casualties_reached"] / max(1, vol["casualties_reached"])))
    else:
        print("\nVOLUME ranking found ZERO catastrophic zones in its top %d." % K)
        print("DRISHTI found %d of %d." % (full["catastrophic_found"],
                                           full["catastrophic_total"]))

    # the headline story: places found ONLY because they were silent
    out, rk = detail["out"], detail["rank"]
    dark = [s for s in rk[:K]
            if out[s]["silence"].get("is_dark") and truth.get(s, {}).get("state")
            in ("CATASTROPHIC", "MAJOR")]
    print("\nHigh-severity zones surfaced BY SILENCE (zero/near-zero reports):")
    for s in dark[:10]:
        v, t = out[s], truth[s]
        print("   %-20s pop=%6d  reports=%4.1f (expected %5.1f)  truth=%s"
              % (v["name"][:20], v["population"], v["silence"]["observed"],
                 v["silence"]["expected"], t["state"]))
    if not dark:
        print("   (none in top-K for this seed)")

    # rumour suppression
    rum = [r for r in recs_t if r["independence"] < 0.15]
    print("\nPanic amplification suppressed: %d of %d reports down-weighted "
          "below 0.15 independence" % (len(rum), len(recs_t)))

    json.dump({"rows": [{"method": n, **s} for n, s in rows], "K": K, "t": T},
              open(os.path.join(DATA, "evaluation.json"), "w"), indent=1)
    print("\nsaved -> data/evaluation.json")


if __name__ == "__main__":
    main()
