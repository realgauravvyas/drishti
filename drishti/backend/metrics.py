"""Rigorous metrics. Separates two questions that rank-at-K conflates:

  (a) DETECTION  - does the belief state DSI actually track true severity?
                   Measured by ROC-AUC, which is population-independent.
  (b) TRIAGE     - given a deployment budget, do we save the most people?
                   Measured by casualties reached at K, which SHOULD be
                   population-weighted, because that is the real objective.

A mechanism can help (a) and look flat on (b) simply because big towns
dominate expected-casualty arithmetic. Reporting only one of them would be
misleading, so we report both.
"""
import json, os, sys
from collections import Counter

import numpy as np

from fusion import FusionEngine
from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX

DATA = os.path.join(os.path.dirname(__file__), "data")


def auc(scores, labels):
    """ROC-AUC via rank statistic (no sklearn dependency needed here)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load():
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))
    eng = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    eng.prepare(reports)
    return eng, truth


VARIANTS = [
    ("DRISHTI (full)",   {"silence": 1, "independence": 1, "credibility": 1, "prior": 1}),
    ("no silence",       {"silence": 0, "independence": 1, "credibility": 1, "prior": 1}),
    ("no terrain prior", {"silence": 1, "independence": 1, "credibility": 1, "prior": 0}),
    ("no independence",  {"silence": 1, "independence": 0, "credibility": 1, "prior": 1}),
    ("no credibility",   {"silence": 1, "independence": 1, "credibility": 0, "prior": 1}),
    ("reports only",     {"silence": 0, "independence": 0, "credibility": 0, "prior": 0}),
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    eng, truth = load()
    T = float(os.environ.get("EVAL_T", "24"))
    results = {}

    print("=" * 76)
    print("DETECTION QUALITY - ROC-AUC for 'is truly MAJOR or CATASTROPHIC'")
    print("(population-independent: pure question of whether belief tracks truth)")
    print("=" * 76)
    print("%-20s %10s %14s %14s" % ("VARIANT", "AUC all", "AUC blind-spot", "AUC reported"))
    print("-" * 76)

    for name, fl in VARIANTS:
        eng.flags = {k: bool(v) for k, v in fl.items()}
        out, _ = eng.fuse(T)
        sids = [s for s in out if s in truth]
        y = [1 if truth[s]["state"] in ("MAJOR", "CATASTROPHIC") else 0 for s in sids]
        d = [out[s]["dsi"] for s in sids]
        blind = [i for i, s in enumerate(sids) if out[s]["raw_report_count"] == 0]
        rep = [i for i, s in enumerate(sids) if out[s]["raw_report_count"] > 0]
        a_all = auc(d, y)
        a_bl = auc([d[i] for i in blind], [y[i] for i in blind])
        a_rp = auc([d[i] for i in rep], [y[i] for i in rep])
        results[name] = {"auc_all": a_all, "auc_blind": a_bl, "auc_reported": a_rp}
        print("%-20s %10.3f %14.3f %14.3f" % (name, a_all, a_bl, a_rp))

    # ------------------------------------------------ dark-flag precision
    eng.flags = {"silence": True, "independence": True, "credibility": True, "prior": True}
    out, _ = eng.fuse(T)
    dark = [s for s in out if out[s]["silence"].get("is_dark") and s in truth]
    hi = [s for s in dark if truth[s]["state"] in ("MAJOR", "CATASTROPHIC")]
    allsid = [s for s in out if s in truth]
    base = sum(1 for s in allsid if truth[s]["state"] in ("MAJOR", "CATASTROPHIC")) / len(allsid)
    prec = len(hi) / max(1, len(dark))
    print("\n" + "=" * 76)
    print("DARK-ZONE FLAG (the operational product of the silence engine)")
    print("=" * 76)
    print("  flagged dark              : %d settlements" % len(dark))
    print("  truly MAJOR/CATASTROPHIC  : %d  -> precision %.1f%%" % (len(hi), 100 * prec))
    print("  district base rate        : %.1f%%" % (100 * base))
    print("  lift over base rate       : %.2fx" % (prec / max(1e-9, base)))
    print("  population in flagged dark: %d" % sum(out[s]["population"] for s in dark))
    truly_cat = sum(1 for s in dark if truth[s]["state"] == "CATASTROPHIC")
    print("  of which truly CATASTROPHIC: %d" % truly_cat)

    # how many of these would ANY report-driven method have seen?
    unseen = [s for s in dark if out[s]["raw_report_count"] == 0]
    print("  ...with ZERO reports (invisible to every baseline): %d" % len(unseen))

    results["dark_flag"] = {
        "flagged": len(dark), "precision": prec, "base_rate": base,
        "lift": prec / max(1e-9, base), "truly_catastrophic": truly_cat,
        "zero_report": len(unseen),
        "population": sum(out[s]["population"] for s in dark),
    }
    json.dump(results, open(os.path.join(DATA, "metrics.json"), "w"), indent=1)
    print("\nsaved -> data/metrics.json")


if __name__ == "__main__":
    main()
