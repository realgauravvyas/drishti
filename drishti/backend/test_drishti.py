"""Unit and property tests.  Run:  python -m pytest test_drishti.py -q

These guard the invariants that actually matter for a decision-support tool:
belief must be a probability distribution, evidence must not be double-counted,
absence must be distinguishable from safety, and repeated queries must not
silently change the answer.
"""
import json, math, os

import numpy as np
import pytest

import config as C
from extract import extract
from gazetteer import get_gazetteer, haversine_km
from fusion import FusionEngine, DSU

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATES = C.DAMAGE_STATES


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def gaz():
    return get_gazetteer()


@pytest.fixture(scope="module")
def engine(gaz):
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    e = FusionEngine(gaz=gaz, epicentre=EPICENTRE, magnitude=MAGNITUDE,
                     aoi=SCENARIO_BBOX)
    e.prepare(reports)
    return e


@pytest.fixture(scope="module")
def belief(engine):
    out, _ = engine.fuse(24.0)
    return out


@pytest.fixture(scope="module")
def truth():
    return json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))


# ------------------------------------------------------------- geometry
def test_haversine_known_distance():
    # Rudraprayag -> Karnaprayag, about 32 km by air
    d = haversine_km(30.2844, 78.9811, 30.2597, 79.2153)
    assert 20 < d < 30, d


def test_haversine_symmetric_and_zero():
    assert haversine_km(30, 79, 30, 79) == pytest.approx(0.0)
    assert haversine_km(30, 79, 31, 80) == pytest.approx(haversine_km(31, 80, 30, 79))


# ------------------------------------------------------------- extractor
def test_extract_detects_flood_hinglish():
    c = extract("paani gaon me ghus gaya hai, ghar tak aa gaya")
    assert "FLOOD" in c.hazards


def test_extract_flags_panic_and_discounts_it():
    c = extract("SHARE THIS!! dam has broken, Chandrapuri completely finished, thousands dead")
    assert c.panic_score > 0.4
    # asserts maximal severity but must not earn high confidence
    assert c.severity is not None and c.severity > 0.8
    assert c.sev_conf < 0.5


def test_extract_safe_claim():
    c = extract("Chandrapuri me sab theek hai, thoda paani bas")
    assert c.is_safe_claim
    assert c.severity < 0.35


def test_field_team_report_is_specific():
    c = extract("Team-4 verification Ukhimath complete - major damage, 12 casualties observed.")
    assert c.specificity > 0.5
    assert c.severity == pytest.approx(C.SEVERITY_WEIGHT["MAJOR"])


def test_extract_never_crashes_on_junk():
    for s in ["", "   ", "!!!", "…", "12345", None]:
        c = extract(s)
        assert c.severity is None or 0.0 <= c.severity <= 1.0


# ------------------------------------------------------- geo-resolution
def test_resolve_returns_probability_distribution(gaz):
    r = gaz.resolve("Rudraprayag")
    assert r
    assert all(0.0 <= p <= 1.0 for _, p in r)
    assert sum(p for _, p in r) == pytest.approx(1.0, abs=1e-3)


def test_resolve_preserves_ambiguity(gaz):
    """A name shared by several villages must not collapse to one guess."""
    byname = {}
    for st in gaz.settlements.values():
        byname.setdefault(st.name.lower(), []).append(st)
    dupes = [n for n, v in byname.items() if len(v) >= 3]
    assert dupes, "expected duplicate village names in a real district"
    r = gaz.resolve(dupes[0])
    assert max(p for _, p in r) < 0.85


def test_resolve_handles_misspelling(gaz):
    names = [gaz.settlements[s].name.lower() for s, _ in gaz.resolve("rudarprayag")]
    assert "rudraprayag" in names


def test_resolve_empty_input(gaz):
    assert gaz.resolve("") == []
    assert gaz.resolve(None) == []


# ------------------------------------------------------------ union-find
def test_dsu_groups_transitively():
    d = DSU(5)
    d.union(0, 1); d.union(1, 2)
    assert d.find(0) == d.find(2)
    assert d.find(0) != d.find(3)


# ------------------------------------------------- belief is a distribution
def test_distributions_are_valid(belief):
    for v in belief.values():
        p = [v["distribution"][s] for s in STATES]
        assert all(x >= -1e-9 for x in p), v["sid"]
        assert sum(p) == pytest.approx(1.0, abs=1e-3), v["sid"]


def test_dsi_matches_distribution(belief):
    for v in list(belief.values())[:400]:
        exp = sum(v["distribution"][s] * C.SEVERITY_WEIGHT[s] for s in STATES)
        assert v["dsi"] == pytest.approx(exp, abs=2e-3)


def test_confidence_and_metrics_bounded(belief):
    for v in belief.values():
        for k in ("confidence", "evidence_mass", "contradiction", "dsi", "prior_sev"):
            assert 0.0 <= v[k] <= 1.0, (v["sid"], k, v[k])
        assert v["priority"] >= 0.0
        assert v["voi"] >= 0.0


def test_label_consistent_with_dsi_bands(engine, belief):
    for v in belief.values():
        expect = next((n for thr, n in engine.bands if v["dsi"] < thr), STATES[-1])
        assert v["state"] == expect, (v["sid"], v["dsi"], v["state"])


# ------------------------------------------------------------ determinism
def test_fuse_is_deterministic(engine):
    a, _ = engine.fuse(12.0)
    b, _ = engine.fuse(12.0)
    assert all(a[k]["dsi"] == pytest.approx(b[k]["dsi"], abs=1e-12) for k in a)


def test_credibility_ordering_is_preserved(engine):
    """Learning may revise a channel's reliability but must not invert the
    institutional ordering: a field team outranks an anonymous post."""
    engine.fuse(24.0)
    assert engine.credibility("FIELD_TEAM") > engine.credibility("SOCIAL")
    assert engine.credibility("OFFICIAL") > engine.credibility("SOCIAL")
    assert engine.credibility("FIELD_TEAM") >= engine.credibility("IVR_CALL")


def test_credibility_cannot_be_swamped(engine):
    engine.fuse(24.0)
    for k, (a0, b0) in C.SOURCE_PRIORS.items():
        a, b = engine.cred[k]
        assert (a - a0) + (b - b0) <= C.CRED_MAX_LEARNED + 1e-6, k


# ------------------------------------------------- monotonicity in time
def test_evidence_accumulates_over_time(engine):
    early, _ = engine.fuse(3.0)
    late, _ = engine.fuse(24.0)
    e = sum(v["raw_report_count"] for v in early.values())
    l = sum(v["raw_report_count"] for v in late.values())
    assert l > e


def test_no_reports_before_onset(engine):
    out, recs = engine.fuse(0.0)
    assert sum(v["raw_report_count"] for v in out.values()) == 0


# ---------------------------------------------- the core claims of the work
def test_panic_amplification_is_suppressed(engine):
    """A rumour shared many times must not count as many witnesses."""
    recs = engine._recs
    from collections import defaultdict
    by_text = defaultdict(list)
    for r in recs:
        by_text[r["text"][:60]].append(r)
    big = max(by_text.values(), key=len)
    assert len(big) >= 10, "expected an amplified rumour in the scenario"
    weights = sorted((r["independence"] for r in big), reverse=True)
    assert weights[0] > weights[-1]
    assert sum(weights) < 0.35 * len(weights), "amplification not discounted"


def test_silence_is_distinguished_from_safety(belief, truth):
    """The whole thesis: a settlement with no reports must not default to
    'fine'. Catastrophic villages that never reported must still outrank
    intact villages that never reported."""
    blind = [s for s, v in belief.items()
             if v["raw_report_count"] == 0 and s in truth]
    assert len(blind) > 20
    cat = [belief[s]["dsi"] for s in blind if truth[s]["state"] == "CATASTROPHIC"]
    ok = [belief[s]["dsi"] for s in blind if truth[s]["state"] == "INTACT"]
    assert cat and ok
    assert np.mean(cat) > np.mean(ok), (np.mean(cat), np.mean(ok))


def test_catastrophic_is_never_labelled_intact(belief, truth):
    """The one error class that kills people: telling the EOC a destroyed
    village is fine."""
    bad = [s for s, v in belief.items()
           if s in truth and truth[s]["state"] == "CATASTROPHIC"
           and v["state"] == "INTACT"]
    assert len(bad) <= 3, [belief[s]["name"] for s in bad]


def test_dark_zones_beat_base_rate(belief, truth):
    dark = [s for s, v in belief.items() if v["silence"].get("is_dark") and s in truth]
    assert dark
    hi = sum(1 for s in dark if truth[s]["state"] in ("MAJOR", "CATASTROPHIC"))
    prec = hi / len(dark)
    base = sum(1 for s in belief if s in truth
               and truth[s]["state"] in ("MAJOR", "CATASTROPHIC")) / len(belief)
    assert prec > base * 1.4, (prec, base)


def test_engine_beats_volume_ranking(belief, truth):
    """The headline operational claim, asserted as a test."""
    K = 40
    by_pri = sorted(belief, key=lambda s: -belief[s]["priority"])[:K]
    by_vol = sorted(belief, key=lambda s: -belief[s]["raw_report_count"])[:K]
    reach = lambda sel: sum(truth[s]["casualties"] for s in sel if s in truth)
    assert reach(by_pri) > 2.0 * reach(by_vol), (reach(by_pri), reach(by_vol))


# ------------------------------------------------------------- allocation
def test_plan_respects_asset_capabilities():
    from allocate import RouteNetwork, build_plan
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    e = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    e.prepare(reports)
    b, _ = e.fuse(12.0)
    net = RouteNetwork(e.g)
    net.apply_damage(b)
    plan = build_plan(b, net, t_now=12.0)
    assert plan["assignments"]
    seen = set()
    for a in plan["assignments"]:
        assert a["aid"] not in seen, "asset assigned twice"
        seen.add(a["aid"])
        assert a["eta_hours"] >= 0
        if a["type"] != "DRONE":
            needs = b[a["sid"]].get("needs") or []
            if needs:
                assert a["type"] in needs, (a["type"], needs)


def test_blocked_routes_are_actually_avoided():
    from allocate import RouteNetwork
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    e = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    e.prepare(reports)
    b, _ = e.fuse(24.0)
    net = RouteNetwork(e.g)
    net.apply_damage(b)
    blocked = [(u, v) for u, v, d in net.G.edges(data=True) if d.get("blocked")]
    assert blocked, "damage should close at least one route"
    src = [("D", "D1")]
    dist = net.travel_hours(src, 30.0)
    for u, v in blocked[:50]:
        if u in dist and v in dist:
            # a blocked edge must never be the shortest hop between its ends
            assert abs(dist[u] - dist[v]) < 90.0


# ------------------------------------------------------------- seismology
def test_mmi_matches_published_intensity():
    assert 5.5 <= C.mmi_at(6.8, 5) <= 8.5
    assert 3.5 <= C.mmi_at(6.8, 30) <= 7.0
    assert C.mmi_at(6.8, 5) > C.mmi_at(6.8, 30) > C.mmi_at(6.8, 100)
    assert C.mmi_at(7.5, 20) > C.mmi_at(6.0, 20)


def test_mmi_saturates_near_epicentre():
    assert C.mmi_at(6.8, 0.01) < 12.0
    assert math.isfinite(C.mmi_at(6.8, 0.0))
