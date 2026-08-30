"""Asset allocation and recon tasking.

Two distinct queues, deliberately separated - this is the operational heart
of the answer to PS-5:

  RESCUE QUEUE  scarce physical assets (boats, excavators, medical teams) are
                committed only where belief is strong enough to justify the
                trip. Ranked by expected lives saved per asset-hour, over a
                road network that has itself been damaged.

  RECON QUEUE   cheap, fast assets (UAVs, and the one field team you can
                spare) are sent where UNCERTAINTY is most expensive - highest
                stakes x lowest confidence. Their job is not to rescue but to
                collapse uncertainty so the next allocation round is right.

Sending an excavator on an inference is how you waste it. Sending a drone on
an inference is how you earn the right to send the excavator.
"""
import heapq, json, math, os
from collections import defaultdict

import networkx as nx

import config as C
from gazetteer import get_gazetteer, haversine_km

DATA = os.path.join(os.path.dirname(__file__), "data")

# Staging bases: real towns in the district with road access and open ground.
DEPOTS = [
    {"did": "D1", "name": "Rudraprayag HQ",  "lat": 30.284, "lon": 78.981},
    {"did": "D2", "name": "Gopeshwar Base",  "lat": 30.407, "lon": 79.318},
    {"did": "D3", "name": "Karnaprayag Hub", "lat": 30.256, "lon": 79.215},
    {"did": "D4", "name": "Guptkashi Fwd",   "lat": 30.527, "lon": 79.055},
]

DEFAULT_FLEET = [
    ("BOAT", "D1", 3), ("BOAT", "D3", 2), ("BOAT", "D4", 2),
    ("EXCAVATOR", "D1", 2), ("EXCAVATOR", "D2", 2), ("EXCAVATOR", "D3", 1),
    ("MEDICAL", "D1", 3), ("MEDICAL", "D2", 2), ("MEDICAL", "D4", 2),
    ("DRONE", "D1", 2), ("DRONE", "D2", 2), ("DRONE", "D4", 2),
]


def _key(lat, lon):
    return (round(lat, 4), round(lon, 4))


class RouteNetwork:
    """Road graph from OSM, degraded by what we now believe about the ground."""

    CACHE = os.path.join(DATA, "roadgraph.pkl")

    def __init__(self, gaz=None, use_cache=True):
        self.g = gaz or get_gazetteer()
        self.G = nx.Graph()
        if use_cache and os.path.exists(self.CACHE):
            try:
                import pickle
                with open(self.CACHE, "rb") as f:
                    self.G, self.settlement_node = pickle.load(f)
                return
            except Exception:
                self.G = nx.Graph()
        self._build()
        if use_cache:
            try:
                import pickle
                with open(self.CACHE, "wb") as f:
                    pickle.dump((self.G, self.settlement_node), f)
            except Exception:
                pass

    def _build(self):
        SPEED = {"trunk": 45, "primary": 40, "secondary": 32, "tertiary": 24}
        for w in self.g.roads:
            cls = w.get("cls") or "tertiary"
            spd = SPEED.get(cls, 24)
            pts = w["coords"]
            for i in range(len(pts) - 1):
                a, b = _key(*pts[i]), _key(*pts[i + 1])
                if a == b:
                    continue
                d = haversine_km(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                if d <= 0:
                    continue
                prev = self.G.get_edge_data(a, b)
                t = d / spd
                if prev is None or t < prev["base_h"]:
                    self.G.add_edge(a, b, km=d, base_h=t, cls=cls,
                                    bridge=bool(w.get("bridge")))
        # attach every settlement to the nearest road node; hill villages that
        # are far from any road get an explicit foot/air access penalty
        nodes = list(self.G.nodes)
        self.node_index = nodes
        self.settlement_node = {}
        if not nodes:
            return
        buckets = defaultdict(list)
        for n in nodes:
            buckets[(round(n[0], 1), round(n[1], 1))].append(n)
        for st in self.g.settlements.values():
            best, bd = None, 1e9
            for dl in (0.0, 0.1, -0.1, 0.2, -0.2):
                for dn in (0.0, 0.1, -0.1, 0.2, -0.2):
                    for n in buckets.get((round(st.lat + dl, 1),
                                          round(st.lon + dn, 1)), ()):
                        d = haversine_km(st.lat, st.lon, n[0], n[1])
                        if d < bd:
                            best, bd = n, d
                if best is not None and bd < 2.0:
                    break
            if best is None:
                continue
            sn = ("S", st.sid)
            # off-road access is slow: 4 km/h on foot over broken hill tracks
            self.G.add_edge(sn, best, km=bd, base_h=bd / 4.0, cls="access",
                            bridge=False)
            self.settlement_node[st.sid] = sn
        for d in DEPOTS:
            best, bd = None, 1e9
            for n in nodes:
                if abs(n[0] - d["lat"]) > 0.25 or abs(n[1] - d["lon"]) > 0.25:
                    continue
                dd = haversine_km(d["lat"], d["lon"], n[0], n[1])
                if dd < bd:
                    best, bd = n, dd
            if best is not None:
                self.G.add_edge(("D", d["did"]), best, km=bd, base_h=bd / 30.0,
                                cls="access", bridge=False)

    # ------------------------------------------------------------ damage
    def apply_damage(self, belief):
        """Degrade edge traversal times using current belief about the ground.

        An edge near a settlement we believe is landslide-hit or deeply
        flooded becomes slow or impassable. Crucially this uses BELIEF, with
        its confidence - we do not close a highway on one unverified tweet.
        """
        infl = []
        for sid, v in belief.items():
            # Use the calibrated probability that this place is severely hit,
            # not a raw severity index. A probability keeps its meaning when
            # the belief scale is recalibrated; a hand-tuned index does not,
            # and silently stopped closing any road when we recalibrated.
            d = v["distribution"]
            p_sev = d.get("MAJOR", 0.0) + d.get("CATASTROPHIC", 0.0)
            if p_sev < C.ROUTE_DEGRADE_P:
                continue
            infl.append((v["lat"], v["lon"], p_sev, set(v.get("hazards", []))))
        for a, b, data in self.G.edges(data=True):
            if isinstance(a[0], str) and a[0] in ("S", "D"):
                mlat, mlon = None, None
            if not (isinstance(a[0], float) and isinstance(b[0], float)):
                data["mult"] = 1.0
                data["blocked"] = False
                continue
            mlat, mlon = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            worst, haz = 0.0, set()
            for (lat, lon, risk, hz) in infl:
                if abs(lat - mlat) > 0.03 or abs(lon - mlon) > 0.03:
                    continue
                d = haversine_km(mlat, mlon, lat, lon)
                if d < C.ROUTE_INFLUENCE_KM and risk > worst:
                    worst, haz = risk, hz
            mult, blocked = 1.0, False
            if worst > 0:
                if data.get("bridge") and worst > C.BRIDGE_BLOCK_P:
                    blocked = True            # damaged span: do not risk it
                elif worst > C.ROAD_BLOCK_P:
                    blocked = True
                else:
                    mult = 1.0 + 4.5 * worst  # debris, water, single-lane
            data["mult"] = mult
            data["blocked"] = blocked
            data["hazard"] = ",".join(sorted(haz)) if haz else ""

    def travel_hours(self, sources, speed_kmph, max_h=14.0):
        """Multi-source Dijkstra over the degraded network."""
        dist = {}
        pq = []
        for s in sources:
            if s in self.G:
                dist[s] = 0.0
                heapq.heappush(pq, (0.0, s))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18) + 1e-12 or d > max_h:
                continue
            for v, data in self.G[u].items():
                if data.get("blocked"):
                    continue
                base = data["km"] / max(4.0, speed_kmph) if data["cls"] == "access" \
                    else data["km"] / max(6.0, speed_kmph * _cls_factor(data["cls"]))
                nd = d + base * data.get("mult", 1.0)
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist


def _cls_factor(cls):
    return {"trunk": 1.0, "primary": 0.92, "secondary": 0.78,
            "tertiary": 0.60, "access": 0.25}.get(cls, 0.6)


# ===================================================================== plan
def survival_factor(t_hours):
    """Fraction of currently-savable casualties still savable after t hours.
    Steep early decline - the first day is what matters."""
    return math.exp(-1.9 * max(0.0, t_hours) / C.GOLDEN_WINDOW_HRS)


def build_plan(belief, net, fleet=None, t_now=0.0, min_confidence=0.30):
    """Greedy marginal-utility assignment.

    At each step we take the (asset, settlement) pair with the highest
    expected lives saved, accounting for travel time on the degraded network
    and for how much of the need that asset type can actually meet. Greedy is
    deliberate: it is transparent, it is instant, and an EOC officer can
    override any single line of it without the rest collapsing.
    """
    fleet = fleet or DEFAULT_FLEET
    assets = []
    for atype, did, n in fleet:
        for i in range(n):
            assets.append({"aid": "%s-%s-%d" % (atype, did, i + 1),
                           "type": atype, "depot": did})

    depot_pos = {d["did"]: ("D", d["did"]) for d in DEPOTS}
    # travel times per asset type (speeds differ), computed once per depot
    ttime = {}
    for atype, spec in C.ASSET_TYPES.items():
        for d in DEPOTS:
            src = depot_pos[d["did"]]
            ttime[(atype, d["did"])] = net.travel_hours([src], spec["speed_kmph"])

    # ---- candidate demand ------------------------------------------------
    rescue_c, recon_c = [], []
    for sid, v in belief.items():
        if v["priority"] <= 0.05:
            continue
        node = net.settlement_node.get(sid)
        if node is None:
            continue
        d_ = v["distribution"]
        p_sev = d_.get("MAJOR", 0.0) + d_.get("CATASTROPHIC", 0.0)
        if p_sev >= C.RESCUE_MIN_P_SEVERE and (
                v["confidence"] >= C.RESCUE_MIN_CONF
                or p_sev >= C.RESCUE_CONF_OVERRIDE_P):
            rescue_c.append((sid, v, node))
        # anything high-stakes and uncertain is worth looking at
        if v["voi"] > 0.05:
            recon_c.append((sid, v, node))

    assigned, unreachable = [], []
    remaining = {sid: v["priority"] for sid, v, _ in rescue_c}
    served = defaultdict(set)

    physical = [a for a in assets if a["type"] != "DRONE"]
    drones = [a for a in assets if a["type"] == "DRONE"]

    for _ in range(len(physical)):
        best = None
        for a in physical:
            if a.get("used"):
                continue
            spec = C.ASSET_TYPES[a["type"]]
            tt = ttime[(a["type"], a["depot"])]
            for sid, v, node in rescue_c:
                if a["type"] in served[sid]:
                    continue          # already covered by same capability
                need = set(v.get("needs") or [])
                if need and a["type"] not in need:
                    continue
                hrs = tt.get(node)
                if hrs is None:
                    continue
                # expected lives saved: unmet need x reachability in time
                cover = min(1.0, spec["capacity"] / max(1.0, v["population"] * 0.35))
                val = (remaining.get(sid, 0.0) * cover
                       * survival_factor(t_now + hrs)
                       * (0.70 + 0.30 * v["confidence"]))
                score = val / max(0.4, hrs)          # lives saved per hour
                if best is None or score > best[0]:
                    best = (score, a, sid, v, hrs, val)
        if best is None:
            break
        score, a, sid, v, hrs, val = best
        a["used"] = True
        served[sid].add(a["type"])
        remaining[sid] = max(0.0, remaining.get(sid, 0.0) - val)
        assigned.append({
            "aid": a["aid"], "type": a["type"], "depot": a["depot"],
            "sid": sid, "name": v["name"], "lat": v["lat"], "lon": v["lon"],
            "eta_hours": round(hrs, 2),
            "expected_lives_saved": round(val, 2),
            "value_per_hour": round(score, 3),
            "state": v["state"], "confidence": v["confidence"],
            "population": v["population"],
            "rationale": _why(v, a["type"], hrs),
        })

    # ---- recon tasking: buy information where it is most expensive -------
    recon_c.sort(key=lambda x: -x[1]["voi"])
    taken = set()
    for a in drones:
        tt = ttime[("DRONE", a["depot"])]
        best = None
        for sid, v, node in recon_c:
            if sid in taken:
                continue
            hrs = tt.get(node)
            if hrs is None or hrs > 3.0:      # UAV endurance
                continue
            score = v["voi"] / max(0.25, hrs)
            if best is None or score > best[0]:
                best = (score, sid, v, hrs)
        if best is None:
            continue
        score, sid, v, hrs = best
        taken.add(sid)
        a["used"] = True
        assigned.append({
            "aid": a["aid"], "type": "DRONE", "depot": a["depot"],
            "sid": sid, "name": v["name"], "lat": v["lat"], "lon": v["lon"],
            "eta_hours": round(hrs, 2), "expected_lives_saved": 0.0,
            "value_per_hour": round(score, 3), "state": v["state"],
            "confidence": v["confidence"], "population": v["population"],
            "rationale": _why_recon(v),
        })

    # ---- what we could not reach at all ---------------------------------
    reach = ttime[("MEDICAL", "D1")]
    for sid, v, node in rescue_c:
        if any(x["sid"] == sid for x in assigned):
            continue
        if node not in reach:
            unreachable.append({
                "sid": sid, "name": v["name"], "lat": v["lat"], "lon": v["lon"],
                "population": v["population"], "dsi": v["dsi"],
                "priority": v["priority"],
                "reason": "no passable surface route - air asset required",
            })
    unreachable.sort(key=lambda x: -x["priority"])

    idle = [a["aid"] for a in assets if not a.get("used")]
    return {
        "assignments": sorted(assigned, key=lambda x: -x["value_per_hour"]),
        "unreachable": unreachable[:25],
        "idle_assets": idle,
        "depots": DEPOTS,
        "summary": {
            "assets_total": len(assets),
            "assets_committed": len(assigned),
            "expected_lives_saved": round(
                sum(x["expected_lives_saved"] for x in assigned), 1),
            "recon_sorties": sum(1 for x in assigned if x["type"] == "DRONE"),
            "unreachable_settlements": len(unreachable),
        },
    }


def _why(v, atype, hrs):
    bits = []
    d = v["distribution"]
    p_sev = d["MAJOR"] + d["CATASTROPHIC"]
    # Report P(severe), not P(label): the triage label comes from calibrated
    # DSI bands, so quoting the single-state probability next to it reads as a
    # contradiction ("CATASTROPHIC (p=0.03)").
    bits.append("%s, P(severe)=%.0f%%" % (v["state"], 100 * p_sev))
    if v["silence"].get("is_dark"):
        bits.append("comms blackout %s" % v["silence"].get("reason", ""))
    if v.get("trapped_est"):
        bits.append("~%d reported trapped" % v["trapped_est"])
    if v["contradiction"] > 0.4:
        bits.append("conflicting reports (%.0f%%)" % (100 * v["contradiction"]))
    bits.append("pop %d" % v["population"])
    bits.append("ETA %.1fh" % hrs)
    return "; ".join(bits)


def _why_recon(v):
    return ("stakes %d people at DSI %.2f but confidence only %.2f%s"
            % (v["population"], v["dsi"], v["confidence"],
               "; contradictory reports" if v["contradiction"] > 0.4 else ""))


if __name__ == "__main__":
    import sys, time
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from fusion import FusionEngine
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX

    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))
    eng = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    eng.prepare(reports)
    belief, _ = eng.fuse(12.0)

    t0 = time.time()
    net = RouteNetwork(eng.g)
    print("road graph: %d nodes, %d edges (%.1fs)"
          % (net.G.number_of_nodes(), net.G.number_of_edges(), time.time() - t0))
    net.apply_damage(belief)
    blocked = sum(1 for _, _, d in net.G.edges(data=True) if d.get("blocked"))
    slowed = sum(1 for _, _, d in net.G.edges(data=True) if d.get("mult", 1) > 1.5)
    print("network degraded: %d edges blocked, %d slowed" % (blocked, slowed))

    plan = build_plan(belief, net, t_now=12.0)
    print("\n%s" % json.dumps(plan["summary"], indent=1))
    print("\n%-14s %-18s %6s %8s  %s" % ("ASSET", "TARGET", "ETA", "LIVES", "WHY"))
    for a in plan["assignments"][:16]:
        tr = truth.get(a["sid"], {}).get("state", "?")
        print("%-14s %-18s %5.1fh %8.1f  [truth=%s] %s"
              % (a["aid"], a["name"][:18], a["eta_hours"],
                 a["expected_lives_saved"], tr[:6], a["rationale"][:60]))
