"""Scenario engine: generates a multi-hazard event with KNOWN ground truth,
then emits the fragmented, contradictory, partially-absent report stream that
a real District EOC would actually receive.

Why a simulator matters for this problem: the whole difficulty of PS-5 is that
nobody knows the truth. To show that our fusion engine recovers truth from fog,
we need a world where truth exists and is hidden from the engine. The engine
NEVER sees ground_truth - only the reports. We score against it afterwards.

Modelled deliberately:
  * report rate is NON-MONOTONIC in damage. Moderately hit villages generate
    the most chatter; catastrophically hit villages go SILENT because the
    people and the towers are both gone. Naive volume-ranking systems
    therefore rank the worst places as "fine". This is the failure mode
    described in the problem statement.
  * rumour amplification: one false seed spawns dozens of near-duplicate
    social posts, which must not count as dozens of independent witnesses.
  * garbled geography: sat-phone fragments and IVR calls name places loosely.
"""
import json, math, os, random, sys
from dataclasses import dataclass, asdict, field

from gazetteer import get_gazetteer, haversine_km
import config as C

DATA = os.path.join(os.path.dirname(__file__), "data")

# ----------------------------------------------------------------- scenario
# Epicentre in the Mandakini valley - the 2013 Kedarnath disaster corridor.
EPICENTRE = (30.545, 79.055)
MAGNITUDE = 6.1
SCENARIO_BBOX = (30.20, 78.85, 30.80, 79.45)   # s, w, n, e - the affected blocks
DURATION_HRS = 24.0
# How much of real damage is driven by things no free map layer reveals.
# 0 = terrain fully determines damage (and the evaluation becomes circular);
# 0.80 = a deliberately hard, honest setting: terrain ranks damage only
# moderately (measured corr ~0.41), so reports must supply most of the signal.
LATENT_WEIGHT = float(os.environ.get('LATENT_WEIGHT', '0.80'))
# Fraction of substation feeder zones that lose grid power regardless of
# structural damage. High by design: this is the confounder that stops tower
# status from being a free copy of the damage label.
GRID_FAIL_RATE = float(os.environ.get('GRID_FAIL_RATE', '0.72'))


@dataclass
class Report:
    rid: str
    t_hours: float                 # hours since event onset
    source: str                    # SOCIAL / SAT_PHONE / FIELD_TEAM / ...
    source_id: str                 # which handset/handle/team - for independence
    raw_text: str
    claimed_place: str = ""        # fuzzy human phrase, may be wrong
    gps: tuple = None              # (lat, lon) or None
    gps_accuracy_km: float = 0.0
    # hidden from the engine, kept only for scoring:
    truth_sid: str = ""
    is_false: bool = False
    rumour_cluster: str = ""

    def public(self):
        d = asdict(self)
        for k in ("truth_sid", "is_false", "rumour_cluster"):
            d.pop(k)
        return d


# ------------------------------------------------------------- text corpus
# Hinglish / Devanagari-transliterated phrasing, as real citizen reports read.
T_FLOOD = [
    "paani gaon me ghus gaya hai, ghar tak aa gaya",
    "water level rising fast near the bridge, {p} side",
    "{p} me pura bazaar doob gaya hai",
    "flooding reported {p}, people moved to school roof",
    "nadi ka paani khetro me bhar gaya, {p}",
    "severe waterlogging {p}, knee deep",
    "{p} - river has broken the bund, water everywhere",
]
T_COLLAPSE = [
    "makaan gir gaya hai {p} me, log dabe hue hain",
    "building collapse at {p}, at least some trapped",
    "{p} school building has come down, need help",
    "puri gali ke ghar damage ho gaye {p}",
    "structural collapse reported near {p}, sending photos",
]
T_LANDSLIDE = [
    "pahaad se malba gir raha hai {p} ke pass",
    "landslide has blocked the road at {p}",
    "debris flow {p}, road completely cut",
    "{p} ke upar wala slope tut gaya",
]
T_BRIDGE = [
    "pul toot gaya hai {p} ke pass, koi cross nahi kar sakta",
    "bridge at {p} is unsafe, cracks visible",
    "{p} bridge washed away, village cut off",
]
T_MEDICAL = [
    "injured log hain {p} me, doctor chahiye",
    "medical help needed urgently {p}, many injuries",
    "{p} - pregnant woman needs evacuation",
]
T_TRAPPED = [
    "{p} me log chhat pe fase hain, boat bhejo",
    "family trapped on rooftop {p}, water still rising",
    "{p} basement me log fase hue hain",
]
T_SAFE = [
    "{p} is safe, minor water only",
    "{p} me sab theek hai, thoda paani bas",
    "no major damage {p}, road is open",
]
T_PANIC = [
    "SHARE THIS!! dam has broken, {p} completely finished, thousands dead",
    "forward: {p} me 500 log mar gaye, government hiding it",
    "URGENT {p} washed away completely, no survivors, PLEASE SHARE",
    "breaking: entire {p} block submerged, army not coming",
]
T_FRAGMENT = [
    "...{p}... water... need boat... over",
    "this is {p} area... [static]... casualties... send help... over",
    "...cannot confirm... {p}... road gone... [signal lost]",
    "repeat repeat... {p}... people on roof... [static]",
]
T_OFFICIAL = [
    "Block report: {p} - preliminary assessment {sev} damage, verification pending",
    "Tehsil update: {p} affected, {sev} impact, relief camp being opened",
    "Patwari report {p}: {sev} damage to residential structures",
]
T_FIELD = [
    "Team-{n} on ground at {p}. Confirmed {sev} damage. Coordinates attached.",
    "Team-{n}: {p} assessed. {sev}. Access via feeder road, passable.",
    "Team-{n} verification {p} complete - {sev} damage, {c} casualties observed.",
]

HAZ_TEXT = {
    "FLOOD": T_FLOOD, "STRUCTURAL_COLLAPSE": T_COLLAPSE, "LANDSLIDE": T_LANDSLIDE,
    "BRIDGE_FAILURE": T_BRIDGE, "MEDICAL": T_MEDICAL, "TRAPPED": T_TRAPPED,
}


def garble(name, rng):
    """Corrupt a place name the way a bad line or a panicked caller would."""
    if len(name) < 4 or rng.random() > 0.55:
        return name
    n = list(name)
    mode = rng.random()
    if mode < 0.3:
        i = rng.randrange(1, len(n))
        n[i] = rng.choice("aeiou")
    elif mode < 0.55:
        i = rng.randrange(1, len(n) - 1)
        n[i], n[i + 1] = n[i + 1], n[i]
    elif mode < 0.8:
        n = n[:max(3, len(n) - rng.randint(1, 2))]
    else:
        n.insert(rng.randrange(1, len(n)), rng.choice("aeiou"))
    return "".join(n)


class Scenario:
    def __init__(self, seed=20260830):
        self.rng = random.Random(seed)
        self.g = get_gazetteer()
        s, w, n, e = SCENARIO_BBOX
        self.zone = self.g.subset_bbox(s, w, n, e)
        self.truth = {}
        self.reports = []
        self.towers = {}
        self._build_truth()
        self._build_towers()
        self._build_reports()

    def _latent_field(self):
        """Spatially-correlated latent vulnerability that NO free geospatial
        layer can observe: unreinforced masonry stock, building age, local
        drainage, culvert blockage, upstream reservoir release timing.

        Without this the simulator would generate damage as a pure function of
        the same terrain variables the engine's prior uses, and the evaluation
        would be circular - the prior would look near-perfect for free. This
        field is what forces the engine to actually earn its accuracy from
        ground reports.
        """
        rng = self.rng
        centres = [(rng.uniform(30.20, 30.80), rng.uniform(78.85, 79.45),
                    rng.uniform(-1.0, 1.0), rng.uniform(0.04, 0.16))
                   for _ in range(14)]
        field = {}
        for st in self.zone:
            v = 0.0
            for (clat, clon, amp, scale) in centres:
                d2 = ((st.lat - clat) ** 2 + (st.lon - clon) ** 2) / (scale ** 2)
                v += amp * math.exp(-0.5 * d2)
            v += rng.gauss(0, 0.45)          # purely idiosyncratic component
            field[st.sid] = v
        vals = list(field.values())
        mu = sum(vals) / len(vals)
        sd = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5 or 1.0
        return {k: (v - mu) / sd for k, v in field.items()}

    # Target severity mix for the district, held fixed across latent weights so
    # that sweeps compare detection difficulty, not disaster size.
    TARGET_MIX = [("INTACT", 0.32), ("MINOR", 0.34), ("MAJOR", 0.23),
                  ("CATASTROPHIC", 0.11)]

    def _thresholds(self, scores):
        xs = sorted(scores)
        cuts, acc = [], 0.0
        for name, frac in self.TARGET_MIX[:-1]:
            acc += frac
            cuts.append(xs[min(len(xs) - 1, int(acc * len(xs)))])
        return cuts

    # ------------------------------------------------------------- truth
    def _build_truth(self):
        rng = self.rng
        latent = self._latent_field()
        self.latent = latent
        raw = {}
        for st in self.zone:
            d = haversine_km(EPICENTRE[0], EPICENTRE[1], st.lat, st.lon)
            # published MMI attenuation, shared with the engine (config.mmi_at)
            shake = C.shake_norm(C.mmi_at(MAGNITUDE, d))

            # Rain accumulates and drives the flood wave down-valley
            rain = 0.55 + 0.45 * math.exp(-abs(st.lat - 30.6) / 0.35)
            flood = st.flood_exposure * rain * (0.75 + 0.5 * rng.random())
            slide = st.landslide_exposure * (0.45 * shake + 0.55 * rain) * (0.6 + 0.7 * rng.random())
            quake = shake * (0.7 + 0.6 * rng.random())

            physical = max(flood * 0.95, slide * 0.9, quake * 0.85) \
                + 0.25 * min(flood, max(slide, quake))
            # Latent vulnerability that no free map layer reveals. It both
            # scales and shifts the outcome, so terrain alone can neither
            # predict damage nor rule it out. Without this the evaluation
            # would be circular: the engine's terrain prior uses exactly the
            # variables that generated the damage.
            lv = latent[st.sid]
            score = physical * (1.0 + 0.45 * LATENT_WEIGHT * lv) \
                + 0.34 * LATENT_WEIGHT * lv
            score = max(0.0, min(1.35, score))
            primary = max([("FLOOD", flood), ("LANDSLIDE", slide),
                           ("STRUCTURAL_COLLAPSE", quake)], key=lambda x: x[1])[0]
            raw[st.sid] = (score, primary, flood, lv)

        c1, c2, c3 = self._thresholds([v[0] for v in raw.values()])
        for st in self.zone:
            score, primary, flood, lv = raw[st.sid]
            state = ("INTACT" if score < c1 else "MINOR" if score < c2
                     else "MAJOR" if score < c3 else "CATASTROPHIC")

            sev_w = C.SEVERITY_WEIGHT[state]
            # Comms survive damage stochastically; catastrophe usually kills them.
            p_down = min(0.96, sev_w ** 1.4 * 1.15)
            comms_down_at = None
            if rng.random() < p_down:
                comms_down_at = round(rng.uniform(0.0, 1.2 + 3.0 * (1 - sev_w)), 2)

            casualties = int(st.population * C.CASUALTY_RATE[state] *
                             (0.6 + 0.8 * rng.random()))

            self.truth[st.sid] = {
                "latent_vuln": round(lv, 3),
                "sid": st.sid, "name": st.name, "lat": st.lat, "lon": st.lon,
                "population": st.population, "state": state,
                "severity": round(min(1.0, score), 3),
                "primary_hazard": primary,
                "flood_depth_m": round(max(0.0, flood * 3.1), 2),
                "comms_down_at": comms_down_at,
                "casualties": casualties,
                "trapped": int(casualties * rng.uniform(1.5, 4.0)),
            }

    def _build_towers(self):
        """Cell-tower liveness, as DoT/TRAI report it to the SEOC.

        A genuinely different observation from citizen chatter: an unharmed
        village is quiet too, but its tower still answers.

        CRITICAL REALISM: towers do not fail only from structural damage. In a
        real event the grid is cut across whole feeder zones - deliberately, to
        prevent electrocution during flooding (this is exactly the scenario in
        PS-4 of this same problem set) - and every tower then runs on battery
        and diesel until it runs out. So most towers in the district go dark
        whether or not their village was touched.

        Without this confounder, tower-down would be a near-perfect proxy for
        the damage label (measured: P(down)=0.06 for INTACT vs 0.94 for
        CATASTROPHIC), and the silence engine would be scoring against a
        leaked answer rather than solving the problem.
        """
        rng = self.rng
        # Feeder zones: contiguous grid areas fed by the same substation.
        zone_fail = {}
        for st in self.zone:
            z = (round(st.lat * 12), round(st.lon * 12))
            if z not in zone_fail:
                zone_fail[z] = (rng.uniform(0.3, 5.0)
                                if rng.random() < GRID_FAIL_RATE else None)
        for st in self.zone:
            if st.population < 250 and st.connectivity < 0.4:
                continue                      # no local BTS; shares a distant one
            tr = self.truth[st.sid]
            z = (round(st.lat * 12), round(st.lon * 12))
            gf = zone_fail.get(z)

            cands = []
            if tr["comms_down_at"] is not None:
                cands.append((tr["comms_down_at"], "structural"))
            if gf is not None:
                # battery + diesel backup buys hours, not days
                backup = rng.uniform(2.5, 14.0)
                cands.append((gf + backup, "power"))
            if not cands:
                self.towers[st.sid] = {
                    "sid": st.sid, "tower_id": "BTS-%s" % st.sid[1:7],
                    "down_at": None, "restored_at": None, "_cause": None}
                continue

            down_at, cause = min(cands)
            restored_at = None
            if cause == "power" and rng.random() < 0.45:
                restored_at = down_at + rng.uniform(6.0, 18.0)   # refuelled
                # ...unless the site is also structurally gone
                if tr["comms_down_at"] is not None and tr["comms_down_at"] <= restored_at:
                    restored_at = None
            self.towers[st.sid] = {
                "sid": st.sid, "tower_id": "BTS-%s" % st.sid[1:7],
                "down_at": round(down_at, 2),
                "restored_at": round(restored_at, 2) if restored_at else None,
                "_cause": cause,
            }

    def tower_alive(self, sid, t):
        tw = self.towers.get(sid)
        if not tw or tw["down_at"] is None:
            return True
        if t < tw["down_at"]:
            return True
        if tw["restored_at"] is not None and t >= tw["restored_at"]:
            return True
        return False

    def comms_up(self, sid, t):
        c = self.truth[sid]["comms_down_at"]
        return c is None or t < c

    # ----------------------------------------------------------- reports
    def _emit(self, t, source, source_id, text, claimed, sid,
              gps=None, acc=0.0, false=False, cluster=""):
        self.reports.append(Report(
            rid="R%05d" % len(self.reports), t_hours=round(t, 3), source=source,
            source_id=source_id, raw_text=text, claimed_place=claimed,
            gps=gps, gps_accuracy_km=acc, truth_sid=sid,
            is_false=false, rumour_cluster=cluster))

    def _build_reports(self):
        rng = self.rng
        g = self.g

        for st in self.zone:
            tr = self.truth[st.sid]
            sev = C.SEVERITY_WEIGHT[tr["state"]]

            # -------- Report propensity is NON-MONOTONIC in damage ----------
            # Nobody reports an intact village; a devastated one cannot report.
            # Peak chatter comes from MAJOR damage with surviving comms.
            propensity = math.exp(-((sev - 0.55) ** 2) / 0.075)
            base = (st.population / 260.0) * st.connectivity * propensity
            n_social = int(rng.gauss(base * 6.0, base * 1.8))
            n_ivr = int(rng.gauss(base * 1.7, base * 0.7))

            haz = tr["primary_hazard"]
            pool = HAZ_TEXT.get(haz, T_FLOOD)

            # ---- citizen social media -------------------------------------
            for _ in range(max(0, n_social)):
                t = rng.uniform(0.1, DURATION_HRS)
                if not self.comms_up(st.sid, t):
                    continue
                tmpl = rng.choice(pool if sev > 0.3 else T_SAFE)
                place = garble(st.name, rng) if rng.random() < 0.35 else st.name
                self._emit(t, "SOCIAL", "u%d" % rng.randrange(9999),
                           tmpl.format(p=place), place, st.sid,
                           gps=(st.lat + rng.gauss(0, .01), st.lon + rng.gauss(0, .01))
                           if rng.random() < 0.25 else None, acc=2.5)

            # ---- IVR / 112 calls ------------------------------------------
            for _ in range(max(0, n_ivr)):
                t = rng.uniform(0.05, DURATION_HRS)
                if not self.comms_up(st.sid, t):
                    continue
                tmpl = rng.choice(T_TRAPPED if sev > 0.6 and rng.random() < .5
                                  else (pool if sev > 0.3 else T_SAFE))
                place = garble(st.name, rng) if rng.random() < 0.5 else st.name
                self._emit(t, "IVR_CALL", "c%d" % rng.randrange(99999),
                           tmpl.format(p=place), place, st.sid)

            # ---- satellite phone: rare, fragmentary, no GPS ---------------
            if rng.random() < 0.05 + 0.13 * sev:
                for _ in range(rng.randint(1, 3)):
                    t = rng.uniform(0.5, DURATION_HRS)
                    place = garble(st.name, rng)
                    self._emit(t, "SAT_PHONE", "sat%d" % rng.randrange(60),
                               rng.choice(T_FRAGMENT).format(p=place), place, st.sid)

            # ---- ham radio relay ------------------------------------------
            if rng.random() < 0.04 + 0.10 * sev:
                t = rng.uniform(1.0, DURATION_HRS)
                self._emit(t, "HAM_RADIO", "vu2%s" % rng.choice("abcdefgh"),
                           rng.choice(pool).format(p=st.name), st.name, st.sid,
                           gps=(st.lat, st.lon), acc=1.0)

            # ---- official channel: delayed, coarse, sometimes stale --------
            if rng.random() < 0.10 + 0.14 * st.connectivity:
                t = rng.uniform(3.0, DURATION_HRS)
                lag_state = tr["state"]
                if rng.random() < 0.3:      # stale: reports an earlier, milder state
                    idx = max(0, C.DAMAGE_STATES.index(lag_state) - 1)
                    lag_state = C.DAMAGE_STATES[idx]
                self._emit(t, "OFFICIAL", "blk%d" % rng.randrange(12),
                           rng.choice(T_OFFICIAL).format(p=st.name, sev=lag_state.lower()),
                           st.name, st.sid, gps=(st.lat, st.lon), acc=0.5)

            # ---- field team: accurate, GPS-tagged, but only where reachable
            reachable = st.connectivity > 0.45 and sev < 0.85
            if reachable and rng.random() < 0.11:
                t = rng.uniform(4.0, DURATION_HRS)
                self._emit(t, "FIELD_TEAM", "T%d" % rng.randrange(1, 15),
                           rng.choice(T_FIELD).format(
                               n=rng.randint(1, 14), p=st.name,
                               sev=tr["state"].lower(), c=tr["casualties"]),
                           st.name, st.sid, gps=(st.lat, st.lon), acc=0.05)

        self._inject_rumours()
        self.reports.sort(key=lambda r: r.t_hours)
        for i, r in enumerate(self.reports):
            r.rid = "R%05d" % i

    def _inject_rumours(self):
        """Panic amplification: a handful of false seeds, each massively
        re-shared. Volume must not be mistaken for corroboration."""
        rng = self.rng
        safe = [s for s in self.zone if self.truth[s.sid]["state"] in ("INTACT", "MINOR")]
        rng.shuffle(safe)
        for k, st in enumerate(safe[:11]):
            cluster = "RUM%d" % k
            seed_t = rng.uniform(0.5, 8.0)
            seed_text = rng.choice(T_PANIC).format(p=st.name)
            n_share = rng.randint(35, 120)
            for j in range(n_share):
                t = seed_t + abs(rng.gauss(0, 2.2))
                if t > DURATION_HRS:
                    continue
                txt = seed_text
                if rng.random() < 0.4:       # slight mutation as it spreads
                    txt = txt.replace("!!", "!").replace("PLEASE SHARE", "pls share")
                self._emit(t, "SOCIAL", "u%d" % rng.randrange(9999), txt,
                           st.name, st.sid, false=True, cluster=cluster)

        # Genuine contradictions: two channels disagreeing about a real place
        hit = [s for s in self.zone if self.truth[s.sid]["state"] in ("MAJOR", "CATASTROPHIC")]
        rng.shuffle(hit)
        for st in hit[:22]:
            t = rng.uniform(2.0, 14.0)
            self._emit(t, "SOCIAL", "u%d" % rng.randrange(9999),
                       rng.choice(T_SAFE).format(p=st.name), st.name, st.sid, false=True)

    # ------------------------------------------------------------- export
    def save(self):
        os.makedirs(DATA, exist_ok=True)
        with open(os.path.join(DATA, "ground_truth.json"), "w", encoding="utf-8") as f:
            json.dump(self.truth, f)
        with open(os.path.join(DATA, "reports.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.reports], f, ensure_ascii=False)
        # what the engine is allowed to see of the telecom feed
        pub = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in self.towers.items()}
        with open(os.path.join(DATA, "towers.json"), "w", encoding="utf-8") as f:
            json.dump(pub, f)

    def summary(self):
        from collections import Counter
        cs = Counter(v["state"] for v in self.truth.values())
        src = Counter(r.source for r in self.reports)
        silent = [v for v in self.truth.values() if v["comms_down_at"] is not None]
        tdown = sum(1 for v in self.towers.values() if v["down_at"] is not None)
        tpow = sum(1 for v in self.towers.values() if v["_cause"] == "power")
        return {"settlements": len(self.truth), "states": dict(cs),
                "towers": len(self.towers), "towers_down": tdown,
                "towers_down_power_only": tpow,
                "reports": len(self.reports), "by_source": dict(src),
                "comms_down": len(silent),
                "false_reports": sum(1 for r in self.reports if r.is_false)}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sc = Scenario()
    sc.save()
    s = sc.summary()
    print(json.dumps(s, indent=2))
    print("\nGround truth severity mix:", s["states"])
    # The headline pathology this project exists to fix:
    from collections import defaultdict
    vol = defaultdict(int)
    for r in sc.reports:
        vol[r.truth_sid] += 1
    cat = [sc.truth[k]["name"] for k in sc.truth
           if sc.truth[k]["state"] == "CATASTROPHIC"]
    zero = [sc.truth[k]["name"] for k in sc.truth
            if sc.truth[k]["state"] == "CATASTROPHIC" and vol[k] == 0]
    print("\nCATASTROPHIC settlements: %d" % len(cat))
    print("...of which emit ZERO reports (invisible to volume ranking): %d" % len(zero))
    print("Examples:", zero[:8])
