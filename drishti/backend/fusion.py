"""DRISHTI fusion engine - the core of the solution to PS-5.

Design thesis: in the first 24 hours you cannot establish truth. You can only
maintain a calibrated, auditable BELIEF, and act on expected loss under that
belief. So every settlement carries a full distribution over damage states
plus an explicit statement of how much of that belief is actually supported
by evidence versus assumed from terrain physics.

Five mechanisms, in order:

  1. GEO-RESOLUTION      fuzzy place phrase -> distribution over settlements
                          (ambiguity preserved, never collapsed to a guess)
  2. CORROBORATION GRAPH near-duplicate reports are clustered, and repeated
                          messages from one channel are discounted, so panic
                          amplification cannot masquerade as corroboration
  3. EVIDENCE FUSION     Dempster-Shafer combination of simple support
                          functions, in closed form, in log space
  4. SILENCE AS EVIDENCE the differentiator. Absence of expected reports is
                          converted into positive evidence of catastrophe
  5. TRIAGE              expected unassisted casualties, and separately the
                          value of information for recon tasking
"""
import math, os, json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C
from gazetteer import get_gazetteer, haversine_km
from extract import extract

STATES = C.DAMAGE_STATES
SEVW = np.array([C.SEVERITY_WEIGHT[s] for s in STATES])
NS = len(STATES)


# ---------------------------------------------------------------- utilities
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ================================================================= ENGINE
class FusionEngine:
    def __init__(self, gaz=None, epicentre=None, magnitude=None, aoi=None):
        self.g = gaz or get_gazetteer()
        self.epicentre = epicentre
        self.magnitude = magnitude or 6.0
        # Area of interest = the administrative blocks this EOC is responsible
        # for. Reports may name places anywhere (that ambiguity is the point),
        # but triage output is scoped to the district. (s, w, n, e)
        self.aoi = aoi
        self._aoi_sids = None
        if aoi:
            s_, w_, n_, e_ = aoi
            self._aoi_sids = {st.sid for st in self.g.subset_bbox(s_, w_, n_, e_)}
        # online source reliability, Beta(alpha, beta) per channel
        self.cred = {k: list(v) for k, v in C.SOURCE_PRIORS.items()}
        # temperature from calibrate.py, fitted on a held-out scenario seed
        self.temperature = 1.0
        self.bands = list(C.DSI_BANDS)
        _cp = os.path.join(os.path.dirname(__file__), "data", "calibration.json")
        if os.path.exists(_cp):
            try:
                _c = json.load(open(_cp))
                self.temperature = float(_c["temperature"])
                if _c.get("dsi_bands"):
                    _b = list(_c["dsi_bands"]) + [1.01]
                    self.bands = list(zip(_b, C.DAMAGE_STATES))
            except Exception:
                pass
        self.towers = {}
        _tp = os.path.join(os.path.dirname(__file__), "data", "towers.json")
        if os.path.exists(_tp):
            self.towers = json.load(open(_tp, encoding="utf-8"))
        self._resolve_cache = {}
        self._recs = None
        self._last = None
        # Ablation switches - each corresponds to one contributed mechanism,
        # so the evaluation harness can measure what each one is worth.
        self.flags = {"silence": True, "independence": True,
                      "credibility": True, "prior": True}

    def _resolve_cached(self, phrase, hint):
        key = (phrase, hint)
        v = self._resolve_cache.get(key)
        if v is None:
            v = self.g.resolve(phrase, hint_latlon=hint)
            self._resolve_cache[key] = v
        return v

    def in_aoi(self, sid):
        return self._aoi_sids is None or sid in self._aoi_sids

    # ------------------------------------------------------- credibility
    def credibility(self, source):
        a, b = self.cred.get(source, C.DEFAULT_PRIOR)
        return a / (a + b)

    def _update_credibility(self, records, posterior, chan_mass):
        """Learn which channels have been right, WITHOUT letting a channel
        vouch for itself.

        Naive version of this is a self-confirmation loop: social media is the
        loudest channel, so in villages where only social reported, the
        consensus IS social, social agrees with it, and its credibility climbs.
        Measured: the SOCIAL prior of 0.29 drifted to 0.81, destroying the
        whole point of source weighting.

        So a report only scores against consensus that is independently
        supported by OTHER channels, and the total learnable evidence per
        channel is bounded so a prior can be revised but never swamped.
        """
        self.cred = {k: list(v) for k, v in C.SOURCE_PRIORS.items()}
        budget = {k: 0.0 for k in self.cred}
        for r in records:
            sid = r["top_sid"]
            if not sid or sid not in posterior or r["claim_sev"] is None:
                continue
            post = posterior[sid]
            if post["confidence"] < C.CRED_MIN_CONFIDENCE:
                continue
            cm = chan_mass.get(sid, {})
            other = sum(m for c, m in cm.items() if c != r["source"])
            if other < C.CRED_MIN_INDEPENDENT_MASS:
                continue                       # only this channel spoke here
            w_pending = 0.12 * r["independence"]
            if budget.get(r["source"], 0.0) + w_pending > C.CRED_MAX_LEARNED:
                continue
            err = abs(r["claim_sev"] - post["dsi"])
            w = 0.12 * r["independence"]
            self.cred.setdefault(r["source"], list(C.DEFAULT_PRIOR))
            if err < 0.22:
                self.cred[r["source"]][0] += w
            elif err > 0.45:
                self.cred[r["source"]][1] += w
            else:
                continue
            budget[r["source"]] = budget.get(r["source"], 0.0) + w

    # ---------------------------------------------------- physical prior
    def hazard_prior(self, st, rain_factor=1.0):
        """What terrain physics alone predicts, before anyone speaks.

        Uses only data available at t=0 from free public feeds: USGS epicentre
        and magnitude, OSM river geometry, Open-Meteo elevation and rainfall.
        This is what keeps a silent village from defaulting to 'fine'.
        """
        shake = 0.0
        if self.epicentre:
            d = haversine_km(self.epicentre[0], self.epicentre[1], st.lat, st.lon)
            shake = C.shake_norm(C.mmi_at(self.magnitude, d))
        flood = st.flood_exposure * rain_factor
        slide = st.landslide_exposure * (0.45 * shake + 0.55 * rain_factor)
        sev = min(1.0, max(flood * 0.9, slide * 0.85, shake * 0.8))
        return sev

    @staticmethod
    def _kernel(sev, conf):
        """Turn an asserted severity into a distribution over damage states.
        Low confidence -> broad, non-committal kernel."""
        bw = 0.42 - 0.26 * max(0.0, min(1.0, conf))    # 0.16 .. 0.42
        d = (SEVW - sev) / bw
        k = np.exp(-0.5 * d * d)
        s = k.sum()
        return k / s if s > 1e-12 else np.full(NS, 1.0 / NS)

    # ===================================================== main entrypoint
    def prepare(self, reports):
        """One-time pass: extract claims, resolve geography, build the
        corroboration graph. Independence weights are assigned by time-rank
        within each cluster, so any time-prefix of the stream reuses these
        weights unchanged - which is what makes the time scrubber instant."""
        g = self.g
        recs = []
        for r in reports:
            cl = extract(r["raw_text"], r["source"])
            hint = tuple(r["gps"]) if r.get("gps") else None
            dist = self._resolve_cached(r.get("claimed_place") or r["raw_text"], hint)
            if not dist and hint:
                nearby = g.near(hint[0], hint[1], 4.0)[:3]
                if nearby:
                    tot = sum(1.0 / (1 + d) for _, d in nearby)
                    dist = [(sid, (1.0 / (1 + d)) / tot) for sid, d in nearby]
            if not dist:
                continue
            recs.append({
                "rid": r["rid"], "t": r["t_hours"], "source": r["source"],
                "source_id": r.get("source_id", ""), "text": r["raw_text"],
                "geo": dist, "top_sid": dist[0][0],
                "claim_sev": cl.severity, "sev_conf": cl.sev_conf,
                "hazards": cl.hazards, "safe": cl.is_safe_claim,
                "panic": cl.panic_score, "spec": cl.specificity,
                "trapped": cl.trapped, "independence": 1.0, "cluster": -1,
            })

        # ---------- 2. corroboration graph -------------------------------
        self._cluster(recs)
        self._independence(recs)
        self._recs = recs
        return recs

    def run(self, reports, t_now, rain_factor=1.0):
        if self._recs is None:
            self.prepare(reports)
        return self.fuse(t_now, rain_factor)

    def fuse(self, t_now, rain_factor=1.0):
        """Fuse the time-prefix of the prepared stream up to t_now.

        Pure with respect to engine state: credibility is re-derived from the
        priors every call, so repeated requests for the same timestep return
        identical results instead of drifting.
        """
        self.cred = {k: list(v) for k, v in C.SOURCE_PRIORS.items()}
        recs = [r for r in self._recs if r["t"] <= t_now]

        # ---------- 3. evidence fusion (Dempster-Shafer, closed form) -----
        # For simple support functions (mass on one singleton + Theta), the
        # n-fold Dempster combination reduces to:
        #     q(s) = PROD_i (m_i(s) + m_i(Theta))  -  PROD_i m_i(Theta)
        #     q(Theta) = PROD_i m_i(Theta)
        # Computed in log space, then normalised, so 10^4 reports never
        # underflow.
        logsum = defaultdict(lambda: np.zeros(NS))   # sum log(m_i(s)+m_i(Th))
        logth = defaultdict(float)                   # sum log m_i(Th)
        obs = defaultdict(float)                     # effective observations
        mass_low = defaultdict(float)                # evidence asserting "safe"
        mass_high = defaultdict(float)               # evidence asserting "severe"
        haz_mass = defaultdict(lambda: defaultdict(float))
        chan_mass = defaultdict(lambda: defaultdict(float))
        trapped = defaultdict(float)
        ev_index = defaultdict(list)

        for rec in recs:
            if rec["claim_sev"] is None:
                strength = 0.0
            else:
                cred = self.credibility(rec["source"]) if self.flags["credibility"] else 0.6
                indep = rec["independence"] if self.flags["independence"] else 1.0
                strength = (C.MAX_EVIDENCE_MASS
                            * cred
                            * (0.35 + 0.65 * rec["sev_conf"])
                            * (0.45 + 0.55 * rec["spec"])
                            * indep)
            _ind = rec["independence"] if self.flags["independence"] else 1.0
            for sid, gp in rec["geo"]:
                obs[sid] += gp * _ind
                ev_index[sid].append((rec["rid"], gp))
                if strength <= 1e-6:
                    continue
                m_tot = min(0.97, strength * gp)
                k = self._kernel(rec["claim_sev"], rec["sev_conf"])
                m_s = k * m_tot
                m_th = 1.0 - m_tot
                logsum[sid] += np.log(np.maximum(m_s + m_th, 1e-300))
                logth[sid] += math.log(max(m_th, 1e-300))
                if rec["claim_sev"] < 0.30:
                    mass_low[sid] += m_tot
                elif rec["claim_sev"] > 0.58:
                    mass_high[sid] += m_tot
                chan_mass[sid][rec["source"]] += m_tot
                for h in rec["hazards"]:
                    haz_mass[sid][h] += m_tot
                if rec["trapped"]:
                    trapped[sid] += rec["trapped"] * gp * _ind

        # ---------- 4. silence engine ------------------------------------
        silence = self._silence(obs, t_now, rain_factor)

        # ---------- 5a. evidence-only belief, per settlement --------------
        stage = {}
        for st in self.g.settlements.values():
            sid = st.sid
            if not self.in_aoi(sid):
                continue          # another district's responsibility
            prior_sev = self.hazard_prior(st, rain_factor)
            sil = silence.get(sid)

            ls = logsum.get(sid)
            lt = logth.get(sid, 0.0)
            if ls is None:
                bel, th = np.zeros(NS), 1.0
                conflict_pair = (0.0, 0.0)
            else:
                d = np.clip(ls - lt, 0.0, 60.0)
                q = np.exp(d) - 1.0          # relative singleton masses
                Z = q.sum() + 1.0            # + q(Theta) = 1 in this scaling
                bel, th = q / Z, 1.0 / Z
                conflict_pair = (mass_low.get(sid, 0.0), mass_high.get(sid, 0.0))

            # Silence enters as a synthetic report from channel "SILENCE".
            if self.flags["silence"] and sil and sil["blackout_risk"] > 0.05:
                m_tot = min(C.SILENCE_MASS_CAP, 0.85 * sil["blackout_risk"])
                k = self._kernel(min(1.0, 0.55 + 0.45 * prior_sev), 0.45)
                bel, th = self._combine_one(bel, th, k * m_tot, 1.0 - m_tot)

            stage[sid] = [st, bel, th, prior_sev, sil, conflict_pair]

        # ---------- 5b. calibrate how far the terrain prior can be trusted -
        slope, intercept, trust, clim, clim_dist = self._calibrate_prior(stage)
        self.prior_trust = trust
        self.prior_fit = (slope, intercept, clim)
        self.clim_dist = clim_dist

        # ---------- 5c. posterior ----------------------------------------
        out = {}
        for sid, (st, bel, th, prior_sev, sil, conflict_pair) in stage.items():
            # Unassigned mass is resolved by terrain physics - but only as far
            # as terrain has actually been earning its keep this event. When
            # trust is low the prior shrinks toward district climatology
            # instead of asserting a confident guess.
            if self.flags["prior"]:
                # map terrain exposure onto the severity scale the evidence
                # actually shows this event to be on, then shrink by how well
                # the prior ranks
                fitted = slope * prior_sev + intercept
                eff_sev = float(np.clip(trust * fitted + (1.0 - trust) * clim, 0.0, 1.0))
                # Start from the district base rate and tilt it toward severity
                # by how far terrain puts this place above or below average.
                # An exponential tilt keeps the result a proper distribution
                # and cannot manufacture a state the base rate says is rare.
                tilt = np.exp(C.PRIOR_TILT * trust * (SEVW - SEVW.mean())
                              * (eff_sev - clim))
                prior_k = clim_dist * tilt
                prior_k = prior_k / prior_k.sum()
            else:
                prior_k = np.full(NS, 1.0 / NS)
            post = bel + th * prior_k
            post = post / post.sum()
            if abs(self.temperature - 1.0) > 1e-6:
                post = np.power(np.maximum(post, 1e-12), 1.0 / self.temperature)
                post = post / post.sum()

            dsi = float((post * SEVW).sum())
            label = next((n for thr, n in self.bands if dsi < thr),
                         C.DAMAGE_STATES[-1])
            ent = float(-(post * np.log(post + 1e-12)).sum() / math.log(NS))
            confidence = float(max(0.0, min(1.0, (1.0 - th) * (1.0 - 0.55 * ent))))

            lo, hi = conflict_pair
            contradiction = float(2 * min(lo, hi) / (lo + hi)) if (lo + hi) > 1e-6 else 0.0

            hz = haz_mass.get(sid, {})
            top_haz = sorted(hz.items(), key=lambda x: -x[1])[:3]
            if not top_haz:
                top_haz = [("FLOOD" if st.flood_exposure > st.landslide_exposure
                            else "LANDSLIDE", 0.0)]

            out[sid] = {
                "sid": sid, "name": st.name, "lat": st.lat, "lon": st.lon,
                "population": st.population, "elevation_m": st.elevation_m,
                "distribution": {s: round(float(p), 4) for s, p in zip(STATES, post)},
                "dsi": round(dsi, 4),
                "state": label,
                "state_argmax": STATES[int(np.argmax(post))],
                "confidence": round(confidence, 4),
                "evidence_mass": round(float(1.0 - th), 4),
                "contradiction": round(contradiction, 4),
                "prior_sev": round(prior_sev, 4),
                "n_reports": round(obs.get(sid, 0.0), 2),
                "raw_report_count": len(ev_index.get(sid, [])),
                "silence": sil or {},
                "hazards": [h for h, _ in top_haz],
                "trapped_est": int(trapped.get(sid, 0)),
                "evidence": [e[0] for e in sorted(ev_index.get(sid, []),
                                                  key=lambda x: -x[1])[:12]],
            }

        self._triage(out, t_now)
        self._update_credibility(recs, out, chan_mass)
        self._last = {"records": {r["rid"]: r for r in recs}, "t": t_now}
        return out, recs

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _combine_one(bel, th, m_s, m_th):
        """Dempster-combine an existing (bel, Theta) state with one new BPA."""
        new = bel * (m_s + m_th) + th * m_s
        nth = th * m_th
        Z = new.sum() + nth
        if Z < 1e-12:
            return bel, th
        return new / Z, nth / Z

    def _cluster(self, recs):
        """Group near-duplicate reports. Blocking by resolved settlement keeps
        this near-linear instead of O(n^2) over the whole stream."""
        if len(recs) < 2:
            return
        texts = [r["text"] for r in recs]
        try:
            V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                min_df=1, max_features=60000).fit_transform(texts)
        except ValueError:
            return
        buckets = defaultdict(list)
        for i, r in enumerate(recs):
            for sid, _ in r["geo"][:2]:
                buckets[sid].append(i)
        dsu = DSU(len(recs))
        for sid, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            idxs = idxs[:400]                      # cap pathological buckets
            sub = V[idxs]
            S = (sub @ sub.T).toarray()
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if S[a, b] < C.DEDUP_TEXT_SIM:
                        continue
                    ra, rb = recs[idxs[a]], recs[idxs[b]]
                    if abs(ra["t"] - rb["t"]) * 60.0 > C.DEDUP_WINDOW_MIN:
                        continue
                    dsu.union(idxs[a], idxs[b])
        for i, r in enumerate(recs):
            r["cluster"] = dsu.find(i)

    def _independence(self, recs):
        """Within a corroboration cluster, the k-th message from the same
        channel is worth INDEPENDENCE_DECAY^(k-1) of the first. One rumour
        shared 90 times is one witness, not ninety."""
        seen = defaultdict(int)
        by_cluster = defaultdict(list)
        for r in recs:
            by_cluster[r["cluster"]].append(r)
        for cl, group in by_cluster.items():
            group.sort(key=lambda x: (x["t"], x["rid"]))
            chan_n = defaultdict(int)
            uid_seen = set()
            for r in group:
                key = r["source"]
                k = chan_n[key]
                chan_n[key] += 1
                w = C.INDEPENDENCE_DECAY ** k
                # a channel repeating from the SAME handset/handle is weaker still
                uid = (r["source"], r["source_id"])
                if uid in uid_seen:
                    w *= 0.35
                uid_seen.add(uid)
                # blatant virality markers cut independence further
                w *= (1.0 - 0.55 * r["panic"])
                r["independence"] = round(max(0.02, w), 4)
                seen[cl] += 1

    def _silence(self, obs, t_now, rain_factor):
        """Convert ABSENCE into positive, calibrated evidence.

        Two independent absence signals, deliberately kept separate:

        A. INFRASTRUCTURE LIVENESS (telecom heartbeat, as DoT/TRAI supply to
           the SEOC). A tower that stops answering is evidence about the
           PLACE, not about whether anyone had something to say. This is what
           separates "quiet because unharmed" from "quiet because gone" - a
           distinction pure chatter-volume cannot make, because undamaged
           villages are quiet too.

        B. CHATTER DEFICIT. Secondary and weaker, used only to corroborate.

        Towers also fail from plain power loss, so tower-down on its own is
        NOT read as catastrophe: terrain exposure supplies the prior that
        separates a structural blackout from a generator that ran dry.
        """
        res = {}
        if t_now < 0.25:
            return res
        for st in self.g.settlements.values():
            if not self.in_aoi(st.sid):
                continue
            exp = (C.BASELINE_REPORTS_PER_1K_PER_HR * (st.population / 1000.0)
                   * st.connectivity * min(t_now, 24.0))
            o = obs.get(st.sid, 0.0)
            z = (exp - o) / math.sqrt(exp + 1.0)
            prior_sev = self.hazard_prior(st, rain_factor)

            tw = self.towers.get(st.sid)
            tower_down, hours_down, restored = False, 0.0, False
            if tw and tw.get("down_at") is not None and t_now >= tw["down_at"]:
                if tw.get("restored_at") is not None and t_now >= tw["restored_at"]:
                    restored = True
                else:
                    tower_down = True
                    hours_down = t_now - tw["down_at"]

            deficit = 0.0
            if exp >= C.SILENCE_MIN_EXPECTED:
                deficit = max(0.0, 1.0 - (o / exp) / C.SILENCE_DARK_RATIO)

            if tower_down:
                # P(structural | tower down) starts near the base rate and is
                # driven up by terrain exposure and by how long it has stayed
                # down. Corroborating chatter deficit adds a little.
                base = C.TOWER_DOWN_BASE_RISK
                risk = base + (1.0 - base) * prior_sev
                risk *= (0.70 + 0.30 * min(1.0, hours_down / 6.0))
                risk *= (0.80 + 0.20 * deficit)
                reason = "tower_down_%.1fh" % hours_down
                # Tower down is necessary but not sufficient: only escalate to
                # an actionable DARK ZONE when terrain corroborates. Otherwise
                # the list fills with generator failures and stops being read.
                is_dark = risk >= C.DARK_ESCALATE_RISK
            elif exp >= C.SILENCE_MIN_EXPECTED and deficit > 0.0 and z > C.SILENCE_ALARM_Z:
                # Tower alive but the place has gone quiet anyway: weaker,
                # ambiguous. Never allowed to dominate.
                risk = 0.45 * _sigmoid(2.2 * (z - C.SILENCE_ALARM_Z)) * deficit * prior_sev
                reason = "chatter_deficit"
                is_dark = False
            else:
                risk, is_dark = 0.0, False
                reason = ("tower_restored" if restored else
                          "no_tower" if not tw else
                          "below_detection_floor" if exp < C.SILENCE_MIN_EXPECTED
                          else "reporting_normally")

            res[st.sid] = {
                "expected": round(exp, 2), "observed": round(o, 2),
                "z": round(z, 3), "blackout_risk": round(min(1.0, risk), 4),
                "is_dark": bool(is_dark), "reason": reason,
                "tower_down_h": round(hours_down, 1) if tower_down else 0.0,
            }
        return res

    def _calibrate_prior(self, stage):
        """Empirical-Bayes recalibration of our own terrain model.

        Terrain exposure ranks damage reasonably but is on the wrong SCALE:
        it is systematically over- or under-confident depending on the event.
        Scoring it by absolute error therefore throws away a usable signal
        (measured: corr 0.41 with truth, yet worse absolute error than the
        district mean). So instead we regress observed evidence on the prior
        at settlements where evidence is strong, and use the fitted map.

        Returns (slope, intercept, trust, climatology).
        """
        anchors, dists = [], []
        for (_st, b_, t, p, _s, _c) in stage.values():
            if (1.0 - t) < C.PRIOR_ANCHOR_MIN_MASS:
                continue
            d_ = b_ / max(1e-9, 1.0 - t)
            anchors.append((p, float((d_ * SEVW).sum())))
            dists.append(d_)
        if len(anchors) < C.PRIOR_MIN_ANCHORS:
            return (0.0, 0.45, C.PRIOR_TRUST_DEFAULT, 0.45,
                    np.array(C.DISTRICT_BASE_RATE, dtype=float))
        pr = np.array([x[0] for x in anchors])
        ev = np.array([x[1] for x in anchors])
        clim = float(ev.mean())
        # Empirical base rate over damage states, measured where evidence is
        # strong. Without this the prior is a severity kernel that piles mass
        # onto whichever state sits nearest the middle of the severity scale,
        # and the whole district gets labelled MAJOR.
        # NOTE: deliberately NOT np.mean(dists). Anchors are settlements that
        # produced enough reports to be well-evidenced, i.e. the damaged ones.
        # Using their mix as the base rate for silent villages is selection
        # bias, and measured at ~90% MAJOR across the whole district.
        clim_dist = np.array(C.DISTRICT_BASE_RATE, dtype=float)
        clim_dist = clim_dist / clim_dist.sum()
        vp = float(pr.var())
        if vp < 1e-9:
            return 0.0, clim, 0.05, clim, clim_dist
        slope = float(((pr - pr.mean()) * (ev - ev.mean())).mean() / vp)
        intercept = clim - slope * float(pr.mean())
        sd = float(pr.std() * ev.std())
        r = float(((pr - pr.mean()) * (ev - ev.mean())).mean() / sd) if sd > 1e-9 else 0.0
        # trust is RANKING skill, not scale agreement
        trust = float(max(0.0, min(1.0, r)))
        if trust <= 0.02:
            return 0.0, clim, 0.05, clim, clim_dist
        return slope, intercept, trust, clim, clim_dist

    # ---------------------------------------------------------- triage    # ---------------------------------------------------------- triage
    def _triage(self, out, t_now):
        """Expected Unassisted Casualties, plus Value of Information."""
        for sid, v in out.items():
            dist = v["distribution"]
            rate = sum(dist[s] * C.CASUALTY_RATE[s] for s in STATES)
            # survival decays through the golden window, steepest early
            decay = math.exp(-1.6 * t_now / C.GOLDEN_WINDOW_HRS)
            euc = v["population"] * rate * (0.45 + 0.55 * decay)
            v["euc"] = round(euc, 2)
            v["priority"] = round(euc, 2)

            # Value of information: worth verifying when stakes are high AND
            # we are unsure. This drives drone/recon tasking, NOT rescue.
            stakes = v["population"] * max(0.0, v["dsi"] - 0.35)
            unsure = (1.0 - v["confidence"]) * (0.55 + 0.45 * v["contradiction"])
            v["voi"] = round(stakes * unsure / 100.0, 3)

            needs = []
            hz = set(v["hazards"])
            p_sev = dist["MAJOR"] + dist["CATASTROPHIC"]
            if p_sev > 0.40:
                if "FLOOD" in hz or v.get("trapped_est"):
                    needs.append("BOAT")
                if "STRUCTURAL_COLLAPSE" in hz or "LANDSLIDE" in hz:
                    needs.append("EXCAVATOR")
                if p_sev > 0.50 or "MEDICAL" in hz:
                    needs.append("MEDICAL")
            if not needs and p_sev > 0.55:
                needs = ["MEDICAL"]
            v["needs"] = needs or []


# ------------------------------------------------------------------ CLI test
if __name__ == "__main__":
    import sys, time
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from simulate import EPICENTRE, MAGNITUDE, SCENARIO_BBOX
    DATA = os.path.join(os.path.dirname(__file__), "data")
    reports = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))

    eng = FusionEngine(epicentre=EPICENTRE, magnitude=MAGNITUDE, aoi=SCENARIO_BBOX)
    t0 = time.time()
    out, recs = eng.run(reports, t_now=24.0)
    print("fused %d reports -> %d settlements in %.2fs"
          % (len(recs), len(out), time.time() - t0))

    rank = sorted(out.values(), key=lambda v: -v["priority"])[:15]
    print("\n%-22s %5s %5s %5s %6s %6s  %s" %
          ("SETTLEMENT", "DSI", "CONF", "CTRD", "PRIOR", "TRUTH", "WHY"))
    for v in rank:
        tr = truth.get(v["sid"], {}).get("state", "-")
        why = "SILENT" if v["silence"].get("is_dark") else "reports"
        print("%-22s %5.2f %5.2f %5.2f %6.1f %6s  %s"
              % (v["name"][:22], v["dsi"], v["confidence"], v["contradiction"],
                 v["priority"], tr[:6], why))
