# How DRISHTI works

A walkthrough of the pipeline, the maths, and — importantly — the things we
got wrong and had to fix. If you are explaining this project to someone, this
is the document to read first.

---

## 0. The problem, stated precisely

Problem Statement 5 describes an EOC that cannot answer one question:

> *Which settlements are destroyed, and where do I send my boats?*

The naive framing is "extract facts from noisy reports". That framing fails,
because the defining feature of the first 24 hours is not that reports are
noisy. It is that **the worst-hit places produce no reports at all.**

Report volume is *non-monotonic* in damage:

```
  reports
    ^
    |          ,-'''-.            a moderately hit village has
    |        ,'       `.          survivors, a working tower and
    |      ,'           `.        every reason to call
    |    ,'               `.
    |  ,'                   `.
    |,'                       `-._____
    +---------------------------------->  damage
    intact                    catastrophic
     (nothing            (nobody left to call,
      to report)          no tower to call on)
```

So any system that ranks by "where is the noise" ranks the catastrophic tail
as *quiet, therefore fine*. In our scenario **492 of 1,102 settlements (45%)
never emit a single report**, and 122 of them are catastrophic.

That reframes the task. We are not extracting facts. We are maintaining a
**calibrated belief** over every settlement — including the silent ones — and
acting on expected loss under that belief.

---

## 1. Geo-resolution: keep the doubt

**File:** `gazetteer.py` → `Gazetteer.resolve()`

Ground reports name places the way people speak: *"the village past the broken
bridge near Ukhimath"*. Two problems compound:

1. Names are garbled by bad lines and panicked callers.
2. Real districts reuse names. Ours has **four** Chandrapuri, **three** Kund,
   and **42 names shared by three or more settlements**.

So `resolve()` returns a distribution, not a guess:

```python
resolve("the village past Gaurikund")
#  0.28  Gaurikund   pop=144   flood=0.82
#  0.24  Kund        pop=317   flood=0.59
#  0.24  Kund        pop=296   flood=1.00
#  0.24  Kund        pop=215   flood=0.65
```

Scoring is `WRatio³` (sharpened, so weak matches die), multiplied by a spatial
prior when partial GPS exists, and by a mild size prior because people mention
big places more often. A report's evidence mass is then **split across
candidates by probability** — it never fully commits to one village.

This is why an EOC officer never gets a confident boat tasking to the wrong
Chandrapuri.

---

## 2. Corroboration graph: panic must not vote twice

**File:** `fusion.py` → `_cluster()`, `_independence()`

The single most dangerous failure in social-media-driven crisis mapping is
mistaking **amplification** for **corroboration**. One false rumour shared
ninety times looks, to a naive counter, like ninety independent witnesses.

Two steps:

**Clustering.** TF-IDF over character 3–5-grams (robust to Hinglish spelling
variation), blocked by resolved settlement so it stays near-linear instead of
O(n²), then union-find over pairs above cosine 0.62 within a 90-minute window.

**Independence weighting.** Within a cluster, reports are sorted by time and
the *k*-th message from a given channel is weighted `0.45^(k-1)`. A repeat from
the same handset/handle is cut a further 65%. Explicit virality markers
("SHARE THIS", "forward:", ALL CAPS) cut it further.

Because weights are assigned by *time-rank within cluster*, any time-prefix of
the stream reuses them unchanged — which is what makes the time scrubber
instant rather than a full recompute.

Result: **~900 of 3,256 reports** end up below 0.15 independence.

---

## 3. Evidence fusion: Dempster–Shafer in closed form

**File:** `fusion.py` → `fuse()`

Frame of discernment: `{INTACT, MINOR, MAJOR, CATASTROPHIC}`.

Each report becomes a **simple support function** — mass `m(s)` spread over
states by a Gaussian kernel centred on the claimed severity (bandwidth widens
when the claim is vague), with the remainder `m(Θ)` left on "don't know".
Mass strength is

```
strength = MAX_MASS × credibility(channel) × claim_confidence
                    × specificity × independence × geo_probability
```

Naively combining thousands of BPAs with Dempster's rule is slow and
numerically fragile. But for simple support functions the *n*-fold combination
has a closed form:

```
q(s) = Π (mᵢ(s) + mᵢ(Θ))  −  Π mᵢ(Θ)
q(Θ) = Π mᵢ(Θ)
```

We compute it in log space and normalise via `exp(dₛ) − 1` where
`dₛ = Σ log(mᵢ(s)+mᵢ(Θ)) − Σ log mᵢ(Θ) ≥ 0`, so the products never underflow
regardless of report count. Cost is O(n · |states|).

**Contradiction** is tracked separately and reported as a first-class number:
mass asserting "safe" versus mass asserting "severe", as
`2·min(lo,hi)/(lo+hi)`. A settlement where sources genuinely disagree gets
flagged for *verification*, not quietly averaged into the middle.

---

## 4. Silence as evidence

**File:** `fusion.py` → `_silence()`

This is the mechanism the project exists for, and the one we had to rebuild
three times.

**Attempt 1 — chatter deficit.** Compare observed reports to what a
settlement's population and connectivity say it should produce. *Failed.*
Measured: 5 of 10 "dark" flags were **INTACT** villages. Of course they were —
an undamaged village is quiet too, because it has nothing to report. Report
volume alone simply cannot separate the two cases.

**Attempt 2 — cell-tower liveness.** Use a genuinely independent observation:
does the tower still answer? An unharmed village is quiet *but its tower is
alive*. This is real — DoT/TRAI supply tower status to State EOCs. Precision
jumped immediately.

**Attempt 3 — confront the confounder.** Then we measured
P(tower down | true state):

```
INTACT 0.06   MINOR 0.20   MAJOR 0.61   CATASTROPHIC 0.94
```

Tower status was very nearly a copy of the answer. Any result built on it
would have been an artefact of our own simulator, not a solution.

Real towers do not fail only from structural damage. In a flood the grid is
**deliberately cut across whole feeder zones to prevent electrocution** — the
exact scenario of PS-4 in this same problem set — and every tower then runs on
battery and diesel until it runs out. So the simulator now fails the grid
across contiguous feeder zones, with staggered backup exhaustion and partial
refuelling:

```
INTACT 0.51   MINOR 0.58   MAJOR 0.82   CATASTROPHIC 0.98
```

Now **half of all undamaged villages also go dark**, tower-down is weak
evidence, and the engine has to actually do inference:

```
risk = P(structural | tower down)
     = base(0.30)  +  (1 − base) × terrain_exposure
     × duration_factor(hours_down)
     × corroboration_factor(chatter_deficit)
```

It enters the same DS machinery as a synthetic report from a channel called
`SILENCE`, capped at mass 0.20 — deliberately small, because the evidence is
weak. It shows up in the evidence panel like any other source, and can be
argued with.

**What it is worth** (ROC-AUC, ablation):

| | All settlements | Never reported |
|---|---:|---:|
| with silence engine | 0.857 | **0.848** |
| without | 0.795 | 0.711 |

The gain is concentrated exactly where it should be.

---

## 5. The engine calibrates its own prior

**File:** `fusion.py` → `_calibrate_prior()`

Terrain exposure (river distance, elevation, seismic attenuation) gives a
prior for settlements we have not heard from. But how much should we trust it?

Our first answer scored the prior by mean absolute error against evidence and
concluded it was worse than climatology — trust 0.05. That was wrong, and
measurably so: the prior correlated **0.41** with truth. It ranked well but
sat on the wrong *scale*.

So the prior is now **regressed** on settlements where evidence is strong:

```
fitted     = slope · prior_sev + intercept        (fixes the scale)
trust      = max(0, corr(prior, evidence))        (measures ranking skill)
eff_sev    = trust · fitted + (1 − trust) · climatology
```

and applied as an **exponential tilt on a stated district base rate** rather
than as a severity kernel:

```
prior(s) ∝ base_rate(s) · exp(TILT · trust · (SEVWₛ − mean) · (eff_sev − clim))
```

### Why the base rate is *stated*, not learned

We first learned the base rate from well-evidenced settlements. It came out
**~90% MAJOR**, and the whole district got labelled MAJOR.

The bug is selection bias: the settlements with enough evidence to be
"anchors" are the ones that generated lots of reports — which are the damaged
ones. Their mix is not the district's mix, and applying it to silent villages
is exactly backwards. So the base rate is an operator-set expectation (the
declared scale of the event), exposed in the UI so it can be challenged.

In the demo scenario the engine settles on **prior trust 0.27** — it has
worked out that terrain is only weakly predictive this time.

---

## 6. Calibration, fitted out-of-sample

**File:** `calibrate.py`

Raw fused beliefs ranked well but were **under-dispersed**: the engine said
0.52 where the true rate was 0.75, and 0.25 where the true rate was 0.04. An
EOC officer reads "60% chance this village is destroyed" as a number, so this
matters.

We fit a single temperature `p^(1/T)` by minimising multiclass log-loss, and
derive the triage band thresholds from the calibrated DSI quantiles — **both
on a different scenario seed (77777) from the one every reported metric is
evaluated on (20260830)**.

Result: `T = 0.30`, log-loss 1.234 → 1.110, ECE 0.192 → 0.106 in-fit and
**0.090 out-of-sample**.

> One subtlety that cost us an hour: the fitter builds its own engine, which
> *loads the existing calibration file*. So the second run fitted a
> temperature on top of a temperature and converged to T ≈ 1, silently
> un-calibrating everything. `collect()` now forces `temperature = 1.0`.

### Why the triage label is not the argmax

`state` comes from **calibrated DSI bands**, not from `argmax(distribution)`.
DSI is an expectation, so it shrinks from the extremes — bands on the raw 0–1
severity scale never fired CATASTROPHIC at all (measured DSI max: 0.756). The
distribution stays the honest belief; the label is a decision, and decisions
need stable thresholds. Both are exposed via the API.

---

## 7. Triage and allocation

**Files:** `fusion.py` → `_triage()`, `allocate.py`

**Expected Unassisted Casualties** drives the rescue queue:

```
EUC = population × Σ P(s)·casualty_rate(s) × survival_decay(t)
```

**Value of Information** drives the recon queue — deliberately a *different*
ranking:

```
VoI = stakes × uncertainty = pop·max(0, DSI−0.35) × (1−confidence)·(1+contradiction)
```

Assets route over the **damaged** network: a real OSM graph (29,005 nodes) in
which edges near settlements we believe are severely hit are slowed, and
bridges are closed once P(severe) nearby exceeds 0.62. Assignment is greedy on
marginal expected lives saved per asset-hour — deliberately, because it is
transparent and an officer can override any single line without the rest
collapsing.

In the demo scenario this leaves **~215 settlements with no passable surface
route**, surfaced as an explicit *air assets required* list. That is not a
bug; that is the 2013 Kedarnath failure mode.

> Threshold lesson: these were originally tuned against raw DSI. After
> recalibration they silently stopped firing — **zero** roads closed. All
> operational thresholds now key on `P(MAJOR) + P(CATASTROPHIC)`, a calibrated
> probability that keeps its meaning when the belief scale changes.

---

## 8. What we got wrong, and fixed

Kept deliberately, because the fixes are most of the engineering.

| # | Bug | How it showed up | Fix |
|---|---|---|---|
| 1 | Silence detector fired on 96% of the district | ablation said the engine was *better without it* | hard gate on near-total absence |
| 2 | Chatter deficit conflated "quiet" with "destroyed" | 5/10 dark flags were INTACT | independent signal: tower liveness |
| 3 | Tower status was a label proxy | P(down) 0.06 vs 0.94 by true state | district-wide grid-failure confounder |
| 4 | Latent vulnerability computed but never applied | AUC 0.98, suspiciously high | actually use it; sweep its weight |
| 5 | Prior judged by absolute error | trust 0.05 despite corr 0.41 | regression recalibration |
| 6 | Base rate learned from anchors | 90% of district labelled MAJOR | operator-stated base rate |
| 7 | Temperature fitted on tempered output | ECE silently reverted to 0.177 | force raw in the fitter |
| 8 | Credibility self-confirmation loop | SOCIAL drifted 0.29 → **0.81** | leave-one-channel-out + bounded budget |
| 9 | `fuse()` mutated engine state | repeated API calls drifted | credibility re-derived every call |
| 10 | Route thresholds on the pre-calibration scale | 0 roads blocked | key on calibrated P(severe) |
| 11 | Seismic attenuation under-predicted near field | MMI 4.6 where published says ~VII | hypocentral-distance relation |

Numbers 3, 5, 6 and 8 were each found by a measurement that contradicted what
we wanted to believe. The ablation table and `validate_real.py` exist so those
contradictions surface early.

---

## 9. Reproducing everything

```bash
cd drishti/backend
python run_all.py             # ~50s: scenario → calibration → all metrics
python -m pytest test_drishti.py -q
python uitest.py              # end-to-end browser test + screenshots
python export_static.py       # build the GitHub Pages demo
```

`run_all.py` regenerates every number quoted in the README from scratch.
