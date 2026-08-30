# DRISHTI — demo script and judge Q&A

Everything you need to present this. Read once, then run the demo.

---

## The 60-second version

> "After a disaster, the emergency operations centre gets thousands of
> messages. The instinct is to build a dashboard that shows where the reports
> are coming from.
>
> That instinct kills people. Because the villages that are completely
> destroyed send **no** reports — the people are gone and the tower is gone.
> They look, on any volume dashboard, exactly like villages that are fine.
>
> DRISHTI treats **silence as evidence**. In our benchmark, for settlements
> that never sent a single report, a report-driven system scores 0.500 on
> ROC-AUC — a coin flip. DRISHTI scores 0.848. Given 40 deployments, volume
> ranking finds 2 of 122 catastrophic villages. We find 12, and reach six
> times more people.
>
> Built on OpenStreetMap, Open-Meteo and USGS. No API keys. Zero rupees."

---

## The 3-minute live demo

Have the app running: `cd drishti/backend && python app.py` →
`http://127.0.0.1:8000`. Start with the scrubber at **T+24:00**.

### Beat 1 — the fog (20s)

*Point at the map.* "1,102 real villages in Rudraprayag and Chamoli, from
OpenStreetMap. Real terrain, real roads, real bridges. 3,256 incoming reports
— WhatsApp forwards, 112 calls, sat-phone fragments, ham radio, patwari
reports. About 900 of them are false."

### Beat 2 — the killer (40s)  ← the whole pitch is this beat

*Click the* **Dark zones** *tab.* "These are settlements that have stopped
reporting while their towers are down and the terrain says they were exposed."

**Click `Agastmuni` — it is rank 1 in the triage list.** This is the single
strongest moment in the demo. A town of **19,758 people** that sent **3
reports when 418 were expected**, tower down for 23.9 hours, and where 65% of
what little did arrive *contradicts itself*.

Point at the belief line, then the ground truth line:

> Belief: **CATASTROPHIC** · evidence mass 33% — the rest is terrain prior
> Ground truth: **CATASTROPHIC · 2,534 casualties · 7,795 trapped — match**

"Three reports. On a volume dashboard this town is quiet, and quiet reads as
fine. It has two and a half thousand dead.

And look at what we're honest about: confidence 16%, evidence mass 33%. We are
*not* claiming certainty. We're saying — this is where the bodies most likely
are, here is exactly how little we know, and here is every scrap we based it
on."

If a judge wants a *pure* zero-report case, use **`Trishula`**: **0 reports**,
11 expected, belief CATASTROPHIC, ground truth CATASTROPHIC. Or **`Gauchar`**:
2 reports of 129 expected, 521 dead.

### Beat 3 — panic doesn't get to vote (30s)

*Go to* **Triage** *→ click a settlement with a "contested" tag → look at the
evidence list.* Strikethrough entries are marked **DUPLICATE — SUPPRESSED**.

"One rumour shared ninety times is one witness, not ninety. Every report
carries its independence weight, its source credibility and its geographic
probability. Nothing is a black box — an officer can audit exactly why we
believe what we believe."

### Beat 4 — the decision (30s)

*Click* **Assets**. "Belief becomes tasking. Boats, excavators and medical
teams routed over the *damaged* road network — bridges we believe are gone are
closed. 215 settlements have no passable surface route at all; those are
flagged for air assets.

And separately, **drones** go where uncertainty is most expensive. You don't
spend your only excavator on an inference. You spend a drone on it, and earn
the right to send the excavator."

### Beat 5 — the proof (40s)

*Click* **Benchmark**. "Same reports, same budget of 40 deployments, scored
against hidden ground truth. Volume ranking: 2 catastrophic zones, a thousand
people reached. Us: 12 zones, six thousand people.

And the ablations — this is us marking our own homework honestly. Remove the
silence engine and blind-spot AUC collapses from 0.85 to 0.71."

### Beat 6 — time (20s)

*Drag the scrubber from 24 back to 3, then play forward.* "This is the first
24 hours replaying. Watch the dark zones appear as towers fail and belief
sharpens. Two dark zones at T+3, seventy-eight by T+24."

---

## Questions you will be asked

**"Isn't this just a dashboard?"**
No. A dashboard shows you the reports. The entire point is the settlements
with *no* reports — 45% of this district. Those are invisible on any
dashboard and they are where the deaths are.

**"You simulated the disaster. Doesn't that make the result meaningless?"**
The opposite — it is the only way to prove anything. The whole difficulty of
PS-5 is that nobody knows the truth, so demonstrating that you *recover* truth
requires a world where truth exists and is hidden from the engine. The engine
never sees ground truth; it is used only for scoring, afterwards.

And we deliberately made the simulator hostile: 45% of settlements never
report, 28% of reports are false, 80% of damage is driven by a latent
vulnerability field no map layer can observe, and the power grid fails
district-wide so tower status can't stand in for the answer.

**"How do we know the terrain prior isn't just giving you the answer?"**
Because we checked, and at first it *was*. Damage was generated from the same
variables the prior reads, and AUC was an implausible 0.98. We added a latent
vulnerability field — building stock, drainage, upstream releases — that
carries 80% of the outcome and is invisible to every free map layer. AUC
dropped to 0.86, which is the honest number. The engine also measures its own
prior's skill each cycle and shrank it to trust 0.27.

**"Tower-down obviously means destroyed. Isn't that circular?"**
It was, and we caught it: P(tower down) was 0.06 for intact villages and 0.94
for catastrophic ones. So we modelled the district-wide grid failure that
really happens — the same scenario as PS-4 in this problem set. Now 51% of
*undamaged* villages also go dark. Tower-down is weak evidence, capped at mass
0.20, and combined with terrain and chatter deficit.

**"Your accuracy is only 43%."**
Exact 4-class accuracy, yes, and we publish it rather than hide it. The
operationally meaningful numbers are 91% within one severity band, ROC-AUC
0.857, and — the row that matters — of 122 truly catastrophic settlements, 2
are labelled INTACT. We do not tell you a destroyed village is fine.

**"What does this cost to run?"**
Zero. OpenStreetMap, Open-Meteo and USGS are free and key-less. It runs on a
laptop, fully offline after first fetch. That is a requirement, not a
constraint — a district EOC in Uttarakhand is not going to expense a Mapbox
plan at 3 a.m.

**"Would a district actually adopt this?"**
The inputs are things a State EOC already has: 112/108 call logs, tower status
from DoT/TRAI, block and patwari reports, social feeds. The outputs are a
ranked list with an audit trail per settlement. Nothing here requires new
sensors.

**"What's the biggest weakness?"**
Tower liveness is modelled rather than live — in deployment it comes from the
telecom operators. And the whole system is validated against a simulator we
wrote. That is why `validate_real.py` exists: 39 checks against published
real-world facts — coordinates, elevations, seismic attenuation against
published MMI, and live USGS/Open-Meteo feeds.

**"Why Dempster–Shafer instead of plain Bayes?"**
Because "we have no evidence" and "the evidence says 50/50" are completely
different operational situations, and Bayesian posteriors alone conflate them.
DS keeps unassigned mass explicit, so the UI can show that a belief is 80%
assumption — which is exactly what an officer needs to know before committing
an excavator.

---

## Numbers to have memorised

| | |
|---|---|
| Blind-spot ROC-AUC: DRISHTI vs report-driven | **0.848 vs 0.500** |
| Demo village: Agastmuni | **3 reports / 418 expected · 2,534 dead** |
| Catastrophic zones found in top 40 | **12 vs 2** |
| People reached | **6,057 vs 1,006** |
| Settlements that never report | **492 of 1,102 (45%)** |
| False reports suppressed | **913 of 3,256** |
| Within one severity band | **91.0%** |
| Catastrophic mislabelled INTACT | **2 of 122** |
| Real-world checks passing | **39 / 39** |
| Unit tests passing | **30 / 30** |
| Cost | **₹0** |

---

## If the demo breaks

- **Server won't start** — `cd drishti/backend && python run_all.py` first.
- **No internet at the venue** — everything is cached; the map falls back and
  the Live tab degrades. Nothing else depends on the network.
- **Total laptop failure** — `docs/demo/` is a static build. Open
  `index.html` from any machine, or the GitHub Pages link.
- **Asked to prove it isn't hardcoded** — `python run_all.py` regenerates
  every number in under a minute, live.
