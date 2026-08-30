# DRISHTI — hackathon roadmap

30-hour hackathon, 22 hours remaining at the start of this build.
Budget: **₹0**. Team: solo.

---

## Status: the system is built, tested and reproducible

Everything below marked ✅ exists, runs, and has its numbers checked in.

```
cd drishti/backend
python run_all.py                  # 50s — regenerates every published number
python -m pytest test_drishti.py   # 30/30 pass
python uitest.py                   # end-to-end browser test, screenshots
python app.py                      # the live dashboard
```

---

## Phase 1 — Foundations ✅

| | Deliverable | Status |
|---|---|---|
| 1.1 | Pick the study area, pull real OSM data | ✅ 2,059 settlements, 401 roads, 125 bridges, 60 rivers (Rudraprayag–Chamoli) |
| 1.2 | Real terrain elevation | ✅ Open-Meteo, 569–3,544 m, cached offline |
| 1.3 | Hazard exposure model (flood / landslide / seismic) | ✅ validated against published MMI |
| 1.4 | Fuzzy gazetteer with ambiguity preserved | ✅ handles 42 duplicate village names |

## Phase 2 — Scenario engine ✅

| | Deliverable | Status |
|---|---|---|
| 2.1 | Multi-hazard event with hidden ground truth | ✅ 1,102 settlements, calibrated severity mix |
| 2.2 | Six report channels, Hinglish text | ✅ 3,256 reports |
| 2.3 | Rumour amplification + contradictions | ✅ 913 false reports |
| 2.4 | Comms blackout + telecom heartbeat feed | ✅ 1,101 towers |
| 2.5 | **Adversarial realism** | ✅ latent vulnerability field (80% of outcome), district-wide grid failure |

## Phase 3 — Fusion engine ✅

| | Deliverable | Status |
|---|---|---|
| 3.1 | Claim extraction from code-mixed text | ✅ rules-based, LLM tier optional |
| 3.2 | Corroboration graph + independence weighting | ✅ ~900 reports suppressed |
| 3.3 | Dempster–Shafer fusion, closed form, log space | ✅ 0.26s for full district |
| 3.4 | Online source credibility | ✅ leave-one-channel-out, bounded |
| 3.5 | **Silence-as-evidence engine** | ✅ +0.14 blind-spot AUC |
| 3.6 | Self-calibrating terrain prior | ✅ regression recalibration, trust 0.27 |
| 3.7 | Probability calibration | ✅ fitted on held-out seed, ECE 0.090 |

## Phase 4 — Decision layer ✅

| | Deliverable | Status |
|---|---|---|
| 4.1 | Damaged road graph | ✅ 29,005 nodes, bridges closed on belief |
| 4.2 | Rescue tasking (expected lives saved / asset-hour) | ✅ |
| 4.3 | Recon tasking (value of information) | ✅ separate queue |
| 4.4 | Unreachable → air-asset list | ✅ 215 settlements |

## Phase 5 — Interface ✅

| | Deliverable | Status |
|---|---|---|
| 5.1 | EOC dashboard, dark ops theme | ✅ zero build step |
| 5.2 | Time scrubber over the first 24h | ✅ 0.26s per step |
| 5.3 | Evidence drill-down with audit trail | ✅ per-report weights shown |
| 5.4 | Benchmark + method tabs | ✅ |

## Phase 6 — Proof ✅

| | Deliverable | Status |
|---|---|---|
| 6.1 | Benchmark vs 3 baselines + 4 ablations | ✅ 6× on both headline metrics |
| 6.2 | Confusion matrix, Brier, ECE | ✅ out-of-sample |
| 6.3 | Unit + property tests | ✅ 30/30 |
| 6.4 | Real-world data validation | ✅ 39/39 |

## Phase 7 — Shipping ✅

| | Deliverable | Status |
|---|---|---|
| 7.1 | README, HOW_IT_WORKS, PITCH | ✅ |
| 7.2 | Static GitHub Pages build | ✅ `docs/demo/`, 6.1 MB, no backend |
| 7.3 | requirements / licence / gitignore | ✅ |
| 7.4 | Screenshots | ✅ `docs/screenshots/` |

---

## Your remaining hours — suggested plan

The build is done. Spend what's left on the things only you can do.

### T-6h to T-4h · Rehearse (highest value)

Run through **[docs/PITCH.md](docs/PITCH.md)** out loud, twice, with the app
open. The demo has six beats and runs three minutes. Beat 2 — clicking a dark
zone and showing *0 reports received, 418 expected* against a CATASTROPHIC
ground truth — is the entire pitch. Land that one cleanly.

Memorise the numbers table at the bottom of PITCH.md.

### T-4h to T-3h · Record a fallback video

Screen-record the 3-minute demo. If the venue laptop dies or the projector
fights you, you present the video and lose nothing.

### T-3h to T-2h · Push to GitHub

```bash
git init && git add -A
git commit -m "DRISHTI — post-disaster information fog resolver (PS-5)"
git branch -M main
git remote add origin https://github.com/<you>/drishti.git
git push -u origin main
```

Then *Settings → Pages → deploy from branch → `main` / `/docs`*, and put the
resulting link in the README and the submission form.

### T-2h to T-1h · Submit

Submission should carry: repo link, Pages demo link, README, and the
benchmark table. Everything is already written.

### T-1h to T-0 · Hold

Do not start new features. Re-run `python run_all.py` once to confirm the
machine is clean, and stop.

---

## If you have spare time — ranked by judge impact

1. **Rehearse a third time.** Genuinely higher value than any feature.
2. **Multi-seed robustness.** Run the pipeline over 5 scenario seeds and quote
   mean ± spread. Turns "it worked once" into "it works". ~30 min:
   loop `simulate.py` with different seeds and collect `metrics.json`.
3. **A screenshot in the README.** `docs/screenshots/overview.png` exists —
   embedding it makes the repo land in three seconds instead of thirty.
4. **Hindi labels in the UI.** Cheap, and it reads as built-for-India.
5. **A live-feed banner.** `/api/live` already returns real USGS quakes and
   real rainfall; surfacing it in the header proves the pipeline is wired to
   the real world.

## Deliberately *not* doing

- **Real satellite change detection.** Sentinel-1 SAR is free but the
  processing chain does not fit the time, and half-working imagery is worse
  than none. Named as future work.
- **A trained ML model.** No labelled disaster corpus exists to train on, and
  a model trained on our own simulator would prove nothing.
- **Mobile app / SMS gateway.** Real deployment need, wrong thing to spend
  the last hours of a hackathon on.
- **Anything that needs a paid API.** Budget is ₹0 and that is a stated
  constraint of the build.
