"""Probability calibration, fitted on a HELD-OUT scenario.

The raw fused distribution ranks well (AUC 0.90) but is under-dispersed: it
says 0.52 where the truth rate is 0.75, and 0.25 where the truth rate is 0.04.
That matters operationally, because an EOC officer reads "60% chance this
village is destroyed" as a number, not as a rank.

So we fit a single temperature by minimising multiclass log-loss - on a
DIFFERENT scenario seed from the one we evaluate on. Fitting the temperature
on the evaluation scenario would make the reported calibration meaningless.

Run:  python calibrate.py          (fits and writes data/calibration.json)
"""
import json, os, subprocess, sys

import numpy as np

import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "calibration.json")
CAL_SEED = 77777          # deliberately different from the demo scenario seed
S = C.DAMAGE_STATES


def build_calibration_scenario():
    """Generate an independent scenario purely for fitting."""
    import simulate
    from importlib import reload
    reload(simulate)
    sc = simulate.Scenario(seed=CAL_SEED)
    return sc


def collect(sc):
    from fusion import FusionEngine
    from dataclasses import asdict
    import simulate
    reports = [asdict(r) for r in sc.reports]
    eng = FusionEngine(epicentre=simulate.EPICENTRE, magnitude=simulate.MAGNITUDE,
                       aoi=simulate.SCENARIO_BBOX)
    # the engine must read THIS scenario's towers, not the demo one's
    eng.towers = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                  for k, v in sc.towers.items()}
    # CRITICAL: fit from the RAW distribution. If the engine loaded a previous
    # calibration we would be fitting a temperature on top of a temperature,
    # which converges to T=1 and silently un-calibrates the system.
    eng.temperature = 1.0
    eng.bands = list(C.DSI_BANDS)
    eng.prepare(reports)
    out, _ = eng.fuse(24.0)
    P, Y = [], []
    for sid, v in out.items():
        t = sc.truth.get(sid)
        if not t:
            continue
        P.append([v["distribution"][s] for s in S])
        Y.append(S.index(t["state"]))
    return np.array(P), np.array(Y)


def fit_temperature(P, Y, grid=None):
    """Temperature scaling on a probability simplex: p^(1/T), renormalised.
    T < 1 sharpens (more confident), T > 1 flattens."""
    grid = grid if grid is not None else np.arange(0.30, 2.51, 0.02)
    best, bestT = 1e18, 1.0
    for T in grid:
        Q = np.power(np.maximum(P, 1e-12), 1.0 / T)
        Q /= Q.sum(axis=1, keepdims=True)
        ll = -np.log(np.maximum(Q[np.arange(len(Y)), Y], 1e-12)).mean()
        if ll < best:
            best, bestT = ll, float(T)
    return bestT, best


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Building held-out calibration scenario (seed %d)..." % CAL_SEED)
    sc = build_calibration_scenario()
    print("  %d settlements, %d reports" % (len(sc.truth), len(sc.reports)))
    P, Y = collect(sc)
    print("  collected %d labelled settlements" % len(Y))

    T0 = -np.log(np.maximum(P[np.arange(len(Y)), Y], 1e-12)).mean()
    T, ll = fit_temperature(P, Y)
    print("\nlog-loss before  T=1.00 : %.4f" % T0)
    print("log-loss after   T=%.2f : %.4f" % (T, ll))

    Q = np.power(np.maximum(P, 1e-12), 1.0 / T)
    Q /= Q.sum(axis=1, keepdims=True)
    sev = np.isin(Y, [2, 3]).astype(int)
    for tag, M in (("before", P), ("after", Q)):
        ps = M[:, 2] + M[:, 3]
        ece = 0.0
        for lo, hi in [(0, .15), (.15, .3), (.3, .45), (.45, .6), (.6, .75), (.75, 1.01)]:
            m = (ps >= lo) & (ps < hi)
            if m.sum() < 5:
                continue
            ece += m.sum() / len(ps) * abs(ps[m].mean() - sev[m].mean())
        print("ECE on calibration scenario (%s): %.4f" % (tag, ece))

    # Triage band thresholds: quantiles of the CALIBRATED DSI that reproduce
    # the declared district base rate. Derived here, on the held-out scenario,
    # so the demo scenario never informs its own thresholds.
    SEVW = np.array([C.SEVERITY_WEIGHT[s] for s in S])
    dsi = (Q * SEVW).sum(axis=1)
    cum = np.cumsum(C.DISTRICT_BASE_RATE)[:-1]
    bands = [float(np.percentile(dsi, 100 * c)) for c in cum]
    print("\nDSI band thresholds fitted on held-out scenario: %s"
          % [round(b, 3) for b in bands])

    json.dump({"temperature": T, "seed": CAL_SEED, "log_loss": ll,
               "log_loss_uncalibrated": float(T0), "n": int(len(Y)),
               "dsi_bands": bands},
              open(OUT, "w"), indent=1)
    print("\nsaved -> data/calibration.json   (applied automatically by fusion.py)")
    print("NOTE: evaluate/validate run on seed 20260830, so this temperature is")
    print("      fitted out-of-sample with respect to every reported metric.")


if __name__ == "__main__":
    main()
