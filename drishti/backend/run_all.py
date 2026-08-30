"""Reproduce every number in this repository, in order, from scratch.

    python run_all.py            # full pipeline
    python run_all.py --quick    # skip the slow held-out calibration refit

Order matters:
  1. simulate    build the demo scenario (seed 20260830) + hidden ground truth
  2. calibrate   fit temperature and triage bands on a DIFFERENT seed (77777)
  3. metrics     detection quality (ROC-AUC) + dark-zone flag precision
  4. validate    confusion matrix, calibration curve, Brier, ECE
  5. evaluate    operational benchmark against the baselines an EOC uses today
  6. real        checks against published real-world facts, not the simulator
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("simulate.py",      "build scenario + hidden ground truth"),
    ("calibrate.py",     "fit calibration on held-out seed"),
    ("metrics.py",       "detection quality (ROC-AUC, ablations)"),
    ("validate.py",      "confusion matrix + calibration curve"),
    ("evaluate.py",      "operational benchmark vs baselines"),
    ("validate_real.py", "real-world data checks"),
]


def main():
    quick = "--quick" in sys.argv
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    failed = []
    for i, (script, what) in enumerate(STEPS, 1):
        if quick and script == "calibrate.py":
            print("\n[%d/%d] SKIP %s (--quick)\n" % (i, len(STEPS), script))
            continue
        print("\n" + "=" * 78)
        print("[%d/%d] %-18s %s" % (i, len(STEPS), script, what))
        print("=" * 78)
        r = subprocess.run([sys.executable, script], cwd=HERE)
        if r.returncode != 0:
            failed.append(script)
            print("!! %s exited %d" % (script, r.returncode))
    print("\n" + "=" * 78)
    print("pipeline finished in %.0fs" % (time.time() - t0))
    if failed:
        print("FAILED: %s" % ", ".join(failed))
        return 1
    print("all steps OK - artefacts written to backend/data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
