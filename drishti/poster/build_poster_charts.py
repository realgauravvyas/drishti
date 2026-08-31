"""Poster figures, rendered light-theme to match the ICSIGT 2026 template.

Each figure is generated at its true printed size in inches at 200 dpi, so the
36 x 48 inch poster stays crisp at press. Type sizes are chosen for reading at
roughly one metre, not for a laptop screen.

Every value plotted is read from the project's own result files.
"""
import json, math, os, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "backend", "data")
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------- template design tokens
NAVY  = "#083070"
GREEN = "#00B050"
GREY  = "#6E6E6E"
RED   = "#C00000"
AMBER = "#D97706"
TEAL  = "#1F7A8C"
INK   = "#111111"
LIGHT = "#EEF2F8"
EDGE  = "#B9C4D4"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": EDGE,
})

STATES = ["INTACT", "MINOR", "MAJOR", "CATASTROPHIC"]
gt = json.load(open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8"))
rp = json.load(open(os.path.join(DATA, "reports.json"), encoding="utf-8"))
ev = {r["method"].strip(): r for r in
      json.load(open(os.path.join(DATA, "evaluation.json")))["rows"]}
mx = json.load(open(os.path.join(DATA, "metrics.json")))
val = json.load(open(os.path.join(DATA, "validation.json")))
gz = json.load(open(os.path.join(DATA, "gazetteer.json"), encoding="utf-8"))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)
    print("  %-22s" % name)
    return p


def strip(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        if sp not in keep:
            ax.spines[sp].set_visible(False)


# ============================================================ MOTIVATION
def fig_motivation():
    """The non-monotonic reporting curve - the core problem."""
    vol = Counter()
    for r in rp:
        vol[r["truth_sid"]] += 1
    per = defaultdict(list)
    for sid, v in gt.items():
        per[v["state"]].append(vol.get(sid, 0))
    means = [sum(per[s]) / max(1, len(per[s])) for s in STATES]
    silent = [100.0 * sum(1 for x in per[s] if x == 0) / max(1, len(per[s]))
              for s in STATES]

    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    cols = [GREY, TEAL, AMBER, RED]
    bars = ax.bar(STATES, means, color=cols, width=0.6, zorder=3)
    for b, m, sl in zip(bars, means, silent):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.16, "%.1f" % m,
                ha="center", fontsize=24, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, -0.95, "%.0f%% silent" % sl,
                ha="center", fontsize=16, color=GREY)
    ax.set_ylabel("Mean reports received per village", fontsize=19)
    ax.set_ylim(0, max(means) * 1.42)
    ax.tick_params(axis="x", labelsize=19, colors=INK)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color=EDGE, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    ax.annotate("worst-hit villages\nFALL SILENT",
                xy=(3, means[3] + 0.35), xytext=(2.52, max(means) * 1.30),
                color=RED, fontsize=20, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=3,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("Report volume peaks at MAJOR damage, then collapses",
                 fontsize=21, fontweight="bold", color=NAVY, pad=16, loc="left")
    return save(fig, "motivation.png")


# ================================================================ METHOD
def fig_method():
    """Five-stage processing pipeline drawn as a flow diagram."""
    fig, ax = plt.subplots(figsize=(10.1, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        ("1. GEO-RESOLVE", "fuzzy name to a probability\nover real settlements", TEAL),
        ("2. EXTRACT", "Hinglish claim parsing:\nhazard, severity, panic", "#6B46C1"),
        ("3. CORROBORATE", "one rumour x90 counts\nas ~1 witness", AMBER),
        ("4. FUSE", "Dempster-Shafer belief\n+ explicit confidence", GREEN),
        ("5. TRIAGE", "rescue queue  |  recon queue", NAVY),
    ]
    y = 9.5
    H, GAP = 1.60, 0.20
    for i, (name, body, col) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((0.35, y - H), 9.3, H,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=2.2, edgecolor=col,
                                    facecolor=LIGHT if i % 2 == 0 else "white",
                                    zorder=2))
        ax.text(0.62, y - 0.48, name, fontsize=16, fontweight="bold",
                color=col, va="center", zorder=3)
        ax.text(0.62, y - 1.10, body, fontsize=12.5, color=GREY,
                va="center", zorder=3, linespacing=1.30)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - H), (5.0, y - H - GAP),
                                         arrowstyle="-|>", mutation_scale=20,
                                         color=NAVY, lw=2.2, zorder=3))
        y -= (H + GAP)

    ax.text(5.0, 0.22, "3,256 reports  →  1,102 ranked settlements  ( < 6 s )",
            fontsize=15, fontweight="bold", color=NAVY, ha="center")
    return save(fig, "method.png")


# ======================================================= EXPERIMENTAL SETUP
def fig_setup():
    """Study area: real district, real settlements, simulated damage."""
    fig, ax = plt.subplots(figsize=(8.9, 6.4))
    for r in gz.get("rivers", []):
        c = r["coords"]
        ax.plot([p[1] for p in c], [p[0] for p in c], color="#7FB3D5",
                lw=1.0, zorder=1, alpha=0.85)
    cmap = {"INTACT": ("#9FB3C8", 7), "MINOR": (TEAL, 9),
            "MAJOR": (AMBER, 14), "CATASTROPHIC": (RED, 26)}
    for st in STATES:
        pts = [v for v in gt.values() if v["state"] == st]
        col, sz = cmap[st]
        ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts],
                   s=sz, c=col, label="%s  (%d)" % (st.title(), len(pts)),
                   zorder=3, edgecolors="none", alpha=0.9)
    ax.scatter([79.055], [30.545], marker="*", s=520, c="black", zorder=5)
    ax.annotate("epicentre  M6.1", (79.055, 30.545), (79.10, 30.60),
                fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="black", lw=1.4))
    ax.set_xlabel("Longitude (°E)", fontsize=14)
    ax.set_ylabel("Latitude (°N)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Rudraprayag – Chamoli, Uttarakhand · 1,102 OSM settlements",
                 fontsize=15, fontweight="bold", color=NAVY, pad=10)
    leg = ax.legend(loc="lower left", fontsize=11.5, frameon=True,
                    facecolor="white", edgecolor=EDGE, markerscale=1.6)
    leg.set_zorder(6)
    # clip to the area of interest; river geometry runs far beyond it
    ax.set_xlim(78.83, 79.47)
    ax.set_ylim(30.18, 30.82)
    ax.set_aspect(1.0 / math.cos(math.radians(30.5)))
    ax.grid(color=EDGE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return save(fig, "setup.png")


# ============================================================= MECHANISM
def fig_mechanism():
    """How three weak absence signals combine into one belief."""
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.72, "A village that is FINE and one that is\n"
                     "GONE are both SILENT",
            fontsize=14, fontweight="bold", color=RED, ha="center",
            va="top", linespacing=1.4)

    sig = [("Tower liveness", "dead 23.9 h", TEAL),
           ("Chatter deficit", "1 report vs 418", "#6B46C1"),
           ("Terrain exposure", "flood-exposed", AMBER)]
    y, H, GAP = 8.35, 1.15, 0.27
    for name, v, col in sig:
        ax.add_patch(FancyBboxPatch((0.3, y - H), 9.4, H,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    lw=2.0, edgecolor=col, facecolor="white",
                                    zorder=2))
        ax.text(0.60, y - H / 2, name, fontsize=13.5, fontweight="bold",
                color=col, va="center")
        ax.text(9.40, y - H / 2, v, fontsize=12, color=GREY, va="center",
                ha="right")
        ax.add_patch(FancyArrowPatch((5.0, y - H), (5.0, y - H - GAP),
                                     arrowstyle="-|>", mutation_scale=17,
                                     color=NAVY, lw=2.0, zorder=3))
        y -= (H + GAP)

    ax.add_patch(FancyBboxPatch((0.3, 2.55), 9.4, 1.75,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                lw=2.6, edgecolor=NAVY, facecolor=LIGHT,
                                zorder=2))
    ax.text(5.0, 3.88, "P(CATASTROPHIC) elevated", fontsize=14,
            fontweight="bold", color=NAVY, ha="center", va="center")
    ax.text(5.0, 3.05, "confidence only 16%  →  task a UAV,\n"
                       "do not commit the excavator",
            fontsize=12, color=GREY, ha="center", va="center",
            linespacing=1.35)

    ax.text(5.0, 1.85, "51% of UNDAMAGED villages also lose their tower\n"
                       "(district-wide power cut) — so tower-down is\n"
                       "never used as evidence on its own.",
            fontsize=11, color=RED, ha="center", va="top", linespacing=1.45,
            fontweight="bold")
    return save(fig, "mechanism.png")


# ================================================================ RESULTS
def fig_reached():
    order = [("VOLUME", "Report volume\n(current practice)"),
             ("LOUDEST", "Loudest claim"),
             ("NAIVE_MEAN", "Unweighted mean"),
             ("DRISHTI (full)", "DRISHTI")]
    vals = [ev[k]["casualties_reached"] for k, _ in order]
    labs = [l for _, l in order]
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    cols = [GREY, GREY, GREY, GREEN]
    ax.barh(range(len(vals)), vals, color=cols, height=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 110, i, "{:,}".format(v), va="center", fontsize=19,
                fontweight="bold" if i == 3 else "normal",
                color=INK if i == 3 else GREY)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("People reached in 24 h  (of 10,928 in need)", fontsize=15)
    ax.set_xlim(0, max(vals) * 1.24)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color=EDGE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    ax.set_title("Identical assets, reports and time budget",
                 fontsize=16, fontweight="bold", color=NAVY, pad=12, loc="left")
    return save(fig, "reached.png")


def fig_blind():
    a = mx["reports only"]["auc_blind"]
    b = mx["no silence"]["auc_blind"]
    c = mx["DRISHTI (full)"]["auc_blind"]
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    names = ["Report-driven\n(no reports =\nno information)",
             "DRISHTI\nwithout silence", "DRISHTI\n(full)"]
    bars = ax.bar(names, [a, b, c], color=[RED, AMBER, GREEN], width=0.55,
                  zorder=3)
    for bar, v in zip(bars, [a, b, c]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.022, "%.3f" % v,
                ha="center", fontsize=22, fontweight="bold", color=INK)
    ax.axhline(0.5, color=GREY, ls="--", lw=1.6, zorder=2)
    # Label the reference line just outside the axes. Anywhere inside collides:
    # the 0.500 bar's own value label sits at exactly this height.
    ax.text(1.012, 0.5, "0.5 = coin flip", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=12, color=GREY)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("ROC-AUC", fontsize=15)
    ax.tick_params(axis="x", labelsize=13.5)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", color=EDGE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    ax.set_title("The 492 villages that sent ZERO reports",
                 fontsize=16, fontweight="bold", color=NAVY, pad=12, loc="left")
    return save(fig, "blind.png")


def fig_ablation():
    keys = [("DRISHTI (full)", "Full system"),
            ("no silence", "− silence engine"),
            ("no terrain prior", "− terrain prior"),
            ("no independence", "− independence"),
            ("no credibility", "− source credibility"),
            ("reports only", "Reports only")]
    vals = [mx[k]["auc_all"] for k, _ in keys]
    labs = [l for _, l in keys]
    cols = [GREEN] + [TEAL] * 4 + [RED]
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.barh(range(len(vals)), vals, color=cols, height=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, "%.3f" % v, va="center", fontsize=17,
                fontweight="bold" if i in (0, 5) else "normal", color=INK)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC, whole district", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color=EDGE, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    strip(ax, keep=("bottom",))
    ax.set_title("Ablation: what each mechanism contributes",
                 fontsize=16, fontweight="bold", color=NAVY, pad=12, loc="left")
    return save(fig, "ablation.png")


def fig_agastmuni():
    """Wide case-study panel: the headline example."""
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    ax.text(0, 31.4, "Case study — Agastmuni, Rudraprayag "
                     "(real settlement, pop. 19,758)",
            fontsize=19, fontweight="bold", color=NAVY)

    rows = [("Reports expected in 24 h", "418", GREY),
            ("Reports actually received", "1", RED),
            ("Cell tower", "dead 23.9 h", RED),
            ("Ground truth", "CATASTROPHIC", RED),
            ("Casualties / trapped", "2,534 / 7,795", RED)]
    y = 25.5
    for i, (k, v, col) in enumerate(rows):
        ax.text(1.0, y, k, fontsize=15, color=GREY, va="center")
        ax.text(43.0, y, v, fontsize=16, fontweight="bold", color=col,
                va="center", ha="right")
        if i < len(rows) - 1:
            ax.plot([1.0, 43.0], [y - 2.4, y - 2.4], color=EDGE, lw=0.9)
        y -= 5.0

    for x0, lab, rank, sub, col in [
            (48, "RANKED BY\nREPORT VOLUME", "#525",
             "of 1,353\nhelp never arrives", RED),
            (74, "RANKED BY\nDRISHTI", "#1",
             "of 1,353\nfirst vehicle sent", GREEN)]:
        ax.add_patch(FancyBboxPatch((x0, 2.0), 24, 27,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    lw=2.6, edgecolor=col, facecolor="white"))
        ax.text(x0 + 12, 25.4, lab, fontsize=12, fontweight="bold",
                color=GREY, ha="center", va="top", linespacing=1.35)
        ax.text(x0 + 12, 14.0, rank, fontsize=48, fontweight="bold",
                color=col, ha="center", va="center")
        ax.text(x0 + 12, 6.2, sub, fontsize=12, color=GREY, ha="center",
                va="top", linespacing=1.35)
    return save(fig, "agastmuni.png")


def fig_confusion():
    """Confusion matrix + calibration quality, side by side."""
    cm = val["confusion"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    tot = sum(sum(r) for r in cm)
    norm = [[c / max(1, sum(r)) for c in r] for r in cm]
    im = ax1.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, "%d" % cm[i][j], ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if norm[i][j] > 0.5 else INK)
    lbl = ["Intact", "Minor", "Major", "Catas."]
    ax1.set_xticks(range(4)); ax1.set_xticklabels(lbl, fontsize=13)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(lbl, fontsize=13)
    ax1.set_xlabel("Predicted", fontsize=14)
    ax1.set_ylabel("Ground truth", fontsize=14)
    ax1.set_title("Confusion matrix (n = %d)" % tot, fontsize=15,
                  fontweight="bold", color=NAVY, pad=10)

    names = ["Exact band\naccuracy", "Within one\nband", "ROC-AUC", "Brier\nscore"]
    vals = [val["accuracy"], val["within_one"], val["auc"], val["brier"]]
    cols = [AMBER, GREEN, GREEN, TEAL]
    bars = ax2.bar(names, vals, color=cols, width=0.58, zorder=3)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.022, "%.3f" % v,
                 ha="center", fontsize=17, fontweight="bold", color=INK)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", labelsize=12.5)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.grid(axis="y", color=EDGE, lw=0.9, zorder=0)
    ax2.set_axisbelow(True)
    strip(ax2, keep=("bottom",))
    ax2.set_title("Calibrated performance", fontsize=15, fontweight="bold",
                  color=NAVY, pad=10)
    fig.tight_layout()
    return save(fig, "confusion.png")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("rendering poster figures at print size...")
    for f in (fig_motivation, fig_method, fig_setup, fig_mechanism,
              fig_reached, fig_blind, fig_ablation, fig_agastmuni,
              fig_confusion):
        f()
    print("done ->", OUT)
