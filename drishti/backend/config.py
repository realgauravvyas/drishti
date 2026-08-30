"""DRISHTI - central configuration.

All tunables live here so they can be justified in the pitch and swapped
by a district administrator without touching engine code.
"""

# ---------------------------------------------------------------- hypotheses
# The frame of discernment for damage state. Order matters (ordinal severity).
DAMAGE_STATES = ["INTACT", "MINOR", "MAJOR", "CATASTROPHIC"]
SEVERITY_WEIGHT = {"INTACT": 0.0, "MINOR": 0.25, "MAJOR": 0.65, "CATASTROPHIC": 1.0}

HAZARDS = ["FLOOD", "STRUCTURAL_COLLAPSE", "LANDSLIDE", "BRIDGE_FAILURE",
           "MEDICAL", "FIRE", "POWER_OUT", "TRAPPED"]

# ------------------------------------------------------------ source priors
# Beta(alpha, beta) reliability prior per channel. These encode institutional
# knowledge: a field team's word is worth far more than an unverified retweet.
# alpha = prior "true" reports, beta = prior "false" reports.
SOURCE_PRIORS = {
    "FIELD_TEAM":  (18.0, 2.0),    # trained responders, GPS-tagged      ~0.90
    "OFFICIAL":    (14.0, 3.0),    # block/tehsil admin, slow but solid  ~0.82
    "HAM_RADIO":   (9.0,  3.0),    # licensed operators, disciplined     ~0.75
    "SAT_PHONE":   (7.0,  3.0),    # genuine but garbled/fragmentary     ~0.70
    "IVR_CALL":    (5.0,  4.0),    # citizen calls, panicked but present ~0.56
    "SOCIAL":      (2.5,  6.0),    # unverified, rumour-amplified        ~0.29
}
DEFAULT_PRIOR = (3.0, 5.0)

# How much a single report from a channel can move belief (mass discount).
# Even a perfect source never gets mass 1.0 - epistemic humility.
MAX_EVIDENCE_MASS = 0.85

# --------------------------------------------- online credibility learning
CRED_MIN_CONFIDENCE = 0.55        # only settled consensus may teach
CRED_MIN_INDEPENDENT_MASS = 0.40  # ...and it must be backed by OTHER channels
CRED_MAX_LEARNED = 3.0            # bound: a prior can be revised, not swamped

# ------------------------------------------------------- corroboration graph
DEDUP_TEXT_SIM = 0.62        # TF-IDF cosine above which two reports may be dupes
DEDUP_RADIUS_KM = 3.0        # and must be within this distance
DEDUP_WINDOW_MIN = 90        # and this many minutes apart

# Independence discounting: N reports from the SAME channel about the SAME
# incident are correlated (one rumour, many mouths). The k-th report from a
# channel contributes INDEPENDENCE_DECAY^(k-1) of a full report.
INDEPENDENCE_DECAY = 0.45

# --------------------------------------------------------------- silence
# A settlement that *should* be reporting and isn't is the core signal.
BASELINE_REPORTS_PER_1K_PER_HR = 0.9   # normal civic chatter rate
SILENCE_MIN_HOURS = 2.0                # need this long a gap to call it silence
SILENCE_ALARM_Z = 2.0                  # z-score at which silence becomes alarming
# A blackout is near-TOTAL absence, not merely "fewer reports than usual".
# Without this gate the detector lights up most of the district and stops
# discriminating - measured: it fired on 96% of settlements at ratio<1.0.
SILENCE_MIN_EXPECTED = 3.0             # never infer from places we would rarely hear from
SILENCE_DARK_RATIO = 0.20              # observed/expected below this = dark
SILENCE_DARK_ABS = 2.0                 # ...and at most this many raw reports
# P(blackout is structural | tower is down) before terrain is considered.
# Towers also die from power loss alone, so this deliberately starts low.
TOWER_DOWN_BASE_RISK = 0.30
# risk above which a blackout becomes an actionable DARK ZONE tasking
DARK_ESCALATE_RISK = 0.72
# cap on how much belief a blackout inference may move. Tower loss is now a
# weak signal (the grid is down district-wide), so it must not dominate.
SILENCE_MASS_CAP = 0.20

# ------------------------------------------------------------ prioritisation
# Expected Unassisted Casualties weighting
CASUALTY_RATE = {"INTACT": 0.0, "MINOR": 0.002, "MAJOR": 0.02, "CATASTROPHIC": 0.11}
# Survival decays with time - the golden 72 hours, steepest in the first 24.
GOLDEN_WINDOW_HRS = 72.0

# ------------------------------------------------------------------- assets
ASSET_TYPES = {
    "BOAT":      {"label": "Inflatable rescue boat", "speed_kmph": 18,
                  "handles": ["FLOOD", "TRAPPED"],              "capacity": 120},
    "EXCAVATOR": {"label": "Heavy excavator",        "speed_kmph": 22,
                  "handles": ["STRUCTURAL_COLLAPSE", "LANDSLIDE", "TRAPPED"], "capacity": 60},
    "MEDICAL":   {"label": "Mobile medical team",    "speed_kmph": 35,
                  "handles": ["MEDICAL", "TRAPPED", "STRUCTURAL_COLLAPSE"],   "capacity": 200},
    "DRONE":     {"label": "Recon UAV",              "speed_kmph": 60,
                  "handles": ["RECON"],                          "capacity": 0},
}

# Road passability by reported flood depth (metres) / damage
IMPASSABLE_DEPTH_M = 0.7

# Route degradation keyed to P(MAJOR or CATASTROPHIC) - a calibrated
# probability, so these stay meaningful across recalibration.
ROUTE_DEGRADE_P = 0.40     # below this, the road is treated as clear
BRIDGE_BLOCK_P = 0.62      # a span we mostly believe is damaged is not crossed
ROAD_BLOCK_P = 0.75        # surface roads need stronger belief to close
ROUTE_INFLUENCE_KM = 2.0   # damage this close to a segment affects it

# Commit a physical asset when the CALIBRATED probability of severe damage
# justifies the trip. Confidence in this district is genuinely low (median
# 0.11) because most settlements are barely reported on - so gating rescue on
# high confidence would deploy almost nothing, which is the failure the
# problem statement describes.
RESCUE_MIN_P_SEVERE = 0.45
RESCUE_MIN_CONF = 0.10
RESCUE_CONF_OVERRIDE_P = 0.55   # strong enough belief to go without confidence


# --------------------------------------------------- prior self-calibration
# The terrain prior is scored against settlements where ground evidence is
# strong, then shrunk toward climatology by how much skill it shows.
PRIOR_ANCHOR_MIN_MASS = 0.35   # evidence mass needed to serve as an anchor
PRIOR_MIN_ANCHORS = 25         # below this we cannot judge the prior at all
PRIOR_TRUST_DEFAULT = 0.45     # tentative trust before enough anchors exist
PRIOR_SKILL_FULL = 0.35        # skill score that earns full trust

# strength of the terrain tilt applied on top of the district base rate
PRIOR_TILT = 26.0

# District base rate over damage states for settlements we have NOT heard from.
# This must NOT be learned from well-reported settlements: those are the ones
# that generated reports, which are the damaged ones, so the learned mix comes
# out ~90% MAJOR and is then wrongly applied to quiet, intact villages. It is
# an operator-set expectation (scale of event declared by the SDMA), and the
# dashboard exposes it so it can be challenged.
DISTRICT_BASE_RATE = [0.30, 0.32, 0.24, 0.14]   # INTACT, MINOR, MAJOR, CATASTROPHIC

# Operational triage label from the belief's expected severity (DSI). Kept
# separate from the argmax of the distribution: the distribution is the honest
# belief, the label is a decision, and decisions need stable thresholds.
# Thresholds are the quantiles of the belief-derived DSI that reproduce
# DISTRICT_BASE_RATE. DSI is an expectation over a distribution, so it shrinks
# away from the extremes - bands set on the raw 0..1 severity scale would
# never fire CATASTROPHIC at all (measured: DSI max 0.756).
DSI_BANDS = [(0.30, "INTACT"), (0.40, "MINOR"), (0.54, "MAJOR"),
             (1.01, "CATASTROPHIC")]


# ------------------------------------------------------- seismic intensity
import math as _math

SEIS_DEPTH_KM = 8.0        # typical Himalayan crustal focal depth
SEIS_A, SEIS_B, SEIS_C = -0.31, 1.5, -2.7


def mmi_at(magnitude, dist_km, depth_km=SEIS_DEPTH_KM):
    """Modified Mercalli intensity from magnitude and epicentral distance.

    Log-distance form on HYPOCENTRAL distance, so intensity saturates near the
    epicentre instead of diverging. Anchored to published Himalayan intensity
    reports: an M6.8 gives about MMI VII within a few km, VI at 30 km and
    IV-V at 100 km. The earlier form under-predicted the near field by ~3
    intensity units, which was caught by validate_real.py.
    """
    r = _math.sqrt(dist_km * dist_km + depth_km * depth_km)
    return max(0.0, SEIS_A + SEIS_B * magnitude + SEIS_C * _math.log10(max(1.0, r)))


def shake_norm(mmi):
    """MMI -> [0,1] structural shaking driver. MMI X is total destruction."""
    return max(0.0, min(1.0, (mmi - 3.0) / 6.0))
