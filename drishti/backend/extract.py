"""Claim extraction: raw human text -> structured, machine-comparable claim.

Deliberately rule-based and dependency-free at its core. An optional LLM
refinement layer (free-tier Groq/Gemini) can be switched on, but the system
must degrade gracefully to rules, because in a real EOC the uplink is the
first thing to fail - and on hackathon wifi, so is the API key.

Handles code-mixed Hindi-English (Hinglish) as actually written by citizens.
"""
import re
import config as C

# ------------------------------------------------------------------ lexicons
HAZARD_LEX = {
    "FLOOD": ["paani", "pani", "flood", "water", "doob", "submerg", "waterlog",
              "bund", "nadi", "river", "inundat", "baadh", "badh", "overflow"],
    "STRUCTURAL_COLLAPSE": ["makaan", "ghar gir", "collapse", "building", "gir gaya",
                            "gir gaye", "structural", "malba ghar", "dhah", "rubble",
                            "damage to residential", "wall fell"],
    "LANDSLIDE": ["landslide", "malba", "debris", "slope", "pahaad", "bhuskhalan",
                  "boulder", "mudslide", "slip"],
    "BRIDGE_FAILURE": ["pul", "bridge", "culvert", "washed away bridge", "span"],
    "MEDICAL": ["injur", "doctor", "medical", "ghayal", "hospital", "pregnant",
                "bleeding", "dawai", "medicine", "casualt"],
    "TRAPPED": ["fase", "phase", "trapped", "stuck", "chhat pe", "rooftop", "roof",
                "dabe", "buried", "basement"],
    "POWER_OUT": ["bijli", "power", "electricity", "blackout", "transformer"],
    "FIRE": ["aag", "fire", "burning", "jal raha"],
}

# Severity cues, signed. Positive = worse.
SEV_STRONG = ["completely", "puri", "pura", "total", "washed away", "no survivors",
              "finished", "khatam", "destroyed", "wiped", "sab", "collapsed entirely",
              "submerged", "doob gaya", "catastroph", "mar gaye", "dead", "bodies"]
SEV_MED = ["severe", "major", "heavy", "badly", "bahut", "zyada", "urgent",
           "trapped", "fase", "collapse", "gir gaya", "cut off", "blocked",
           "unsafe", "rising"]
SEV_LOW = ["minor", "thoda", "slight", "knee deep", "small", "little", "partial",
           "waterlogging", "theek", "safe", "no major", "manageable", "open"]

SAFE_CUES = ["is safe", "sab theek", "no major damage", "road is open",
             "minor water only", "all clear", "koi nuksan nahi"]

# Panic / virality markers - these mark a message as LOW independence, not as false.
PANIC_CUES = ["share this", "pls share", "please share", "forward:", "forwarded",
              "urgent!!", "breaking:", "government hiding", "!!", "viral"]

NEG = ["nahi", "no ", "not ", "koi nahi", "cannot", "unconfirmed", "afwah", "rumour", "rumor"]

NUM = re.compile(r"\b(\d{1,5})\b")
DEPTH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre|feet|ft|foot)\b", re.I)


class Claim:
    __slots__ = ("hazards", "severity", "sev_conf", "trapped", "casualties",
                 "depth_m", "is_safe_claim", "panic_score", "specificity")

    def __init__(self):
        self.hazards = []
        self.severity = None      # 0..1 asserted damage, None if not asserted
        self.sev_conf = 0.0       # how strongly the text pins severity
        self.trapped = None
        self.casualties = None
        self.depth_m = None
        self.is_safe_claim = False
        self.panic_score = 0.0
        self.specificity = 0.0    # richer, more specific text -> more informative

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def _count(text, terms):
    return sum(1 for t in terms if t in text)


def extract(raw_text, source=None):
    """Parse one report into a Claim. Pure function, no I/O, ~40us."""
    t = (raw_text or "").lower()
    c = Claim()

    for hz, lex in HAZARD_LEX.items():
        if any(w in t for w in lex):
            c.hazards.append(hz)

    strong, med, low = _count(t, SEV_STRONG), _count(t, SEV_MED), _count(t, SEV_LOW)
    safe = _count(t, SAFE_CUES)
    c.panic_score = min(1.0, _count(t, PANIC_CUES) / 2.0
                        + (0.4 if raw_text and sum(ch.isupper() for ch in raw_text)
                           > 0.35 * max(1, len(raw_text)) else 0.0))

    if safe or (low and not strong and not med):
        c.is_safe_claim = True
        c.severity = 0.10 if low else 0.05
        c.sev_conf = 0.55 + 0.15 * safe
    else:
        score = 1.0 * strong + 0.55 * med - 0.45 * low
        if strong or med or low:
            c.severity = max(0.0, min(1.0, 0.30 + 0.24 * score))
            c.sev_conf = min(0.9, 0.30 + 0.18 * (strong + med + low))
        elif c.hazards:
            # hazard named but no intensity language: weak mid evidence
            c.severity, c.sev_conf = 0.45, 0.20

    # A panicked forward asserts maximal severity but earns little confidence.
    if c.panic_score > 0.4:
        c.sev_conf *= 0.45

    # Explicit official/field severity words map directly onto the state ladder
    for st in C.DAMAGE_STATES:
        if st.lower() in t:
            c.severity = C.SEVERITY_WEIGHT[st]
            c.sev_conf = max(c.sev_conf, 0.7)

    d = DEPTH.search(t)
    if d:
        v = float(d.group(1))
        if "f" in d.group(0).lower():
            v *= 0.3048
        c.depth_m = round(v, 2)

    if "trapped" in t or "fase" in t or "dabe" in t:
        m = NUM.search(t)
        c.trapped = int(m.group(1)) if m and int(m.group(1)) < 5000 else None
    if "dead" in t or "mar gaye" in t or "casualt" in t:
        m = NUM.search(t)
        c.casualties = int(m.group(1)) if m and int(m.group(1)) < 100000 else None

    # Specificity: concrete detail (numbers, coords, landmarks) beats vague panic
    c.specificity = min(1.0, (len(t.split()) / 22.0) * 0.5
                        + (0.25 if NUM.search(t) else 0)
                        + (0.25 if len(c.hazards) >= 1 else 0)
                        - 0.3 * c.panic_score)
    c.specificity = max(0.05, c.specificity)
    return c


# ----------------------------------------------------------- optional LLM tier
def llm_available():
    import os
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def llm_extract_batch(texts, timeout=20):
    """Refine ambiguous reports with a free-tier LLM. Returns list of dicts or
    None on any failure - callers MUST fall back to rules. Never blocks the
    pipeline. Free tiers used: Groq (llama-3.3-70b) or Google Gemini flash.
    """
    import os, json, urllib.request
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    prompt = ("For each numbered disaster report, output strict JSON list of "
              '{"i":int,"hazard":str,"severity":0..1,"safe":bool,"panic":bool}. '
              "Reports:\n" + "\n".join("%d. %s" % (i, t) for i, t in enumerate(texts)))
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "response_format": {"type": "json_object"},
    }).encode()
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode())
        return json.loads(j["choices"][0]["message"]["content"])
    except Exception:
        return None


if __name__ == "__main__":
    samples = [
        "SHARE THIS!! dam has broken, Chandrapuri completely finished, thousands dead",
        "Team-4 verification Ukhimath complete - major damage, 12 casualties observed.",
        "paani gaon me ghus gaya hai, ghar tak aa gaya",
        "Chandrapuri is safe, minor water only",
        "...Gaurikund... water... need boat... over",
        "family trapped on rooftop Bhiri, water still rising",
    ]
    for s in samples:
        c = extract(s)
        print("\n%s\n  -> hazards=%s sev=%s conf=%.2f safe=%s panic=%.2f spec=%.2f"
              % (s[:70], c.hazards, c.severity, c.sev_conf, c.is_safe_claim,
                 c.panic_score, c.specificity))
