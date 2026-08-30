"""End-to-end UI test with Playwright: loads the dashboard, exercises every
tab and the time scrubber, captures console errors, and saves screenshots.

Run:  python uitest.py
"""
import os, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "screenshots"))
URL = "http://127.0.0.1:8000/"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(SHOTS, exist_ok=True)
    errors, failed = [], []

    with sync_playwright() as pw:
        br = pw.chromium.launch()
        pg = br.new_page(viewport={"width": 1680, "height": 950})
        pg.on("console", lambda m: errors.append("%s: %s" % (m.type, m.text))
              if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
        pg.on("requestfailed",
              lambda r: failed.append("%s %s" % (r.url, r.failure)))

        print("loading %s" % URL)
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)

        # wait for the engine to warm and the app to appear
        pg.wait_for_selector("#app", state="visible", timeout=180000)
        pg.wait_for_function("() => document.querySelectorAll('.kpi').length > 3",
                             timeout=120000)
        time.sleep(3.5)                      # let the plan request land
        print("app rendered")

        kpis = pg.eval_on_selector_all(".kpi", "els => els.map(e => e.innerText)")
        print("KPIs: %s" % " | ".join(k.replace("\n", " ") for k in kpis))

        markers = pg.evaluate("() => document.querySelectorAll('canvas').length")
        print("map canvas layers: %d" % markers)

        for tab, name in [("triage", "triage"), ("dark", "dark-zones"),
                          ("recon", "recon"), ("plan", "assets"),
                          ("bench", "benchmark"), ("about", "method")]:
            pg.click('.tab[data-k="%s"]' % tab)
            time.sleep(1.4)
            txt = pg.inner_text("#pane")
            print("  tab %-10s -> %d chars" % (tab, len(txt)))
            if len(txt) < 40:
                errors.append("tab %s rendered almost nothing" % tab)
            pg.screenshot(path=os.path.join(SHOTS, "tab-%s.png" % name))

        # click the top triage row -> evidence drill-down
        pg.click('.tab[data-k="triage"]')
        time.sleep(1.0)
        rows = pg.query_selector_all(".row[data-sid]")
        print("triage rows: %d" % len(rows))
        if rows:
            rows[0].click()
            time.sleep(2.2)
            ev = pg.inner_text("#pane")
            print("  evidence pane -> %d chars" % len(ev))
            pg.screenshot(path=os.path.join(SHOTS, "tab-evidence.png"))

        # find a dark zone and open it - the headline demo moment
        pg.click('.tab[data-k="dark"]')
        time.sleep(1.2)
        drows = pg.query_selector_all(".row[data-sid]")
        print("dark-zone rows: %d" % len(drows))
        if drows:
            drows[0].click()
            time.sleep(2.2)
            pg.screenshot(path=os.path.join(SHOTS, "dark-zone-evidence.png"))

        # time scrubber
        for t in (3, 9, 24):
            pg.evaluate("""(t)=>{const s=document.getElementById('scrub');
                s.value=t; s.dispatchEvent(new Event('change'));}""", t)
            time.sleep(3.0)
            lbl = pg.inner_text("#tnow")
            k = pg.eval_on_selector_all(".kpi .v", "e=>e.map(x=>x.innerText)")
            print("  scrub %-3s -> %s  kpis=%s" % (t, lbl, k))
        pg.screenshot(path=os.path.join(SHOTS, "overview.png"), full_page=False)

        br.close()

    print("\n" + "=" * 60)
    if failed:
        print("FAILED REQUESTS (%d):" % len(failed))
        for f in failed[:10]:
            print("   ", f)
    if errors:
        print("CONSOLE ERRORS (%d):" % len(errors))
        for e in errors[:15]:
            print("   ", e)
        sys.exit(1)
    _compress(SHOTS)
    print("UI TEST PASSED - no console errors")
    print("screenshots -> docs/screenshots/")


def _compress(d):
    """Keep the repo lean: full-resolution map screenshots are ~1 MB each."""
    try:
        from PIL import Image
    except ImportError:
        return
    for f in os.listdir(d):
        if not f.endswith(".png"):
            continue
        p = os.path.join(d, f)
        im = Image.open(p).convert("RGB")
        w = 1400 if f == "overview.png" else 1200
        if im.width > w:
            im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        im.save(p, "PNG", optimize=True)
        if os.path.getsize(p) > 380_000:
            im.convert("P", palette=Image.ADAPTIVE,
                       colors=192).save(p, "PNG", optimize=True)


if __name__ == "__main__":
    main()
