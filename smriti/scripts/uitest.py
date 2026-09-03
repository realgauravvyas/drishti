#!/usr/bin/env python
"""Drive the real UI in a real browser, end to end.

The API tests prove the contract; this proves the product. It starts a server on
a scratch data directory, then does what two humans would do:

  organiser -> create an album, drop a folder of photos in, watch it index
  guest     -> open the share link, add a selfie, get their photos, download

Every step is asserted against what the page actually renders, and screenshots
are written to ``--out`` so the result can be looked at.

    pip install playwright && python -m playwright install chromium
    python scripts/uitest.py --photos ./album --selfie ./me.jpg
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"server never answered at {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True, type=Path, help="folder of album photos")
    parser.add_argument("--selfie", required=True, type=Path)
    parser.add_argument("--engine", default=os.environ.get("SMRITI_ENGINE", "sface"))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    parser.add_argument("--limit", type=int, default=40, help="max photos to upload")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    from playwright.sync_api import expect, sync_playwright

    photos = sorted(p for p in args.photos.iterdir()
                    if p.suffix.lower() in SUFFIXES)[:args.limit]
    if not photos:
        sys.exit(f"no photos in {args.photos}")
    if not args.selfie.exists():
        sys.exit(f"no selfie at {args.selfie}")
    args.out.mkdir(parents=True, exist_ok=True)

    port = free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="smriti-uitest-"))
    env = {**os.environ, "SMRITI_DATA_DIR": str(data_dir), "SMRITI_ENGINE": args.engine,
           "SMRITI_PORT": str(port), "SMRITI_HOST": "127.0.0.1"}
    # Reuse the already-downloaded weights so the test does not re-fetch 40 MB.
    env.setdefault("SMRITI_WEIGHTS_DIR", str(ROOT / "data" / "weights"))

    # Server output goes to a file, never to an unread PIPE: uvicorn logs every
    # request, and once the OS pipe buffer fills with nobody draining it the
    # server blocks mid-write and the whole run deadlocks.
    log_path = data_dir / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "run.py", "--port", str(port)],
                              cwd=ROOT, env=env,
                              stdout=log_file, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    failures: list[str] = []
    try:
        wait_for(f"{base}/api/health")
        print(f"  server up on {base} (engine {args.engine}, {len(photos)} photos)")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda e: failures.append(f"JS error: {e}"))
            page.on("console", lambda m: failures.append(f"console.error: {m.text}")
                    if m.type == "error" else None)

            # ---------------- organiser ----------------------------------
            page.goto(f"{base}/admin.html")
            expect(page.locator("#step-create")).to_be_visible()
            page.fill("#name", "Goa trip 2026")
            page.fill("#notes", "Day 1-3, shot by Ravi")
            page.select_option("#retention", "30")
            page.screenshot(path=args.out / "01-create.png")
            page.click("#create-btn")

            expect(page.locator("#step-dash")).to_be_visible(timeout=15_000)
            share_code = page.locator("#share-code").inner_text().strip()
            assert len(share_code) == 8, f"bad share code {share_code!r}"
            print(f"  organiser: album created, share code {share_code}")

            page.set_input_files("#file", [str(p) for p in photos])
            expect(page.locator("#upload-text")).to_contain_text("added", timeout=180_000)
            print(f"  organiser: {page.locator('#upload-text').inner_text().strip()}")

            # Indexing runs in the background; the stats panel polls until done.
            # "Pending 0" alone is not the finish line -- it is transiently true
            # between upload batches, before the later photos even exist.
            def stat(label: str) -> int:
                text = page.locator(".stat", has_text=label).locator("b").inner_text()
                return int(text.replace(",", "").strip() or 0)

            deadline = time.time() + 300
            while time.time() < deadline:
                if stat("Photos") == len(photos) and stat("Pending") == 0:
                    break
                time.sleep(1)
            else:
                failures.append("indexing did not finish within 5 minutes")

            page.click("#refresh-gallery")
            expect(page.locator("#gallery .tile").first).to_be_visible(timeout=30_000)
            shown = page.evaluate(
                "() => Object.fromEntries([...document.querySelectorAll('#stats .stat')]"
                ".map(s => [s.querySelector('span').textContent, s.querySelector('b').textContent]))")
            print(f"  organiser: stats {shown}")
            assert stat("Indexed") == len(photos), f"not every photo indexed: {shown}"
            assert stat("Failed") == 0, f"photos failed to index: {shown}"

            page.click("#load-people")
            expect(page.locator("#people p")).to_contain_text("distinct", timeout=120_000)
            print(f"  organiser: {page.locator('#people p').inner_text().strip()[:70]}")
            page.screenshot(path=args.out / "02-admin.png", full_page=True)

            # ---------------- guest --------------------------------------
            guest = browser.new_page(viewport={"width": 430, "height": 932})  # a phone
            guest.on("pageerror", lambda e: failures.append(f"JS error (guest): {e}"))
            guest.goto(f"{base}/index.html")
            guest.fill("#code", share_code)
            guest.screenshot(path=args.out / "03-join.png")
            guest.click("#go")

            expect(guest.locator("#step-selfie")).to_be_visible(timeout=15_000)
            guest.set_input_files("#file", str(args.selfie))
            expect(guest.locator("#selfies .tile")).to_have_count(1)
            guest.screenshot(path=args.out / "04-selfie.png")
            guest.click("#search")

            expect(guest.locator("#step-results")).to_be_visible(timeout=60_000)
            title = guest.locator("#results-title").inner_text()
            found = guest.locator("#tiers .tile").count()
            print(f"  guest: {title!r} -- {found} photos rendered")
            assert found > 0, "the guest matched nothing"
            guest.screenshot(path=args.out / "05-results.png", full_page=True)

            # Pre-selection, selection controls and the ZIP download.
            preselected = guest.locator("#tiers .tile.picked").count()
            print(f"  guest: {preselected} pre-selected as 'sure'")
            guest.click("#select-all")
            assert guest.locator("#tiers .tile.picked").count() == found, "select all failed"
            guest.click("#select-none")
            assert guest.locator("#tiers .tile.picked").count() == 0, "clear failed"

            guest.locator("#tiers .tile").first.click()
            with guest.expect_download(timeout=120_000) as download:
                guest.click("#download")
            saved = args.out / "guest-download.zip"
            download.value.save_as(saved)
            import zipfile
            names = zipfile.ZipFile(saved).namelist()
            assert len(names) == 1, f"expected 1 photo in the ZIP, got {names}"
            print(f"  guest: downloaded {saved.name} containing {names}")
            saved.unlink()

            # A thumbnail must actually load, not 404 behind a broken token.
            broken = guest.evaluate(
                "() => [...document.querySelectorAll('#tiers .tile img')]"
                ".filter(i => !i.complete || i.naturalWidth === 0).length")
            assert broken == 0, f"{broken} thumbnails failed to load"
            print("  guest: every thumbnail loaded")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()

    if failures:
        print(f"\n  server log: {log_path}")
        print("\n  FAILURES:")
        for failure in failures:
            print(f"    {failure}")
        return 1
    print(f"\n  UI test passed. Screenshots in {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
