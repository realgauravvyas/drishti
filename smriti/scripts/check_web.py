#!/usr/bin/env python
"""Static checks for the frontend, which has no build step to catch anything.

Three classes of bug are invisible until a real user hits them, because they
fail silently in the browser console:

* an id in ``$('#thing')`` that no page actually contains,
* a name imported from ``common.js`` that it does not export,
* an unbalanced brace, which kills the whole module.

This is not a JavaScript parser. It is the 60 lines of checking that catch the
mistakes a build step would have caught.

    python scripts/check_web.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"

# Which script belongs to which page, so ids can be checked against the right HTML.
PAGES = {
    "static/find.js": ["find.html"],
    "static/admin.js": ["admin.html"],
}

ID_RE = re.compile(r"""\$\(\s*['"]#([A-Za-z0-9_-]+)['"]""")
HTML_ID_RE = re.compile(r"""\bid\s*=\s*['"]([A-Za-z0-9_-]+)['"]""")
# Identifiers may start with or contain '$' and '_', which \w does not cover.
EXPORT_RE = re.compile(r"^export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([\w$]+)",
                       re.MULTILINE)
EXPORT_LIST_RE = re.compile(r"^export\s*\{([^}]*)\}", re.MULTILINE)
IMPORT_RE = re.compile(r"import\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", re.DOTALL)
SRC_RE = re.compile(r"""(?:src|href)\s*=\s*['"](/static/[^'"]+)['"]""")


def balanced(text: str, path: Path, problems: list[str]) -> None:
    """Bracket balance, ignoring strings, template literals and comments."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack, i, line = [], 0, 1
    quote = None
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            line += 1
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif text.startswith("//", i):
            i = text.find("\n", i)
            if i == -1:
                break
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            line += text[i:end].count("\n") if end != -1 else 0
            i = end + 2 if end != -1 else len(text)
            continue
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                problems.append(f"{path.name}:{line}: unbalanced '{ch}'")
                return
            stack.pop()
        i += 1
    if stack:
        problems.append(f"{path.name}: unclosed '{stack[-1][0]}' opened at line {stack[-1][1]}")
    if quote:
        problems.append(f"{path.name}: unterminated {quote} string")


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    common = WEB / "static" / "common.js"
    exported = set(EXPORT_RE.findall(common.read_text(encoding="utf-8")))
    for group in EXPORT_LIST_RE.findall(common.read_text(encoding="utf-8")):
        exported |= {n.strip().split()[-1] for n in group.split(",") if n.strip()}

    js_files = sorted((WEB / "static").glob("*.js"))
    html_files = sorted(WEB.glob("*.html"))

    for path in js_files + html_files:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".js":
            balanced(text, path, problems)

        # imports resolve to real exports
        for names, source in IMPORT_RE.findall(text):
            wanted = {n.strip() for n in names.split(",") if n.strip()}
            if source.endswith("common.js"):
                for name in sorted(wanted - exported):
                    problems.append(f"{path.name}: imports '{name}', not exported by common.js")
                body = text.split("}", 1)[1] if "}" in text else text
                for name in sorted(wanted & exported):
                    # \b only works either side of a word character, so names
                    # like `$` and `$$` need a bare substring search instead.
                    pattern = (rf"\b{re.escape(name)}\b" if name[0].isalnum()
                               else re.escape(name))
                    if not re.search(pattern, body):
                        notes.append(f"{path.name}: imports '{name}' but never uses it")

        # referenced static assets exist
        for ref in SRC_RE.findall(text):
            if not (WEB / ref.lstrip("/")).exists():
                problems.append(f"{path.name}: references missing asset {ref}")

    # every $('#id') exists in the page(s) that load the script
    for script, pages in PAGES.items():
        script_path = WEB / script
        if not script_path.exists():
            problems.append(f"missing script {script}")
            continue
        wanted = set(ID_RE.findall(script_path.read_text(encoding="utf-8")))
        available: set[str] = set()
        for page in pages:
            available |= set(HTML_ID_RE.findall((WEB / page).read_text(encoding="utf-8")))
        for missing in sorted(wanted - available):
            problems.append(f"{script}: uses #{missing}, absent from {', '.join(pages)}")

    for note in notes:
        print(f"  note:  {note}")
    for problem in problems:
        print(f"  ERROR: {problem}")
    checked = len(js_files) + len(html_files)
    print(f"\n  checked {checked} files: {len(problems)} error(s), {len(notes)} note(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
