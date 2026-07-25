#!/usr/bin/env python3
"""Check that the "Page X of Y" footers agree with the compiled PDF.

The CAS class writes the page total before deferred floats are flushed, so
the footer silently under-reports Y as soon as a table or figure lands on a
page of its own -- which is how a submission ends up claiming "Page 17 of
16". main.tex therefore fixes the total by hand; this script fails if that
constant, the footers and the PDF ever disagree.
"""
from __future__ import annotations

import os
import re
import sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"Page (\d+) of (\d+)")


def check(pdf: str) -> list[str]:
    doc = fitz.open(pdf)
    n_pdf = len(doc)
    seen, bad = [], []
    for i, page in enumerate(doc):
        m = PAT.search(page.get_text())
        seen.append((i + 1, m.groups() if m else None))
    numbered = [(p, g) for p, g in seen if g]
    if not numbered:
        return [f"{pdf}: no page footers found"]
    totals = {g[1] for _, g in numbered}
    if len(totals) > 1:
        bad.append(f"footers disagree on the total: {sorted(totals)}")
    total = int(next(iter(totals)))
    unnumbered = n_pdf - len(numbered)
    if total != len(numbered):
        bad.append(f"footer total {total} but {len(numbered)} numbered pages "
                   f"({n_pdf} in the PDF, {unnumbered} unnumbered)")
    for p, g in numbered:
        if int(g[0]) > total:
            bad.append(f"PDF page {p} claims page {g[0]} of {total}")
    expected = [i for i in range(1, len(numbered) + 1)]
    if [int(g[0]) for _, g in numbered] != expected:
        bad.append("numbered pages are not consecutive from 1")
    return bad


def main():
    pdfs = sys.argv[1:] or [os.path.join(ROOT, "paper", "KBS", "main.pdf")]
    fail = False
    for pdf in pdfs:
        if not os.path.exists(pdf):
            print(f"missing {pdf}")
            fail = True
            continue
        bad = check(pdf)
        if bad:
            fail = True
            print(f"FAIL {os.path.relpath(pdf, ROOT)}")
            for b in bad:
                print(f"  {b}")
        else:
            doc = fitz.open(pdf)
            print(f"ok   {os.path.relpath(pdf, ROOT)}: {len(doc)} PDF pages, "
                  f"footers consistent")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
