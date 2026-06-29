#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flatten.py
==========
Produce a single self-contained submission file `main_submission.tex` from the
modular `main.tex`, as required by Springer Nature (no \\input of other .tex
files in the submitted manuscript).  Each `\\input{sections/...}` is replaced by
the verbatim contents of the corresponding section file, and the
`\\bibliography{...}` command is replaced by the contents of the generated
`main.bbl`, so the result needs only `pdflatex` (no bibtex) and the figures.

Run from anywhere:
    python flatten.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.normpath(os.path.join(HERE, "..", "article"))

src = os.path.join(ART, "main.tex")
out = os.path.join(ART, "main_submission.tex")

with open(src, encoding="utf-8") as f:
    lines = f.readlines()

input_re = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
bib_re = re.compile(r"^\s*\\bibliography\{[^}]+\}\s*$")

result = []
for ln in lines:
    m = input_re.match(ln)
    if m:
        rel = m.group(1)
        if not rel.endswith(".tex"):
            rel += ".tex"
        path = os.path.join(ART, rel)
        with open(path, encoding="utf-8") as g:
            body = g.read()
        result.append(f"%% ===== begin {rel} =====\n")
        result.append(body.rstrip("\n") + "\n")
        result.append(f"%% ===== end {rel} =====\n")
        continue
    if bib_re.match(ln):
        bbl = os.path.join(ART, "main.bbl")
        if os.path.exists(bbl):
            with open(bbl, encoding="utf-8") as g:
                result.append("%% ===== inlined bibliography (main.bbl) =====\n")
                result.append(g.read().rstrip("\n") + "\n")
        else:
            result.append(ln)  # fall back to \bibliography if no .bbl yet
        continue
    result.append(ln)

with open(out, "w", encoding="utf-8") as f:
    f.writelines(result)

print(f"Wrote {out} ({sum(len(x) for x in result)} chars)")
