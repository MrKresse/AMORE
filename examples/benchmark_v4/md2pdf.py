# -*- coding: utf-8 -*-
"""Dependency-free Markdown -> PDF (matplotlib only). Text/tables on text pages
(monospace, preserves pipe-tables), each ![](img) on its own full page so the
panels render. Usage: python md2pdf.py in.md out.pdf"""
import sys, os, re, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

md_path, out_pdf = sys.argv[1], sys.argv[2]
base = os.path.dirname(os.path.abspath(md_path))
raw = open(md_path, encoding="utf-8").read().split("\n")
img_re = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')

segments, buf = [], []
for ln in raw:
    m = img_re.search(ln)
    if m:
        if buf: segments.append(("text", buf)); buf = []
        segments.append(("img", m.group(1)))
    else:
        buf.append(ln)
if buf: segments.append(("text", buf))

PER = 60          # lines per text page
WRAP = 115        # wrap width (monospace)
with PdfPages(out_pdf) as pdf:
    for kind, payload in segments:
        if kind == "text":
            wrapped = []
            for ln in payload:
                ln = ln.rstrip()
                if ln == "" or len(ln) <= WRAP or ln.lstrip().startswith("|"):
                    wrapped.append(ln)                     # keep tables/short lines intact
                else:
                    wrapped += textwrap.wrap(ln, WRAP)
            # drop leading/trailing blank-only runs, skip wholly-empty pages
            for i in range(0, len(wrapped), PER):
                chunk = wrapped[i:i + PER]
                if not any(c.strip() for c in chunk): continue
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.05, 0.97, "\n".join(chunk), va="top", ha="left",
                         family="monospace", fontsize=8.0)
                pdf.savefig(fig); plt.close(fig)
        else:
            p = payload if os.path.isabs(payload) else os.path.normpath(os.path.join(base, payload))
            if not os.path.exists(p):
                fig = plt.figure(figsize=(8.5, 2)); fig.text(0.05, 0.5, f"[missing image: {payload}]",
                                                             family="monospace", fontsize=9); pdf.savefig(fig); plt.close(fig)
                continue
            img = mpimg.imread(p); h, w = img.shape[:2]; ar = w / h
            W = 10.5; H = W / ar
            if H > 13.8: H = 13.8; W = H * ar
            fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
            pdf.savefig(fig, dpi=100); plt.close(fig)
print("wrote", out_pdf)
